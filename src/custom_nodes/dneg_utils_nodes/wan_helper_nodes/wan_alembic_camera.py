# src\comfyui_remote\custom_nodes\wan_helper_nodes\wan_alembic_camera.py
"""
Wan helper nodes for ComfyUI.
Loads camera data from Alembic (.abc) files and outputs WAN_CAMERA_EMBEDDING
tensors (Plucker ray embeddings) for the Wan video generation model.

Pipeline (mirrors comfy_extras.nodes_camera_trajectory.WanCameraEmbedding):

    .abc file
      -> extract_camera_data (with axis + translation conversion)
      -> cam_params (length, 23)
      -> process_pose_params -> Plucker rays (F, H, W, 6)
      -> pad first frame x4, repack into latent temporal groups
      -> WAN_CAMERA_EMBEDDING tensor (B, C*4, F//4, H, W)
"""
from __future__ import annotations

import os

import torch

import comfy.model_management
from comfy_api.latest import io
from comfy_extras.nodes_camera_trajectory import process_pose_params

from .alembic_camera import (
    AXIS_CONVENTIONS,
    HAS_ALEMBIC,
    extract_camera_data,
    list_cameras_in_abc,
    require_alembic,
)


class WanHelper_WanAlembicCamera(io.ComfyNode):
    """
    Load camera animation from an Alembic (.abc) file and produce Plucker
    ray embeddings for the Wan video model.

    Connects to WanCameraImageToVideo's `camera_conditions` input.

    Defaults are tuned for Maya-exported Alembic files:
      - axis_convention: "maya_to_opencv" (flips local Y and Z to match Wan
        training convention)
      - translation_scale: 0.0 -> auto-fit max translation to target_motion_range
      - target_motion_range: 1.5 (matches Wan preset `base_T_norm`)
      - use_preset_intrinsics: True (fx=fy=0.5, ~90deg FOV; what the built-in
        WanCameraEmbedding presets use)

    Override `use_preset_intrinsics` only if your camera's lens is genuinely
    different and you know Wan can handle it. Override `translation_scale`
    only if you want to preserve absolute scale across runs (rare).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        wan_camera_embedding = io.Custom("WAN_CAMERA_EMBEDDING")
        return io.Schema(
            node_id="WanHelper_WanAlembicCamera",
            display_name="WanHelper: Wan Alembic Camera Embedding",
            category="WanHelper/camera",
            description=cls.__doc__ or "",
            inputs=[
                io.String.Input(
                    "abc_file",
                    default="",
                    tooltip="Path to the Alembic (.abc) file containing camera data.",
                ),
                io.String.Input(
                    "camera_name",
                    default="",
                    tooltip=(
                        "Camera name (or substring, case-insensitive) in the Alembic "
                        "hierarchy. Leave empty to use the first camera found."
                    ),
                ),
                io.Int.Input(
                    "width", default=832, min=16, max=16384, step=16,
                    tooltip="Output width for the Plucker embedding (pixel-space).",
                ),
                io.Int.Input(
                    "height", default=480, min=16, max=16384, step=16,
                    tooltip="Output height for the Plucker embedding (pixel-space).",
                ),
                io.Int.Input(
                    "length", default=81, min=5, max=16384, step=4,
                    tooltip=(
                        "Number of output frames. Must satisfy (length - 1) % 4 == 0 "
                        "to align with Wan's 4x temporal VAE compression (81, 85, ...)."
                    ),
                ),
                io.Int.Input(
                    "start_frame", default=0, min=0, max=100000, step=1,
                    tooltip=(
                        "Starting frame in the Alembic timeline. Frame 0 is repeated "
                        "4x internally, so this should be a stable pose."
                    ),
                ),
                io.Float.Input(
                    "fps", default=24.0, min=1.0, max=120.0, step=0.001,
                    tooltip="Frames per second for Alembic time sampling.",
                ),
                io.Combo.Input(
                    "axis_convention",
                    options=list(AXIS_CONVENTIONS),
                    default="maya_to_opencv",
                    tooltip=(
                        "Coordinate-basis conversion applied to each c2w before "
                        "Plucker embedding. Maya/Alembic exports use +Y up, -Z "
                        "forward; Wan expects OpenCV (+Y down, +Z forward). Use "
                        "'maya_to_opencv' (default). Try 'passthrough' only to "
                        "compare against raw behavior."
                    ),
                ),
                io.Float.Input(
                    "translation_scale",
                    default=0.0, min=0.0, max=1000.0, step=0.0001,
                    tooltip=(
                        "Explicit multiplier for camera translations. 0 = auto-fit "
                        "max translation to 'target_motion_range'. Use a fixed value "
                        "only if you want consistent absolute scale across shots."
                    ),
                ),
                io.Float.Input(
                    "target_motion_range",
                    default=1.5, min=0.01, max=100.0, step=0.01,
                    tooltip=(
                        "Target max |T_i - T_0| after auto-scaling. Wan presets use "
                        "1.5 as the base translation norm; values in [0.5, 3.0] are "
                        "usually safe."
                    ),
                ),
                io.Boolean.Input(
                    "use_preset_intrinsics",
                    default=True,
                    tooltip=(
                        "If True, override the Alembic lens with fx=fy=0.5 "
                        "(cx=cy=0.5), matching the built-in Wan presets. If False, "
                        "derive normalized intrinsics from the Alembic focal length "
                        "and aperture. The preset is usually the better choice."
                    ),
                ),
                io.Float.Input(
                    "fx", default=0.0, min=0.0, max=10.0, step=0.000001,
                    optional=True,
                    tooltip="Normalized focal X override. 0 = use computed value.",
                ),
                io.Float.Input(
                    "fy", default=0.0, min=0.0, max=10.0, step=0.000001,
                    optional=True,
                    tooltip="Normalized focal Y override. 0 = use computed value.",
                ),
                io.Float.Input(
                    "cx", default=0.0, min=0.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Normalized principal point X override. 0 = use computed value.",
                ),
                io.Float.Input(
                    "cy", default=0.0, min=0.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Normalized principal point Y override. 0 = use computed value.",
                ),
            ],
            outputs=[
                wan_camera_embedding.Output(
                    "camera_embedding",
                    tooltip="Plucker ray embedding for WanCameraImageToVideo.",
                ),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("length"),
                io.String.Output(
                    "info",
                    tooltip=(
                        "Diagnostics: source camera path, intrinsics, translation "
                        "scaling factor, max rotation delta, and any warnings. Read "
                        "this if the conditioning looks wrong."
                    ),
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, abc_file: str, camera_name: str, length: int, **kwargs):
        if not HAS_ALEMBIC:
            return (
                "Alembic Python bindings (alembicAbc) are not installed. "
                "This node needs the VFX Alembic wrapper that exposes Abc and AbcGeom."
            )

        if (length - 1) % 4 != 0:
            return (
                f"length={length} must satisfy (length - 1) % 4 == 0 "
                f"(valid: 1, 5, 9, ..., 81, 85, ...). The step=4 UI control "
                f"starts from 1, so use 81 for a default Wan 81-frame clip."
            )

        if abc_file:
            if not os.path.exists(abc_file):
                return f"Alembic file not found: {abc_file}"
            try:
                available = list_cameras_in_abc(abc_file)
                if not available:
                    return f"No cameras found in {abc_file}"
                if camera_name:
                    matches = [c for c in available if camera_name.lower() in c.lower()]
                    if not matches:
                        return (
                            f"Camera '{camera_name}' not found in {abc_file}. "
                            f"Available: {available}"
                        )
            except Exception as e:  # pragma: no cover - runtime Alembic errors
                return f"Error reading Alembic file: {e}"

        return True

    @classmethod
    def execute(
        cls,
        abc_file: str,
        camera_name: str,
        width: int,
        height: int,
        length: int,
        start_frame: int,
        fps: float,
        axis_convention: str,
        translation_scale: float,
        target_motion_range: float,
        use_preset_intrinsics: bool,
        fx: float = 0.0,
        fy: float = 0.0,
        cx: float = 0.0,
        cy: float = 0.0,
    ) -> io.NodeOutput:
        require_alembic()

        cam_params, stats, cam_path = extract_camera_data(
            abc_path=abc_file,
            camera_name=camera_name,
            length=length,
            start_frame=start_frame,
            fps=fps,
            axis_convention=axis_convention,
            translation_scale=translation_scale,
            target_motion_range=target_motion_range,
            use_preset_intrinsics=use_preset_intrinsics,
            fx_override=fx,
            fy_override=fy,
            cx_override=cx,
            cy_override=cy,
        )

        # Mirror comfy_extras.nodes_camera_trajectory.WanCameraEmbedding:
        # process_pose_params returns a (F, H, W, 6) tensor on CPU.
        control_camera_video = process_pose_params(cam_params, width=width, height=height)
        control_camera_video = (
            control_camera_video.permute([3, 0, 1, 2])
            .unsqueeze(0)
            .to(device=comfy.model_management.intermediate_device())
        )

        # Repeat the first frame 4x so the latent packing sees a stable anchor
        # block. Wan's temporal VAE compresses 4 pixel frames -> 1 latent frame.
        control_camera_video = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:],
            ],
            dim=2,
        ).transpose(1, 2)

        # Repack into (B, C*4, F//4, H, W) latent-temporal layout.
        b, f, c, h, w = control_camera_video.shape
        control_camera_video = (
            control_camera_video.contiguous()
            .view(b, f // 4, 4, c, h, w)
            .transpose(2, 3)
        )
        control_camera_video = (
            control_camera_video.contiguous()
            .view(b, f // 4, c * 4, h, w)
            .transpose(1, 2)
        )

        info_lines = [
            stats.to_info_string(cam_path, start_frame),
            f"Output resolution: {width} x {height}",
            f"Output tensor shape: {tuple(control_camera_video.shape)}",
        ]

        return io.NodeOutput(
            control_camera_video,
            width,
            height,
            length,
            "\n".join(info_lines),
        )
