# src\comfyui_remote\custom_nodes\wan_helper_nodes\position_relative.py
"""
World Position Relative Nodes

ComfyUI nodes for transforming world position AOV passes relative to
a locator or joint from an Alembic file.

This allows normalizing position passes so that a specific joint (e.g., head)
is always at the origin, regardless of character animation.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np
import torch

from comfy_api.latest import io

from ._progress import report_progress
from .alembic_locator import (
    HAS_ALEMBIC,
    require_alembic,
    get_locator_transform,
    get_locator_transforms_batch,
    list_locators_in_abc,
    LocatorTransform,
)


# Module-level transform cache (replaces instance-level cache since V3 execute is classmethod).
_transform_cache: dict[tuple, List[LocatorTransform]] = {}


def _get_cached_transforms(
    abc_file: str,
    locator_name: str,
    frames: List[int],
    fps: float,
) -> List[LocatorTransform]:
    """Get transforms with caching."""
    cache_key = (abc_file, locator_name, tuple(frames), fps)

    if cache_key not in _transform_cache:
        transforms = get_locator_transforms_batch(
            abc_file, locator_name, frames, fps
        )
        _transform_cache[cache_key] = transforms

    return _transform_cache[cache_key]


class WanHelper_WorldPositionToHeadRelative(io.ComfyNode):
    """
    Transform a world position AOV to be relative to a locator or joint from an Alembic file.

    This node takes a world position pass (P AOV) and transforms it so that the
    specified locator (e.g., head joint) is always at the origin. This is useful
    for creating position-based effects that should follow the head regardless of
    body animation.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanHelper_WorldPositionToHeadRelative",
            display_name="WanHelper: World Position to Head Relative",
            category="WanHelper/3d/position",
            inputs=[
                io.Image.Input(
                    "position_pass",
                    tooltip="World position AOV image [B,H,W,3] where RGB channels represent XYZ world coordinates.",
                ),
                io.String.Input(
                    "abc_file",
                    default="",
                    tooltip="Path to the Alembic (.abc) file containing the locator or joint hierarchy.",
                ),
                io.String.Input(
                    "locator_name",
                    default="head",
                    tooltip="Name of the locator or joint to use as the new origin. Partial matching is supported.",
                ),
                io.Int.Input(
                    "frame_start",
                    default=1,
                    min=0,
                    max=100000,
                    tooltip="Starting frame number for the position pass sequence.",
                ),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=1.0,
                    max=120.0,
                    step=0.001,
                    tooltip="Frames per second for Alembic time sampling.",
                ),
                io.Combo.Input(
                    "transform_mode",
                    options=["position_only", "position_and_rotation"],
                    default="position_only",
                    tooltip="position_only: Only subtract locator position. position_and_rotation: Apply full inverse transform.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Optional mask to limit transformation to specific areas.",
                ),
                io.Float.Input(
                    "position_scale",
                    default=1.0,
                    min=0.001,
                    max=1000.0,
                    step=0.001,
                    optional=True,
                    tooltip="Scale factor for the position values (useful for unit conversion).",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "position_pass",
                    tooltip="Transformed position pass with locator at origin [B,H,W,3].",
                ),
                io.String.Output(
                    "info",
                    tooltip="Information about the locator transform applied.",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        abc_file,
        locator_name,
        **kwargs,
    ):
        if not HAS_ALEMBIC:
            return (
                "Alembic library (PyAlembic) is not installed. "
                "Install it with: pip install alembic or conda install -c conda-forge alembic"
            )

        if abc_file:
            import os

            if not os.path.exists(abc_file):
                return f"Alembic file not found: {abc_file}"

            try:
                available = list_locators_in_abc(abc_file)
                matches = [p for p in available if locator_name.lower() in p.lower()]
                if not matches:
                    return (
                        f"Locator '{locator_name}' not found in {abc_file}. "
                        f"Available: {available[:5]}{'...' if len(available) > 5 else ''}"
                    )
            except Exception as e:
                return f"Error reading Alembic file: {e}"

        return True

    @classmethod
    def execute(
        cls,
        position_pass: torch.Tensor,
        abc_file: str,
        locator_name: str,
        frame_start: int,
        fps: float,
        transform_mode: str,
        mask: Optional[torch.Tensor] = None,
        position_scale: float = 1.0,
    ) -> io.NodeOutput:
        require_alembic()

        B, H, W, C = position_pass.shape

        if C < 3:
            raise ValueError(
                f"Position pass must have at least 3 channels (XYZ), got {C}"
            )

        frames = [frame_start + i for i in range(B)]
        transforms = _get_cached_transforms(abc_file, locator_name, frames, fps)

        output_frames = []
        info_lines = []

        for i in range(B):
            frame = frames[i]
            xform = transforms[i]

            pos_np = position_pass[i].detach().cpu().numpy().astype(np.float64)
            pos_np[:, :, :3] *= position_scale

            locator_pos = xform.position

            if transform_mode == "position_only":
                pos_np[:, :, 0] -= locator_pos[0]
                pos_np[:, :, 1] -= locator_pos[1]
                pos_np[:, :, 2] -= locator_pos[2]

                info_lines.append(
                    f"Frame {frame}: Offset by ({locator_pos[0]:.4f}, "
                    f"{locator_pos[1]:.4f}, {locator_pos[2]:.4f})"
                )

            else:  # position_and_rotation
                inv_matrix = np.linalg.inv(xform.full_matrix)
                h, w = pos_np.shape[:2]
                positions = pos_np[:, :, :3].reshape(-1, 3)

                ones = np.ones((positions.shape[0], 1), dtype=np.float64)
                positions_h = np.hstack([positions, ones])

                transformed = (inv_matrix @ positions_h.T).T
                transformed_xyz = transformed[:, :3]

                pos_np[:, :, :3] = transformed_xyz.reshape(h, w, 3)

                info_lines.append(
                    f"Frame {frame}: Full inverse transform applied "
                    f"(pos: {locator_pos[0]:.4f}, {locator_pos[1]:.4f}, {locator_pos[2]:.4f})"
                )

            if mask is not None:
                mask_np = (
                    mask[i].detach().cpu().numpy()
                    if i < mask.shape[0]
                    else mask[0].detach().cpu().numpy()
                )
                original = position_pass[i].detach().cpu().numpy().astype(np.float64)
                original[:, :, :3] *= position_scale
                mask_3d = mask_np[:, :, np.newaxis]
                pos_np[:, :, :3] = pos_np[:, :, :3] * mask_3d + original[:, :, :3] * (
                    1 - mask_3d
                )

            pos_np[:, :, :3] /= position_scale

            output_frames.append(torch.from_numpy(pos_np.astype(np.float32)))
            report_progress(i + 1, B)

        output = torch.stack(output_frames, dim=0)

        info = f"Locator: {transforms[0].name}\n"
        info += f"Transform mode: {transform_mode}\n"
        info += f"Frames processed: {B}\n"
        info += "\n".join(info_lines[:5])
        if len(info_lines) > 5:
            info += f"\n... and {len(info_lines) - 5} more frames"

        return io.NodeOutput(output, info)


class WanHelper_ListAlembicLocators(io.ComfyNode):
    """
    List all locators and joints available in an Alembic file.

    This is a utility node to help identify the correct locator name
    to use with the WorldPositionToHeadRelative node.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanHelper_ListAlembicLocators",
            display_name="WanHelper: List Alembic Locators",
            category="WanHelper/3d/position",
            inputs=[
                io.String.Input(
                    "abc_file",
                    default="",
                    tooltip="Path to the Alembic (.abc) file to inspect.",
                ),
            ],
            outputs=[
                io.String.Output("locator_list"),
            ],
        )

    @classmethod
    def validate_inputs(cls, abc_file, **kwargs):
        if not HAS_ALEMBIC:
            return (
                "Alembic library (PyAlembic) is not installed. "
                "Install it with: pip install alembic"
            )

        if abc_file:
            import os

            if not os.path.exists(abc_file):
                return f"Alembic file not found: {abc_file}"

        return True

    @classmethod
    def execute(cls, abc_file: str) -> io.NodeOutput:
        require_alembic()

        if not abc_file:
            return io.NodeOutput("No file specified")

        try:
            locators = list_locators_in_abc(abc_file)
            if not locators:
                return io.NodeOutput("No transform nodes found in file")

            result = f"Found {len(locators)} transform nodes:\n\n"
            result += "\n".join(locators)
            return io.NodeOutput(result)

        except Exception as e:
            return io.NodeOutput(f"Error: {e}")


class WanHelper_GetLocatorPosition(io.ComfyNode):
    """
    Get the world position of a locator or joint from an Alembic file at a specific frame.

    Useful for debugging or for passing the position to other nodes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanHelper_GetLocatorPosition",
            display_name="WanHelper: Get Locator Position",
            category="WanHelper/3d/position",
            inputs=[
                io.String.Input(
                    "abc_file",
                    default="",
                    tooltip="Path to the Alembic (.abc) file.",
                ),
                io.String.Input(
                    "locator_name",
                    default="head",
                    tooltip="Name of the locator or joint.",
                ),
                io.Int.Input(
                    "frame",
                    default=1,
                    min=0,
                    max=100000,
                    tooltip="Frame number to sample.",
                ),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=1.0,
                    max=120.0,
                    tooltip="Frames per second.",
                ),
            ],
            outputs=[
                io.Float.Output("x"),
                io.Float.Output("y"),
                io.Float.Output("z"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        abc_file: str,
        locator_name: str,
        frame: int,
        fps: float,
    ) -> io.NodeOutput:
        require_alembic()

        try:
            xform = get_locator_transform(abc_file, locator_name, frame, fps)

            info = (
                f"Locator: {xform.name}\n"
                f"Frame: {frame} (time: {xform.time:.4f}s)\n"
                f"Position: ({xform.position[0]:.6f}, {xform.position[1]:.6f}, {xform.position[2]:.6f})\n"
                f"Rotation Matrix:\n{xform.rotation_matrix}"
            )

            return io.NodeOutput(
                float(xform.position[0]),
                float(xform.position[1]),
                float(xform.position[2]),
                info,
            )

        except Exception as e:
            return io.NodeOutput(0.0, 0.0, 0.0, f"Error: {e}")


class WanHelper_NormalizePositionPass(io.ComfyNode):
    """
    Normalize a position pass by subtracting a fixed offset.

    Simpler version that does not require Alembic; it just subtracts a constant
    offset from the position values. Useful when you know the head position
    or want to manually specify the offset.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanHelper_NormalizePositionPass",
            display_name="WanHelper: Normalize Position Pass",
            category="WanHelper/3d/position",
            inputs=[
                io.Image.Input(
                    "position_pass",
                    tooltip="World position AOV image [B,H,W,3] where RGB = XYZ.",
                ),
                io.Float.Input(
                    "offset_x",
                    default=0.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.001,
                    tooltip="X offset to subtract.",
                ),
                io.Float.Input(
                    "offset_y",
                    default=0.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.001,
                    tooltip="Y offset to subtract.",
                ),
                io.Float.Input(
                    "offset_z",
                    default=0.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.001,
                    tooltip="Z offset to subtract.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Optional mask to limit offset to specific areas.",
                ),
            ],
            outputs=[
                io.Image.Output("position_pass"),
            ],
        )

    @classmethod
    def execute(
        cls,
        position_pass: torch.Tensor,
        offset_x: float,
        offset_y: float,
        offset_z: float,
        mask: Optional[torch.Tensor] = None,
    ) -> io.NodeOutput:
        B, H, W, C = position_pass.shape

        offset = torch.tensor(
            [offset_x, offset_y, offset_z],
            dtype=position_pass.dtype,
            device=position_pass.device,
        )

        result = position_pass.clone()
        result[:, :, :, :3] = result[:, :, :, :3] - offset

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(-1)

            result[:, :, :, :3] = (
                result[:, :, :, :3] * mask
                + position_pass[:, :, :, :3] * (1 - mask)
            )

        return io.NodeOutput(result)
