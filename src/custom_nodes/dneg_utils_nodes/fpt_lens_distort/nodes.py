# src\comfyui_remote\custom_nodes\fpt_lens_distort\nodes.py
"""
FPT Lens Distortion Nodes
ComfyUI nodes for lens distortion and undistortion.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from comfy_api.latest import io

from ._progress import report_progress
from .fpt_lens_ops import (
    FPTCameraCalib,
    FPTSTMapSpec,
    apply_unbulge,
    build_opencv_maps,
    build_stmap_maps,
    build_unbulge_maps,
    compute_valid_bbox,
    cv_border_mode,
    cv_interpolation,
    generate_stmap_from_calib,
    get_available_cameras,
    invert_maps,
    parse_camera_from_json,
    read_calib_file,
    remap_image,
    require_cv2,
)


# Module-level maps cache (replaces instance-level cache since V3 execute is classmethod).
_maps_cache: Dict[Any, Any] = {}


class FPTLensDistortUndistort(io.ComfyNode):
    """
    Apply lens undistort/distort using JSON camera calibration (OpenCV model).

    Outputs:
      - IMAGE: processed image [B,H,W,C] float32 [0..1]
      - MASK: invalid mask [B,H,W] float32 where 1 = invalid/outside
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FPT_LensDistortUndistort",
            display_name="FPT: Lens Distort/Undistort (JSON)",
            category="fpt/image/geometry",
            inputs=[
                io.Image.Input("image", tooltip="Input images [B,H,W,C] float32 in [0..1]."),
                io.String.Input("calib_file", default="", tooltip="Path to JSON calibration file."),
                io.String.Input("camera_name", default="top", tooltip="Camera name matching metadata.camera or metadata.name in JSON."),
                io.Combo.Input("mode", options=["undistort", "distort"], default="undistort", tooltip="undistort: remove lens distortion. distort: apply lens distortion."),
                io.Float.Input("balance", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="0 = crop to valid pixels only, 1 = keep all pixels with black borders."),
                io.Boolean.Input("scale_to_input", default=True, tooltip="Scale intrinsics if input resolution differs from calibration resolution."),
                io.Boolean.Input("crop_valid", default=False, tooltip="Crop output to valid region (undistort mode only)."),
                io.Combo.Input("border", options=["constant", "replicate", "reflect"], default="constant", tooltip="Border handling for pixels outside source image."),
                io.Combo.Input("interpolation", options=["linear", "nearest", "cubic", "lanczos"], default="linear", tooltip="Interpolation method for resampling."),
            ],
            outputs=[
                io.Image.Output("image", tooltip="Processed image [B,H,W,C] float32 [0..1]."),
                io.Mask.Output("invalid_mask", tooltip="Invalid mask [B,H,W] float32 (1 = invalid/outside)."),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def validate_inputs(cls, calib_file, camera_name, **kwargs):
        try:
            require_cv2()
        except Exception as e:
            return str(e)

        try:
            text, _ = read_calib_file(str(calib_file))
            _ = parse_camera_from_json(text, str(camera_name))
        except Exception as e:
            return f"Invalid calibration: {e}"

        return True

    @classmethod
    def fingerprint_inputs(cls, calib_file, **kwargs):
        try:
            _, sha = read_calib_file(str(calib_file))
            return sha
        except Exception:
            return "file_missing"

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        calib_file: str,
        camera_name: str,
        mode: str,
        balance: float,
        scale_to_input: bool,
        crop_valid: bool,
        border: str,
        interpolation: str,
    ) -> io.NodeOutput:
        require_cv2()

        b, ih, iw, c = image.shape
        image_size = (int(iw), int(ih))

        border_mode = cv_border_mode(border)
        interp = cv_interpolation(interpolation)

        calib_text, file_sha = read_calib_file(calib_file)
        cam = parse_camera_from_json(calib_text, camera_name)

        # Build cached maps
        key = (
            file_sha, cam.name, image_size[0], image_size[1],
            round(balance, 4), scale_to_input,
            tuple(cam.dist.round(12).tolist()),
            tuple(cam.K.flatten().round(12).tolist()),
        )

        if key in _maps_cache:
            map_ud_1, map_ud_2, map_du_1, map_du_2, roi = _maps_cache[key]
        else:
            maps = build_opencv_maps(cam, image_size, balance, scale_to_input)
            _maps_cache[key] = maps
            map_ud_1, map_ud_2, map_du_1, map_du_2, roi = maps

        if mode == "undistort":
            map1, map2 = map_ud_1, map_ud_2
            roi_xywh = roi
        else:
            map1, map2 = map_du_1, map_du_2
            roi_xywh = (0, 0, iw, ih)

        out_images: List[torch.Tensor] = []
        out_masks: List[torch.Tensor] = []

        for i in range(b):
            img_np = image[i].detach().cpu().numpy().astype(np.float32)
            img_np = np.clip(img_np, 0.0, 1.0)

            out_np, invalid_np = remap_image(img_np, map1, map2, border_mode, interp)

            if crop_valid and mode == "undistort":
                x, y, rw, rh = roi_xywh
                if rw > 0 and rh > 0:
                    out_np = out_np[y:y+rh, x:x+rw]
                    invalid_np = invalid_np[y:y+rh, x:x+rw]

            out_images.append(torch.from_numpy(np.ascontiguousarray(out_np)))
            out_masks.append(torch.from_numpy(np.ascontiguousarray(invalid_np)))
            report_progress(i + 1, b)

        out_image = torch.stack(out_images, dim=0).float()
        out_mask = torch.stack(out_masks, dim=0).float()

        info = "\n".join(
            [
                f"Camera: {cam.name}",
                f"Mode: {mode}",
                f"Image size: {image_size[0]}x{image_size[1]}",
                f"Balance: {balance:.3f}",
                f"Scale to input: {scale_to_input}",
                f"Crop valid: {crop_valid}",
                f"Border: {border}",
                f"Interpolation: {interpolation}",
                f"ROI: {roi_xywh[0]}, {roi_xywh[1]}, {roi_xywh[2]}, {roi_xywh[3]}",
            ]
        )

        return io.NodeOutput(out_image, out_mask, info)


class FPTSTMapDistort(io.ComfyNode):
    """
    Apply distortion/undistortion using an STMap (UV remap image).

    STMap is a standard UV remap format where:
    - Each pixel stores the source UV coordinates
    - R/G channels (or B/A) contain U/V values
    - Values can be normalized [0..1] or absolute pixel coordinates
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FPT_STMapDistort",
            display_name="FPT: STMap Distort/Undistort",
            category="fpt/image/geometry",
            inputs=[
                io.Image.Input("image", tooltip="Input images [B,H,W,C] float32 in [0..1]."),
                io.Image.Input("stmap", tooltip="STMap image containing UV coordinates [B,H,W,C]."),
                io.Combo.Input("stmap_kind", options=["undistort", "distort"], default="undistort", tooltip="What the STMap produces when applied directly."),
                io.Combo.Input("mode", options=["undistort", "distort"], default="undistort", tooltip="Desired operation. If different from stmap_kind, map will be inverted."),
                io.Combo.Input("stmap_space", options=["uv_0_1", "pixel_xy"], default="uv_0_1", tooltip="uv_0_1: normalized [0..1], pixel_xy: absolute pixel coordinates."),
                io.Combo.Input("stmap_channels", options=["RG", "BA"], default="RG", tooltip="Which channels contain U/V coordinates."),
                io.Boolean.Input("stmap_resample", default=False, tooltip="Allow resizing STMap to match image size."),
                io.Boolean.Input("crop_valid", default=False, tooltip="Crop output to bounding box of valid pixels."),
                io.Combo.Input("border", options=["constant", "replicate", "reflect"], default="constant", tooltip="Border handling for pixels outside source image."),
                io.Combo.Input("interpolation", options=["linear", "nearest", "cubic", "lanczos"], default="linear", tooltip="Interpolation method for resampling."),
            ],
            outputs=[
                io.Image.Output("image", tooltip="Processed image [B,H,W,C] float32 [0..1]."),
                io.Mask.Output("invalid_mask", tooltip="Invalid mask [B,H,W] float32 (1 = invalid/outside)."),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        try:
            require_cv2()
        except Exception as e:
            return str(e)
        return True

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        stmap: torch.Tensor,
        stmap_kind: str,
        mode: str,
        stmap_space: str,
        stmap_channels: str,
        stmap_resample: bool,
        crop_valid: bool,
        border: str,
        interpolation: str,
    ) -> io.NodeOutput:
        require_cv2()

        b, ih, iw, c = image.shape
        sb = int(stmap.shape[0])
        image_size = (int(iw), int(ih))

        border_mode = cv_border_mode(border)
        interp = cv_interpolation(interpolation)

        spec = FPTSTMapSpec(
            kind=stmap_kind,
            space=stmap_space,
            channels=stmap_channels,
            resample=stmap_resample,
        )

        out_images: List[torch.Tensor] = []
        out_masks: List[torch.Tensor] = []

        for i in range(b):
            si = i if sb == b else 0
            st_np = stmap[si].detach().cpu().numpy().astype(np.float32)

            img_np = image[i].detach().cpu().numpy().astype(np.float32)
            img_np = np.clip(img_np, 0.0, 1.0)

            map1, map2 = build_stmap_maps(st_np, image_size, spec)

            if mode != spec.kind:
                map1, map2 = invert_maps(map1, map2)

            out_np, invalid_np = remap_image(img_np, map1, map2, border_mode, interp)

            if crop_valid:
                valid = 1.0 - invalid_np
                bbox = compute_valid_bbox(valid)
                if bbox is not None:
                    x, y, rw, rh = bbox
                    out_np = out_np[y:y+rh, x:x+rw]
                    invalid_np = invalid_np[y:y+rh, x:x+rw]

            out_images.append(torch.from_numpy(np.ascontiguousarray(out_np)))
            out_masks.append(torch.from_numpy(np.ascontiguousarray(invalid_np)))
            report_progress(i + 1, b)

        out_image = torch.stack(out_images, dim=0).float()
        out_mask = torch.stack(out_masks, dim=0).float()

        info = "\n".join(
            [
                f"Mode: {mode}",
                f"STMap kind: {stmap_kind}",
                f"STMap space: {stmap_space}",
                f"STMap channels: {stmap_channels}",
                f"Image size: {image_size[0]}x{image_size[1]}",
                f"STMap size: {int(stmap.shape[2])}x{int(stmap.shape[1])}",
                f"Resample: {stmap_resample}",
                f"Crop valid: {crop_valid}",
                f"Border: {border}",
                f"Interpolation: {interpolation}",
            ]
        )

        return io.NodeOutput(out_image, out_mask, info)


class FPTLensCalibInfo(io.ComfyNode):
    """
    Extract camera calibration info from JSON file.
    Outputs list of cameras and detailed parameters.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FPT_LensCalibInfo",
            display_name="FPT: Lens Calibration Info",
            category="fpt/image/geometry",
            inputs=[
                io.String.Input("calib_file", default="", tooltip="Path to JSON calibration file."),
            ],
            outputs=[
                io.String.Output("camera_list"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(cls, calib_file: str) -> io.NodeOutput:
        try:
            text, _ = read_calib_file(calib_file)
        except Exception as e:
            return io.NodeOutput("", f"Error: {e}")

        cameras = get_available_cameras(text)
        camera_list = ", ".join(cameras)

        info_lines = []
        for cam_name in cameras:
            try:
                cam = parse_camera_from_json(text, cam_name)
                info_lines.append(f"=== {cam.name} ===")
                info_lines.append(f"  Resolution: {cam.authored_size[0]} x {cam.authored_size[1]}")
                info_lines.append(f"  Focal: fx={cam.K[0,0]:.2f}, fy={cam.K[1,1]:.2f}")
                info_lines.append(f"  Principal: cx={cam.K[0,2]:.2f}, cy={cam.K[1,2]:.2f}")

                if len(cam.dist) >= 5:
                    info_lines.append(f"  Distortion: k1={cam.dist[0]:.6f}, k2={cam.dist[1]:.6f}, k3={cam.dist[4]:.6f}")
                    info_lines.append(f"  Tangential: p1={cam.dist[2]:.6f}, p2={cam.dist[3]:.6f}")
                if len(cam.dist) >= 8:
                    info_lines.append(f"  Rational: k4={cam.dist[5]:.6f}, k5={cam.dist[6]:.6f}, k6={cam.dist[7]:.6f}")
                info_lines.append("")
            except Exception as e:
                info_lines.append(f"=== {cam_name} === ERROR: {e}")
                info_lines.append("")

        return io.NodeOutput(camera_list, "\n".join(info_lines))


class FPTGenerateSTMap(io.ComfyNode):
    """
    Generate an STMap image from camera calibration.
    Useful for debugging, verification, or exporting to other software.

    The STMap encodes the UV remap coordinates:
    - R channel = U (horizontal) coordinate
    - G channel = V (vertical) coordinate
    - B channel = 0
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FPT_GenerateSTMap",
            display_name="FPT: Generate STMap from Calib",
            category="fpt/image/geometry",
            inputs=[
                io.String.Input("calib_file", default="", tooltip="Path to JSON calibration file."),
                io.String.Input("camera_name", default="top", tooltip="Camera name from calibration file."),
                io.Int.Input("width", default=1920, min=64, max=8192, tooltip="Output STMap width."),
                io.Int.Input("height", default=1080, min=64, max=8192, tooltip="Output STMap height."),
                io.Combo.Input("mode", options=["undistort", "distort"], default="undistort", tooltip="What operation this STMap performs when applied."),
                io.Float.Input("balance", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="0 = crop to valid, 1 = keep all with borders."),
                io.Boolean.Input("normalize", default=True, tooltip="True = UV in [0,1], False = pixel coordinates."),
            ],
            outputs=[
                io.Image.Output("stmap"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        calib_file: str,
        camera_name: str,
        width: int,
        height: int,
        mode: str,
        balance: float,
        normalize: bool,
    ) -> io.NodeOutput:
        require_cv2()

        calib_text, _ = read_calib_file(calib_file)
        cam = parse_camera_from_json(calib_text, camera_name)

        stmap_np = generate_stmap_from_calib(
            cam=cam,
            image_size=(width, height),
            mode=mode,
            balance=balance,
            scale_to_input=True,
            normalize=normalize,
        )

        stmap_tensor = torch.from_numpy(stmap_np).unsqueeze(0).float()

        info = "\n".join(
            [
                f"Camera: {cam.name}",
                f"Output size: {width}x{height}",
                f"Mode: {mode}",
                f"Balance: {balance:.3f}",
                f"Normalize: {normalize}",
            ]
        )

        return io.NodeOutput(stmap_tensor, info)


class FPTUnbulge(io.ComfyNode):
    """
    Unbulge - Shrink center to fix HMC wide-angle nose distortion.

    Shrinks the center of the image (nose area) while keeping edges intact.
    Uses smooth cosine falloff between inner and outer radius.

    Processing order: 1. Unbulge (shrink nose)  2. Scale (zoom to frame)
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FPT_Unbulge",
            display_name="FPT: Unbulge (HMC Defish)",
            category="fpt/image/geometry",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("strength", default=0.35, min=0.0, max=1.0, step=0.01, tooltip="Shrink amount. Higher = smaller nose."),
                io.Float.Input("center_x", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Horizontal center (0=left, 0.5=center, 1=right)."),
                io.Float.Input("center_y", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Vertical center (0=top, 0.5=center, 1=bottom)."),
                io.Float.Input("inner_radius", default=0.15, min=0.0, max=1.0, step=0.01, tooltip="Full effect inside this radius (fraction of image)."),
                io.Float.Input("outer_radius", default=0.5, min=0.1, max=1.5, step=0.01, tooltip="Effect fades to zero at this radius. Edges stay intact."),
                io.Float.Input("scale", default=1.0, min=0.5, max=2.0, step=0.01, tooltip="Uniform scale applied after unbulge. >1 = zoom in, <1 = zoom out."),
                io.Float.Input("scale_x", default=1.0, min=0.5, max=2.0, step=0.01, tooltip="Horizontal scale multiplier. >1 = stretch wider."),
                io.Float.Input("scale_y", default=1.0, min=0.5, max=2.0, step=0.01, tooltip="Vertical scale multiplier. >1 = stretch taller."),
                io.Combo.Input("interpolation", options=["lanczos", "cubic", "linear", "nearest"], default="lanczos", optional=True, tooltip="Resampling method. Lanczos is highest quality."),
                io.Combo.Input("border", options=["replicate", "constant", "reflect"], default="replicate", optional=True, tooltip="How to handle pixels at image borders."),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("effect_mask"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        strength: float,
        center_x: float,
        center_y: float,
        inner_radius: float,
        outer_radius: float,
        scale: float,
        scale_x: float,
        scale_y: float,
        interpolation: str = "lanczos",
        border: str = "replicate",
    ) -> io.NodeOutput:
        require_cv2()

        B, H, W, C = image.shape
        results = []
        masks = []

        for b in range(B):
            img_np = image[b].cpu().numpy().astype(np.float32)

            corrected, weight = apply_unbulge(
                image=img_np,
                strength=strength,
                center_x=center_x,
                center_y=center_y,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                scale=scale,
                scale_x=scale_x,
                scale_y=scale_y,
                interpolation=interpolation,
                border=border,
            )

            results.append(corrected)
            masks.append(weight)
            report_progress(b + 1, B)

        result_tensor = torch.from_numpy(np.stack(results, axis=0)).float()
        mask_tensor = torch.from_numpy(np.stack(masks, axis=0)).float()

        return io.NodeOutput(result_tensor, mask_tensor)
