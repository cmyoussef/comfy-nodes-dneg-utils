"""
deep_hdr.core

Wrapper for deep_hdr.core.deep_hdr.Run.

Manages temporary EXR files, saturation mask computation, and the Run
lifecycle.  Uses OpenImageIO directly for temp file I/O so that no
gamma correction is applied to the preprocessing step.

Only imports from deep_hdr.core.*  — never from deep_hdr.gizmos,
deep_hdr.config, nuke, or nukebridge.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Callable, List, Optional, Tuple

import numpy as np
import OpenImageIO as oiio


def _require_deep_hdr():
    """Import deep_hdr.core modules, raising a clear error if unavailable."""
    try:
        from deep_hdr.core.deep_hdr import Run
        from deep_hdr.core.deep_hdr_pkg.pkg_utils.utils import get_saturated_regions
        return Run, get_saturated_regions
    except ImportError as exc:
        raise ImportError(
            "The 'deep_hdr' package is not available in this runtime environment.\n"
            "Ensure 'deep_hdr' is listed in comfyui-remote's dneg.json deployment_tools "
            "with target 'ml_cv', then restart the ComfyUI server.\n"
            f"Original error: {exc}"
        ) from exc


def _load_exr(path: str) -> np.ndarray:
    """Load an EXR as a float32 HxWx3 numpy array."""
    buf = oiio.ImageBuf(path)
    if buf.has_error:
        raise ValueError(f"Failed to open EXR '{path}': {buf.geterror()}")
    pixels = buf.get_pixels(oiio.FLOAT)
    if pixels is None or pixels.size == 0:
        raise ValueError(f"Empty or unreadable EXR: {path}")
    if pixels.ndim == 2:
        pixels = pixels[:, :, np.newaxis]
    if pixels.shape[2] < 3:
        pixels = np.concatenate([pixels] * 3, axis=2)[:, :, :3]
    elif pixels.shape[2] > 3:
        pixels = pixels[:, :, :3]
    return pixels.astype(np.float32)


def _write_exr(pixels: np.ndarray, path: str) -> None:
    """Write a float32 HxWxC array to an EXR file with no gamma correction."""
    h, w, c = pixels.shape
    spec = oiio.ImageSpec(w, h, c, oiio.FLOAT)
    buf = oiio.ImageBuf(spec)
    buf.set_pixels(oiio.ROI(0, w, 0, h, 0, 1, 0, c), np.ascontiguousarray(pixels))
    if not buf.write(path):
        raise RuntimeError(f"Failed to write EXR '{path}': {buf.geterror()}")


def run_deep_hdr(
    input_paths: List[str],
    output_dir: str,
    weights_path: str,
    device: str,
    multiply: float,
    saturation_threshold: float,
    start_frame: int,
    output_prefix: str,
    report_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[str], str]:
    """
    Run DeepHDR highlight reconstruction on a list of input EXR frames.

    Workflow:
      1. Load each input EXR with OIIO (float32, 3-channel).
      2. Apply multiply factor (matches Nuke multiply_knob).
      3. Compute saturation mask via get_saturated_regions().
      4. Write pre-multiplied input and mask to temp EXRs (no gamma correction).
      5. Call deep_hdr.core.deep_hdr.Run(...).predict().
      6. Collect output EXRs written by predict() — these have gamma_correct=0.5
         baked in (values are H^0.5), matching the Nuke gizmo output.
      7. Clean up temp dir.

    Returns:
        (output_paths, output_pattern)
    """
    Run, get_saturated_regions = _require_deep_hdr()

    if not input_paths:
        raise ValueError("input_paths is empty — no frames to process")

    weights_path = os.path.abspath(os.path.expanduser(weights_path))
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}\n"
            "Default location: /jobs/ADGRE/ldev_pipe/nuke/ai/deep_hdr/deephdr/ldr2hdr.pth"
        )

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    n_frames = len(input_paths)
    pad = max(6, len(str(start_frame + n_frames - 1)))
    output_pattern = os.path.join(output_dir, f"{output_prefix}%0{pad}d.exr")

    # Progress: n_frames preprocessing ticks + 1 tick for inference
    total_steps = n_frames + 1

    tmpdir = tempfile.mkdtemp(prefix="cremote_deep_hdr_")
    try:
        input_pattern = os.path.join(tmpdir, "input_%06d.exr")
        mask_pattern = os.path.join(tmpdir, "mask_%06d.exr")

        for i, in_path in enumerate(input_paths):
            frame_num = start_frame + i
            in_path = os.path.abspath(os.path.expanduser(str(in_path)))
            if not os.path.isfile(in_path):
                raise FileNotFoundError(f"Input EXR not found: {in_path}")

            img_np = _load_exr(in_path)
            img_np = img_np * float(multiply)
            mask_np = get_saturated_regions(img_np, th=float(saturation_threshold))

            _write_exr(img_np, input_pattern % frame_num)
            _write_exr(mask_np, mask_pattern % frame_num)

            if report_progress:
                report_progress(i + 1, total_steps)

        frame_range = (start_frame, start_frame + n_frames - 1)

        inference = Run(
            input=input_pattern,
            output=output_pattern,
            mask=mask_pattern,
            weights=weights_path,
            device_menu=device,
            frame_range=frame_range,
        )
        inference.predict()

        if report_progress:
            report_progress(total_steps, total_steps)

        out_paths: List[str] = []
        for i in range(n_frames):
            out_path = output_pattern % (start_frame + i)
            if not os.path.isfile(out_path):
                raise RuntimeError(
                    f"Expected output EXR was not produced: {out_path}\n"
                    "The inference may have failed silently — check server logs."
                )
            out_paths.append(out_path)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return out_paths, output_pattern
