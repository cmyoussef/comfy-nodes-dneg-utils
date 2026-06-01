# src\comfyui_remote\custom_nodes\ddcoloring\node_sequence.py
"""ddcoloring.node_sequence

ComfyUI node for DDColor *image sequence* colorization.

This node:
- Accepts a JSON list of image file paths
- Loads the images from disk itself
- Runs temporal-consistent colorization (optical-flow warp + blend)
- Saves outputs to an output folder
- Returns:
    1) JSON list of output file paths
    2) A printf-style filename pattern (e.g. /out/frame_%06d.png)

Note:
- If no reference image path is provided, the first processed frame becomes the
  palette reference automatically (auto-reference).
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import cv2
import torch

from comfy_api.latest import io

from ._progress import report_progress
from .core import ColorizationPipeline, VideoColorizationPipeline


def _parse_paths(images_json: str) -> List[str]:
    """Parse a JSON list of paths (with a few robust fallbacks)."""
    s = (images_json or "").strip()
    if not s:
        return []

    # JSON list (preferred)
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(p) for p in obj if str(p).strip()]
        if isinstance(obj, str):
            return [obj]
    except Exception:
        pass

    # Newline-separated list fallback
    if "\n" in s:
        return [ln.strip() for ln in s.splitlines() if ln.strip()]

    # Single path fallback
    return [s]


# Cache pipelines to avoid re-loading weights on every run.
_pipeline_cache: dict[str, ColorizationPipeline] = {}


def _get_pipeline(
    model_path: str,
    model_size: str,
    input_size: int,
    device: str,
) -> ColorizationPipeline:
    abs_path = os.path.abspath(os.path.expanduser(model_path))
    key = f"{abs_path}|{device}|{model_size}|{int(input_size)}"
    pipe = _pipeline_cache.get(key)
    if pipe is None:
        pipe = ColorizationPipeline(
            model_path=abs_path,
            model_size=model_size,
            input_size=int(input_size),
            device=device,
        )
        _pipeline_cache[key] = pipe
    return pipe


class DDColorSequenceNode(io.ComfyNode):
    """DDColor Colorization Node for ComfyUI (sequence / temporal-consistent)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DDColorSequenceNode",
            display_name="DDColor Sequence (Temporal Consistent)",
            category="image/color",
            inputs=[
                io.String.Input(
                    "images_json",
                    default="[]",
                    multiline=True,
                    tooltip='JSON list of input image file paths (e.g. ["/path/f0001.png", ...])',
                ),
                io.String.Input(
                    "output_dir",
                    default="",
                    multiline=False,
                    tooltip="Directory to write colorized frames into.",
                ),
                io.String.Input(
                    "model_path",
                    default="",
                    multiline=False,
                    tooltip="Path to DDColor model weights (.pth/.pt)",
                ),
                io.Int.Input(
                    "input_size",
                    default=512,
                    min=256,
                    max=1024,
                    step=64,
                    tooltip="Model input size (higher = better quality, slower)",
                ),
                io.Combo.Input(
                    "model_size",
                    options=["large", "tiny"],
                    default="large",
                    tooltip="Model variant (ignored if checkpoint has model_config)",
                ),
                io.Float.Input(
                    "temporal_strength",
                    default=0.8,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Temporal smoothing strength (0 disables; 1 = max smoothing).",
                ),
                io.Boolean.Input(
                    "edge_weighting",
                    default=True,
                    tooltip="Reduce temporal blending near edges to preserve detail.",
                ),
                io.Float.Input(
                    "reference_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Reference palette strength (0 disables palette steering).",
                ),
                io.Combo.Input(
                    "reference_method",
                    options=["meanstd", "cov"],
                    default="meanstd",
                    tooltip="Reference match method.",
                ),
                io.String.Input(
                    "reference_image_path",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Optional reference image path. If empty, auto-reference uses first frame.",
                ),
                io.String.Input(
                    "output_prefix",
                    default="frame_",
                    multiline=False,
                    optional=True,
                    tooltip="Prefix used for output filenames.",
                ),
                io.String.Input(
                    "output_ext",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Output extension (e.g. .png). Leave empty to use first input's extension.",
                ),
                io.Int.Input(
                    "start_index",
                    default=0,
                    min=0,
                    max=10_000_000,
                    step=1,
                    optional=True,
                    tooltip="Starting index for output numbering.",
                ),
            ],
            outputs=[
                io.String.Output("output_paths_json"),
                io.String.Output("output_pattern"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images_json: str,
        output_dir: str,
        model_path: str,
        input_size: int,
        model_size: str,
        temporal_strength: float,
        edge_weighting: bool,
        reference_strength: float,
        reference_method: str,
        reference_image_path: str = "",
        output_prefix: str = "frame_",
        output_ext: str = "",
        start_index: int = 0,
    ) -> io.NodeOutput:
        if not model_path:
            raise ValueError("model_path is required")

        paths = _parse_paths(images_json)
        if not paths:
            raise ValueError("images_json did not contain any paths")

        if not output_dir:
            raise ValueError("output_dir is required")

        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(output_dir, exist_ok=True)

        # Resolve output extension
        first_ext = os.path.splitext(paths[0])[1]
        ext = (output_ext or first_ext or ".png").strip()
        if not ext.startswith("."):
            ext = "." + ext

        # Determine padding (at least 4 digits; often 6)
        pad = max(6, len(str(start_index + len(paths))))
        pattern = os.path.join(output_dir, f"{output_prefix}%0{pad}d{ext}")

        # Load reference image if provided
        ref_bgr_u8: Optional[object] = None
        ref_path = (reference_image_path or "").strip()
        if ref_path and float(reference_strength) > 0:
            ref_path = os.path.abspath(os.path.expanduser(ref_path))
            ref_bgr_u8 = cv2.imread(ref_path, cv2.IMREAD_COLOR)
            if ref_bgr_u8 is None:
                raise ValueError(f"Failed to read reference image: {ref_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = _get_pipeline(model_path, model_size, input_size, device)

        video = VideoColorizationPipeline(
            pipe,
            temporal_strength=float(temporal_strength),
            use_edge_weighting=bool(edge_weighting),
            reference_bgr_u8=ref_bgr_u8,
            reference_strength=float(reference_strength),
            reference_method=str(reference_method),
            auto_reference=(ref_bgr_u8 is None),
        )

        reference_mode = "explicit" if ref_bgr_u8 is not None else "auto"

        out_paths: List[str] = []
        for i, in_path in enumerate(paths):
            in_path = os.path.abspath(os.path.expanduser(str(in_path)))
            img_bgr_u8 = cv2.imread(in_path, cv2.IMREAD_COLOR)
            if img_bgr_u8 is None:
                raise ValueError(f"Failed to read input image: {in_path}")

            out_bgr_u8 = video.colorize_frame(img_bgr_u8)
            out_path = pattern % (start_index + i)

            ok = cv2.imwrite(out_path, out_bgr_u8)
            if not ok:
                raise ValueError(f"Failed to write output image: {out_path}")

            out_paths.append(out_path)
            report_progress(i + 1, len(paths))

        info = "\n".join(
            [
                f"Input frames: {len(paths)}",
                f"Output dir: {output_dir}",
                f"Output pattern: {pattern}",
                f"Device: {device}",
                f"Reference mode: {reference_mode}",
                f"Temporal strength: {float(temporal_strength):.3f}",
                f"Edge weighting: {'ON' if edge_weighting else 'OFF'}",
            ]
        )

        return io.NodeOutput(json.dumps(out_paths), pattern, info)

