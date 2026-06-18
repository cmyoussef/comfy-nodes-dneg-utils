"""
deep_hdr.node

CREMOTE_DeepHDR — ComfyUI node for DeepHDR highlight reconstruction.

Wraps deep_hdr.core.deep_hdr.Run.  Accepts a JSON list of scene-linear EXR
paths, computes the saturation mask in Python, writes temporary EXRs for the
model, runs inference, and returns output EXR paths.

Behaviour matches the existing Nuke Bridge gizmo as closely as possible.
Output EXRs have gamma_correct=0.5 baked in (values are H^0.5), identical to
what Nuke reads back from the gizmo before any inverse-multiply Grade node.
"""
from __future__ import annotations

import json
import os
from typing import List

from comfy_api.latest import io

from ._progress import report_progress
from .core import run_deep_hdr

_DEFAULT_WEIGHTS = "/jobs/ADGRE/ldev_pipe/nuke/ai/deep_hdr/deephdr/ldr2hdr.pth"


def _parse_paths(images_json: str) -> List[str]:
    """Parse a JSON list of paths (with a few robust fallbacks)."""
    s = (images_json or "").strip()
    if not s:
        return []

    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(p) for p in obj if str(p).strip()]
        if isinstance(obj, str):
            return [obj]
    except Exception:
        pass

    if "\n" in s:
        return [ln.strip() for ln in s.splitlines() if ln.strip()]

    return [s]


class CREMOTE_DeepHDR(io.ComfyNode):
    """DeepHDR highlight reconstruction node (DNEG / ADGRE)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CREMOTE_DeepHDR",
            display_name="CREMOTE: DeepHDR (Highlight Reconstruction)",
            category="image/dneg",
            inputs=[
                io.String.Input(
                    "input_paths_json",
                    default="[]",
                    multiline=True,
                    tooltip=(
                        'JSON list of scene-linear input EXR paths, '
                        'e.g. ["/path/frame.1001.exr", "/path/frame.1002.exr"]. '
                        'Pixels with values >= saturation_threshold are treated as clipped.'
                    ),
                ),
                io.String.Input(
                    "output_dir",
                    default="",
                    multiline=False,
                    tooltip="Directory to write reconstructed EXR frames into.",
                ),
                io.String.Input(
                    "weights_path",
                    default=_DEFAULT_WEIGHTS,
                    multiline=False,
                    tooltip="Absolute path to the ldr2hdr.pth model weights file.",
                ),
                io.Combo.Input(
                    "device",
                    options=["cuda", "cpu"],
                    default="cuda",
                    tooltip="Inference device. Use 'cuda' for GPU (recommended).",
                ),
                io.Float.Input(
                    "multiply",
                    default=1.0,
                    min=0.0,
                    max=4.0,
                    step=0.01,
                    tooltip=(
                        "Pre-multiply applied to input before inference "
                        "(matches Nuke multiply_knob, default 1.0). "
                        "Note: the Nuke gizmo applies an inverse Grade to the output; "
                        "this node does not — divide output manually if multiply != 1.0."
                    ),
                ),
                io.Float.Input(
                    "saturation_threshold",
                    default=0.95,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Pixels at or above this value are treated as saturated/clipped "
                        "(matches the hardcoded 0.95 in Nuke's saturation_mask Expression node)."
                    ),
                ),
                io.Int.Input(
                    "start_frame",
                    default=1001,
                    min=0,
                    max=999999,
                    step=1,
                    tooltip=(
                        "Frame number for the first input path. "
                        "Output files are numbered from start_frame onwards."
                    ),
                ),
                io.String.Input(
                    "output_prefix",
                    default="frame_",
                    multiline=False,
                    optional=True,
                    tooltip="Prefix for output EXR filenames (e.g. 'frame_' → 'frame_001001.exr').",
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
        input_paths_json: str,
        output_dir: str,
        weights_path: str,
        device: str,
        multiply: float,
        saturation_threshold: float,
        start_frame: int,
        output_prefix: str = "frame_",
    ) -> io.NodeOutput:
        paths = _parse_paths(input_paths_json)
        if not paths:
            raise ValueError(
                "input_paths_json contains no paths. "
                'Provide a JSON list such as ["/path/frame.1001.exr", ...].'
            )

        if not output_dir:
            raise ValueError("output_dir is required")

        if not weights_path:
            raise ValueError("weights_path is required")

        out_paths, out_pattern = run_deep_hdr(
            input_paths=paths,
            output_dir=output_dir,
            weights_path=weights_path,
            device=device,
            multiply=float(multiply),
            saturation_threshold=float(saturation_threshold),
            start_frame=int(start_frame),
            output_prefix=output_prefix or "frame_",
            report_progress=report_progress,
        )

        info_lines = [
            f"Frames processed: {len(out_paths)}",
            f"Device: {device}",
            f"Weights: {weights_path}",
            f"Multiply: {float(multiply):.3f}",
            f"Saturation threshold: {float(saturation_threshold):.3f}",
            f"Output dir: {os.path.abspath(os.path.expanduser(output_dir))}",
            f"Output pattern: {out_pattern}",
            "Output EXRs have gamma_correct=0.5 baked in (values are H^0.5).",
        ]
        if abs(float(multiply) - 1.0) > 1e-6:
            info_lines.append(
                f"WARNING: multiply={float(multiply):.3f} != 1.0. "
                "The Nuke gizmo applies an inverse Grade (output / multiply) after inference. "
                "This node does not — apply the inverse manually if needed."
            )

        return io.NodeOutput(json.dumps(out_paths), out_pattern, "\n".join(info_lines))
