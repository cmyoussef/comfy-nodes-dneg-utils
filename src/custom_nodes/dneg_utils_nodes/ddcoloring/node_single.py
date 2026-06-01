# src\comfyui_remote\custom_nodes\ddcoloring\node_single.py
"""ddcoloring.node_single

ComfyUI node for DDColor single-image colorization.

Features:
- Standalone (no dneg_ddcolor dependency)
- Optional reference image palette steering (ab distribution matching)
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from comfy_api.latest import io

from ._progress import report_progress
from .core import ColorizationPipeline, bgr_u8_to_comfy_tensor, comfy_tensor_to_bgr_u8


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


class DDColorNode(io.ComfyNode):
    """DDColor Colorization Node for ComfyUI (single image / batch)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DDColorNode",
            display_name="DDColor Colorization (Ref Optional)",
            category="image/color",
            inputs=[
                io.Image.Input("image"),
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
                    tooltip="Model architecture variant (ignored if checkpoint has model_config)",
                ),
                io.Image.Input(
                    "reference_image",
                    optional=True,
                    tooltip="Optional reference image used to steer the palette (no retraining).",
                ),
                io.Float.Input(
                    "reference_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip="0 disables reference steering; 1 = full match.",
                ),
                io.Combo.Input(
                    "reference_method",
                    options=["meanstd", "cov"],
                    default="meanstd",
                    optional=True,
                    tooltip="Reference match method: mean/std (fast) or covariance (slower).",
                ),
            ],
            outputs=[
                io.Image.Output("colorized_image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        model_path: str,
        input_size: int,
        model_size: str,
        reference_image: Optional[torch.Tensor] = None,
        reference_strength: float = 1.0,
        reference_method: str = "meanstd",
    ) -> io.NodeOutput:
        """Colorize a ComfyUI IMAGE batch.

        Args:
            image: ComfyUI IMAGE tensor [B,H,W,C] RGB float32 in [0,1].
            reference_image: Optional IMAGE tensor (first element used if batched).
        """
        if not model_path:
            raise ValueError("model_path is required")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = _get_pipeline(model_path, model_size, input_size, device)

        ref_bgr_u8 = None
        if reference_image is not None and float(reference_strength) > 0:
            ref_bgr_u8 = comfy_tensor_to_bgr_u8(reference_image[0])

        results = []
        for i in range(image.shape[0]):
            img_bgr_u8 = comfy_tensor_to_bgr_u8(image[i])
            out_bgr_u8 = pipe.colorize_bgr(
                img_bgr_u8,
                reference_bgr_u8=ref_bgr_u8,
                reference_strength=float(reference_strength),
                reference_method=str(reference_method),
            )
            results.append(bgr_u8_to_comfy_tensor(out_bgr_u8))
            report_progress(i + 1, image.shape[0])

        return io.NodeOutput(torch.stack(results, dim=0))
