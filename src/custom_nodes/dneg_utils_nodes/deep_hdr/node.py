"""
deep_hdr.node

CREMOTE_DeepHDR — ComfyUI node for DeepHDR highlight reconstruction.

Accepts IMAGE and optional MASK tensors, runs deep_hdr.core.deep_hdr.Run
internally (via a temp-file round-trip that is invisible to the user),
and returns IMAGE + MASK tensors compatible with downstream Comfy nodes.

Behaviour matches the existing Nuke Bridge gizmo as closely as possible.
Output IMAGE has gamma_correct=0.5 baked in (values are H^0.5), identical
to what Nuke reads back from the gizmo before any inverse-multiply Grade node.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from comfy_api.latest import io

from ._progress import report_progress
from .core import run_deep_hdr_arrays


def _resolve_weights() -> str:
    """
    Find ldr2hdr.pth via a prioritised search, with no hardcoded show paths.

    Resolution order:
      1. DEEP_HDR_WEIGHTS_PATH env var — explicit admin override.
      2. DNEG shared Comfy models repository.
      3. COMFY_MODELS_ROOT/deep_hdr/ldr2hdr.pth.
      4. COMFY_MODELS_ROOT/models/deep_hdr/ldr2hdr.pth.
      5. ComfyUI folder_paths checkpoints / models_dir.
      6. COMFYUI_HOME/models/deep_hdr/ldr2hdr.pth.
    """
    candidates: list[tuple[str, str]] = []

    env_path = os.environ.get("DEEP_HDR_WEIGHTS_PATH", "").strip()
    if env_path:
        candidates.append((
            "DEEP_HDR_WEIGHTS_PATH",
            os.path.abspath(os.path.expanduser(env_path)),
        ))

    candidates.append((
        "DNEG shared models",
        "/jobs/SITE/V_PROD/VP_RND/StableDiffusion/SD_models_repo/models/deep_hdr/ldr2hdr.pth",
    ))

    models_root = os.environ.get("COMFY_MODELS_ROOT", "").strip()
    if models_root:
        models_root = os.path.abspath(os.path.expanduser(models_root))
        candidates.extend([
            (
                "COMFY_MODELS_ROOT/deep_hdr",
                os.path.join(models_root, "deep_hdr", "ldr2hdr.pth"),
            ),
            (
                "COMFY_MODELS_ROOT/models/deep_hdr",
                os.path.join(models_root, "models", "deep_hdr", "ldr2hdr.pth"),
            ),
        ])

    try:
        import folder_paths

        search_dirs: list[str] = []

        try:
            search_dirs += folder_paths.get_folder_paths("checkpoints")
        except Exception:
            pass

        try:
            search_dirs.append(folder_paths.models_dir)
        except Exception:
            pass

        for base in search_dirs:
            if not base:
                continue
            base = os.path.abspath(os.path.expanduser(base))
            candidates.append((
                f"Comfy model path: {base}",
                os.path.join(base, "deep_hdr", "ldr2hdr.pth"),
            ))
    except ImportError:
        pass

    comfyui_home = os.environ.get("COMFYUI_HOME", "").strip()
    if comfyui_home:
        comfyui_home = os.path.abspath(os.path.expanduser(comfyui_home))
        candidates.append((
            "COMFYUI_HOME/models/deep_hdr",
            os.path.join(comfyui_home, "models", "deep_hdr", "ldr2hdr.pth"),
        ))

    missing = []
    for source, path in candidates:
        if os.path.isfile(path):
            return path
        missing.append(f"  - {source}: {path}")

    raise FileNotFoundError(
        "DeepHDR weights (ldr2hdr.pth) not found.\n"
        "Tried the following locations:\n"
        + "\n".join(missing)
        + "\n\nSet DEEP_HDR_WEIGHTS_PATH to the absolute path of ldr2hdr.pth, "
        "or place it at:\n"
        "  /jobs/SITE/V_PROD/VP_RND/StableDiffusion/SD_models_repo/models/deep_hdr/ldr2hdr.pth"
    )


class CREMOTE_DeepHDR(io.ComfyNode):
    """DeepHDR highlight reconstruction node (DNEG)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CREMOTE_DeepHDR",
            display_name="CREMOTE: DeepHDR (Highlight Reconstruction)",
            category="image/dneg",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "Scene-linear input image(s). "
                        "Pixels at or above saturation_threshold are treated as clipped "
                        "and will be reconstructed by the model."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional saturation mask (values in [0, 1], 1 = saturated). "
                        "If connected, this overrides the auto-computed saturation mask "
                        "and saturation_threshold is ignored. "
                        "If absent, the mask is computed from the image using saturation_threshold."
                    ),
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
                        "Threshold for the auto-computed saturation mask: pixels at or above "
                        "this value are treated as saturated/clipped and passed to DeepHDR for "
                        "reconstruction (matches the hardcoded 0.95 in Nuke's saturation_mask "
                        "Expression node).\n\n"
                        "Ignored when a MASK input is connected — the widget is greyed out "
                        "automatically in that case."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        device: str,
        multiply: float,
        saturation_threshold: float,
        mask: Optional[torch.Tensor] = None,
    ) -> io.NodeOutput:
        weights_path = _resolve_weights()

        # image: [B,H,W,3]
        b = image.shape[0]
        images_np = [image[i].cpu().numpy().astype(np.float32) for i in range(b)]

        # mask: [B,H,W] — expand each frame to [H,W,1] for run_deep_hdr_arrays
        masks_np = None
        if mask is not None:
            masks_np = [
                mask[i].cpu().numpy().astype(np.float32)[:, :, np.newaxis]
                for i in range(b)
            ]

        out_images, masks_used = run_deep_hdr_arrays(
            images=images_np,
            weights_path=weights_path,
            device=device,
            multiply=float(multiply),
            saturation_threshold=float(saturation_threshold),
            masks=masks_np,
            report_progress=report_progress,
        )

        # Stack outputs back to Comfy tensor shapes
        out_tensor = torch.from_numpy(np.stack(out_images, axis=0))  # [B,H,W,3]

        # masks_used are [H,W,1] — squeeze channel dim and stack to [B,H,W]
        mask_arrays = [
            m[:, :, 0] if m.ndim == 3 else m
            for m in masks_used
        ]
        mask_tensor = torch.from_numpy(np.stack(mask_arrays, axis=0))  # [B,H,W]

        mask_source = (
            "provided"
            if mask is not None
            else f"auto (saturation_threshold={float(saturation_threshold):.2f})"
        )
        info_lines = [
            f"Frames processed: {b}",
            f"Device: {device}",
            f"Weights: {weights_path}",
            f"Multiply: {float(multiply):.3f}",
            f"Mask: {mask_source}",
            "Output has gamma_correct=0.5 baked in (values are H^0.5).",
        ]
        if abs(float(multiply) - 1.0) > 1e-6:
            info_lines.append(
                f"WARNING: multiply={float(multiply):.3f} != 1.0. "
                "The Nuke gizmo applies an inverse Grade (output / multiply) after inference. "
                "This node does not — apply the inverse manually if needed."
            )

        return io.NodeOutput(out_tensor, mask_tensor, "\n".join(info_lines))
