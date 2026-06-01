"""
Depth Normalize node for ComfyUI.

Normalizes a z-depth input to 0-1 range and provides front/back push
sliders to remap the depth range, with an invert option.
"""
from __future__ import annotations

import torch
from comfy_api.latest import io


class WanHelper_DepthNormalize(io.ComfyNode):
    """
    Normalize a z-depth image to 0-1 and remap with front/back push controls.

    The input depth is first normalized (min-max) to the 0-1 range.
    ``near_push`` clips and remaps from the near (dark/close) side,
    ``far_push`` clips and remaps from the far (bright/distant) side.
    The result is re-normalized to fill the full 0-1 range.
    ``invert`` flips the final output (1 - depth).
    """

    DESCRIPTION = (
        "Normalize z-depth to 0-1 with front/back remap controls.\n\n"
        "1. Min-max normalizes the input depth to 0-1.\n"
        "2. near_push clips the near (dark) end of the range.\n"
        "3. far_push clips the far (bright) end of the range.\n"
        "4. The remaining range is re-stretched to 0-1.\n"
        "5. Invert flips the output (1 - depth)."
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanHelper_DepthNormalize",
            display_name="WanHelper: Depth Normalize",
            category="WanHelper/depth",
            description=cls.DESCRIPTION,
            inputs=[
                io.Image.Input("image", tooltip="Input z-depth image [B,H,W,C]."),
                io.Float.Input(
                    "near_push",
                    default=0.0,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    tooltip=(
                        "Push the near (dark/close) end of the depth range. "
                        "Values below this threshold become black (0). "
                        "The remaining range is re-normalized to 0-1."
                    ),
                ),
                io.Float.Input(
                    "far_push",
                    default=0.0,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    tooltip=(
                        "Push the far (bright/distant) end of the depth range. "
                        "Values above this threshold become white (1). "
                        "The remaining range is re-normalized to 0-1."
                    ),
                ),
                io.Boolean.Input(
                    "invert",
                    default=False,
                    tooltip="Invert the final depth output (1 - depth).",
                ),
            ],
            outputs=[
                io.Image.Output("image", tooltip="Normalized depth image [B,H,W,C]."),
                io.Mask.Output("mask", tooltip="Normalized depth as single-channel mask [B,H,W]."),
                io.String.Output("info", tooltip="Original non-zero depth min/max values."),
            ],
        )

    @staticmethod
    def _log(message: str) -> None:
        print(f"[WanHelper_DepthNormalize] {message}")

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        near_push: float,
        far_push: float,
        invert: bool,
    ) -> io.NodeOutput:
        # image shape: [B, H, W, C]
        # Convert to grayscale for depth processing (average across channels)
        if image.shape[-1] > 1:
            depth = image.mean(dim=-1)  # [B, H, W]
        else:
            depth = image[..., 0]  # [B, H, W]

        # Build a mask of non-zero pixels
        nonzero_mask = depth > 0.0

        # Min-max normalize using only non-zero pixels
        if nonzero_mask.any():
            # Replace zeros with inf/−inf so they don't affect min/max
            masked = torch.where(nonzero_mask, depth, torch.full_like(depth, float("inf")))
            d_min = masked.amin(dim=(-2, -1), keepdim=True)
            masked = torch.where(nonzero_mask, depth, torch.full_like(depth, float("-inf")))
            d_max = masked.amax(dim=(-2, -1), keepdim=True)
        else:
            d_min = depth.amin(dim=(-2, -1), keepdim=True)
            d_max = depth.amax(dim=(-2, -1), keepdim=True)

        d_range = d_max - d_min
        # Avoid division by zero for flat depth
        d_range = torch.where(d_range == 0, torch.ones_like(d_range), d_range)

        depth = (depth - d_min) / d_range
        depth = torch.clamp(depth, 0.0, 1.0)
        # Force originally-zero pixels back to zero
        depth = torch.where(nonzero_mask, depth, torch.zeros_like(depth))

        raw_min = d_min.min().item()
        raw_max = d_max.max().item()
        cls._log(f"Non-zero depth range: [{raw_min:.6f}, {raw_max:.6f}]")

        # Apply near/far push remap
        if near_push > 0.0 or far_push > 0.0:
            low = near_push
            high = 1.0 - far_push
            if high <= low:
                cls._log(
                    f"Warning: near_push ({near_push}) + far_push ({far_push}) >= 1.0. "
                    f"Output will be clamped."
                )
                high = low + 1e-6  # prevent division by zero

            depth = (depth - low) / (high - low)
            depth = torch.clamp(depth, 0.0, 1.0)
            # Keep zero pixels as zero after remap
            depth = torch.where(nonzero_mask, depth, torch.zeros_like(depth))
            cls._log(f"Applied remap: near_push={near_push:.2f}, far_push={far_push:.2f}")

        # Invert
        if invert:
            # Only invert non-zero pixels, keep zero pixels as zero
            depth = torch.where(nonzero_mask, 1.0 - depth, torch.zeros_like(depth))
            cls._log("Applied invert.")

        # Build outputs
        mask_out = depth  # [B, H, W]
        image_out = depth.unsqueeze(-1).expand_as(image)  # [B, H, W, C]
        info = f"min: {raw_min:.6f} | max: {raw_max:.6f}"

        return io.NodeOutput(image_out, mask_out, info)
