"""
Mask Ramp node for ComfyUI.

Zeroes mask frames before a start point (with optional handle offset)
and outputs a strength ramp from 0 to 1 over a configurable delay.
Useful for ramping in effects or IP-adapter strength across a sequence.
"""
from __future__ import annotations

import torch
from comfy_api.latest import io


class WanHelper_MaskRamp(io.ComfyNode):
    """
    Zero out mask frames before a start point and produce a 0-1 strength ramp.

    Frames before ``mask_start + handles_added`` are zeroed in the output mask.
    The strength output ramps linearly from 0 to 1 over ``delay`` frames
    starting at that point, quantised to 0.01 steps.
    """

    DESCRIPTION = (
        "Zero mask frames before a start point and output a strength ramp.\n\n"
        "Frames before (mask_start + handles_added) are zeroed.\n"
        "Strength ramps 0 -> 1 over 'delay' frames from that point,\n"
        "quantised to 0.01 steps."
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanHelper_MaskRamp",
            display_name="WanHelper: Mask Ramp",
            category="WanHelper/mask",
            description=cls.DESCRIPTION,
            inputs=[
                io.Mask.Input("mask", tooltip="Input mask [frames, H, W]."),
                io.Int.Input(
                    "mask_start",
                    default=0,
                    min=0,
                    step=1,
                    tooltip="Frame index where the mask begins to take effect.",
                ),
                io.Int.Input(
                    "delay",
                    default=1,
                    min=1,
                    step=1,
                    tooltip="Number of frames over which strength ramps from 0 to 1.",
                ),
                io.Int.Input(
                    "handles_added",
                    default=0,
                    min=0,
                    step=1,
                    optional=True,
                    tooltip="Extra handle frames to offset start point (e.g. from padding).",
                ),
            ],
            outputs=[
                io.Mask.Output("mask", tooltip="Mask with frames before start zeroed out [frames, H, W]."),
                io.Float.Output("strength", tooltip="Per-frame strength ramp 0-1, quantised to 0.01 steps [frames]."),
            ],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        mask_start: int,
        delay: int,
        handles_added: int | None = 0,
    ) -> io.NodeOutput:
        frame_count = mask.shape[0]
        start = max(0, mask_start + (handles_added or 0))

        mask_out = mask.clone()
        cutoff = min(start, frame_count)
        if cutoff > 0:
            mask_out[:cutoff] = 0.0

        frames = torch.arange(frame_count, device=mask.device, dtype=torch.float32)

        if delay <= 0:
            strength = (frames >= start).float()
        else:
            strength = (frames - start) / float(delay)
            strength = torch.clamp(strength, 0.0, 1.0)

        strength = torch.round(strength * 100.0) / 100.0
        return io.NodeOutput(mask_out, strength)
