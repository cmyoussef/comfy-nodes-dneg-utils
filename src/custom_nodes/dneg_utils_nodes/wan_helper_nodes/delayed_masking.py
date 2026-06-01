"""
Delayed Masking compatibility node for ComfyUI.

Supports both the latest API shape (anim_start + anim_length) and the legacy
shape (anim_start + anim_end). Also supports the common legacy pattern where
absolute shot-frame values (for example 1008-1015) are wired into a local batch
mask by converting them relative to a timeline start.
"""
from __future__ import annotations

import torch
from comfy_api.latest import io


class WanHelper_DelayedMasking(io.ComfyNode):
    """
    Generate a linear wipe mask across a frame batch.

    Compatibility behavior:
    - `anim_end` is accepted as a legacy inclusive end-frame input.
    - When `anim_start` is outside the local batch range, values are treated as
      absolute timeline frames and converted relative to `timeline_start`.
    """

    DESCRIPTION = (
        "Generate a linear wipe mask across a frame batch.\n\n"
        "Compatibility notes:\n"
        "  - Accepts legacy anim_end and converts it to anim_length.\n"
        "  - If anim_start is outside the local batch range, values are treated\n"
        "    as absolute shot frames and converted relative to timeline_start."
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DelayedMasking",
            display_name="WanHelper: Delayed Masking",
            category="WanHelper/mask",
            description=cls.DESCRIPTION,
            inputs=[
                io.Int.Input(
                    "width",
                    default=1024,
                    min=1,
                    step=1,
                    tooltip="Mask width in pixels.",
                ),
                io.Int.Input(
                    "height",
                    default=1024,
                    min=1,
                    step=1,
                    tooltip="Mask height in pixels.",
                ),
                io.Int.Input(
                    "frames",
                    default=48,
                    min=1,
                    step=1,
                    tooltip="Number of frames in the output batch.",
                ),
                io.Int.Input(
                    "anim_start",
                    default=0,
                    min=0,
                    step=1,
                    tooltip="Start frame for the wipe. Can be local batch index or absolute shot frame.",
                ),
                io.Int.Input(
                    "anim_length",
                    default=24,
                    min=1,
                    step=1,
                    optional=True,
                    tooltip="Length of the animation in frames. Latest API input.",
                ),
                io.Int.Input(
                    "anim_end",
                    default=0,
                    min=0,
                    step=1,
                    optional=True,
                    tooltip="Legacy inclusive end frame. If provided, overrides anim_length.",
                ),
                io.Int.Input(
                    "handles",
                    default=0,
                    min=0,
                    step=1,
                    optional=True,
                    tooltip="Kept for workflow compatibility. Currently does not affect the wipe.",
                ),
                io.Combo.Input(
                    "direction",
                    options=[
                        "top_to_bottom",
                        "bottom_to_top",
                        "left_to_right",
                        "right_to_left",
                    ],
                    default="top_to_bottom",
                    tooltip="Direction of the linear wipe.",
                ),
                io.Int.Input(
                    "timeline_start",
                    default=1001,
                    min=0,
                    step=1,
                    optional=True,
                    tooltip=(
                        "Shot start frame used when anim_start/anim_end are absolute "
                        "timeline values instead of local batch indices."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    "mask",
                    tooltip="Animated wipe mask [frames, H, W].",
                ),
            ],
        )

    @staticmethod
    def _resolve_window(
        frames: int,
        anim_start: int,
        anim_length: int | None,
        anim_end: int | None,
        timeline_start: int | None,
    ) -> tuple[int, int, int]:
        start = int(anim_start)
        end = int(anim_end) if anim_end is not None else None
        length = max(1, int(anim_length or 1))
        offset = int(timeline_start or 0)

        # Legacy workflows often wire absolute shot-frame values (1001-based)
        # into a local batch wipe. Convert them when the start lies outside the
        # batch range and an offset is available.
        if start >= frames and offset > 0:
            start -= offset
            if end is not None and end > 0:
                end -= offset

        if end is not None and end >= start:
            # Legacy anim_end is inclusive, so the effective frame count is
            # (end - start + 1) before we convert it to the internal length.
            length = max(1, end - start + 1)
        else:
            end = start + length - 1

        start = max(0, start)
        end = max(start, int(end))
        span = max((end - start) + 1, 1)
        return start, end, span

    @staticmethod
    def _apply_wipe(mask: torch.Tensor, index: int, progress: float, direction: str) -> None:
        h = mask.shape[1]
        w = mask.shape[2]

        if direction == "top_to_bottom":
            cutoff = int(h * progress)
            mask[index, :cutoff, :] = 1.0
        elif direction == "bottom_to_top":
            cutoff = int(h * progress)
            mask[index, h - cutoff:h, :] = 1.0
        elif direction == "left_to_right":
            cutoff = int(w * progress)
            mask[index, :, :cutoff] = 1.0
        elif direction == "right_to_left":
            cutoff = int(w * progress)
            mask[index, :, w - cutoff:w] = 1.0

    @classmethod
    def execute(
        cls,
        width: int,
        height: int,
        frames: int,
        anim_start: int,
        anim_length: int | None = 24,
        anim_end: int | None = None,
        handles: int | None = 0,
        direction: str = "top_to_bottom",
        timeline_start: int | None = 1001,
    ) -> io.NodeOutput:
        del handles  # kept for compatibility with the legacy node interface

        start, end, span = cls._resolve_window(
            frames=frames,
            anim_start=anim_start,
            anim_length=anim_length,
            anim_end=anim_end,
            timeline_start=timeline_start,
        )

        mask = torch.zeros((frames, height, width), dtype=torch.float32)
        anim_stop = end + 1

        for i in range(frames):
            if i < start:
                progress = 0.0
            elif i < anim_stop:
                progress = (i - start) / max(span - 1, 1)
            else:
                progress = 1.0

            progress = max(0.0, min(progress, 1.0))
            cls._apply_wipe(mask, i, progress, direction)

        return io.NodeOutput(mask)
