"""
Frame Extender node for ComfyUI.

Extends an image batch to an exact target frame count. No A*n+B math:
the user supplies the length they want, and the node fills the gap
using one of several padding strategies. If the input is already at
or beyond target_length, the batch is returned unchanged (no trim).
"""
from __future__ import annotations

import torch
from comfy_api.latest import io


class WanHelper_WanFrameExtender(io.ComfyNode):
    """
    Extend an image sequence to an exact target frame count.

    Padding modes choose which existing frames fill the gap, position
    controls whether new frames go at the head or tail, and an optional
    crossfade smooths the seam.
    """

    DESCRIPTION = (
        "Extend an image batch to an exact target_length.\n\n"
        "Padding modes:\n"
        "  last_frame     - repeat the last frame\n"
        "  first_frame    - repeat the first frame\n"
        "  repeat_last_n  - cycle the last N frames\n"
        "  repeat_first_n - cycle the first N frames\n"
        "  ping_pong      - bounce forward/backward through the whole batch\n\n"
        "If input length >= target_length, the batch is returned unchanged."
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanHelper_WanFrameExtender",
            display_name="WanHelper: Wan Frame Extender",
            category="WanHelper/video",
            description=cls.DESCRIPTION,
            inputs=[
                io.Image.Input("images", tooltip="Image batch [B,H,W,C] to extend."),
                io.Int.Input(
                    "target_length",
                    default=81,
                    min=1,
                    max=10000,
                    step=1,
                    tooltip="Exact frame count to reach.",
                ),
                io.Combo.Input(
                    "padding_mode",
                    options=[
                        "last_frame",
                        "first_frame",
                        "repeat_last_n",
                        "repeat_first_n",
                        "ping_pong",
                    ],
                    default="last_frame",
                    tooltip="How to generate the added frames.",
                ),
                io.Combo.Input(
                    "position",
                    options=["append", "prepend"],
                    default="append",
                    tooltip="Attach added frames at the end or the beginning.",
                ),
                io.Int.Input(
                    "crossfade_length",
                    default=0,
                    min=0,
                    max=128,
                    step=1,
                    tooltip="Frames to linearly blend at the seam (0 = hard cut).",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional mask [B,H,W]. Pre-aligned to the image batch length, "
                        "then extended in lockstep with the images."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("images", tooltip="Extended image batch [B,H,W,C]."),
                io.Int.Output("count", tooltip="Total frame count after processing."),
                io.String.Output("info", tooltip="Summary of the extension."),
                io.Mask.Output(
                    "mask",
                    tooltip="Mask aligned to the image output. None passthrough if no mask was provided.",
                ),
            ],
        )

    @staticmethod
    def _log(message: str) -> None:
        print(f"[WanHelper_WanFrameExtender] {message}")

    @staticmethod
    def _align_to(seq: torch.Tensor, target_length: int) -> torch.Tensor:
        """Force *seq* to *target_length* by truncating or repeating its last frame."""
        cur = seq.shape[0]
        if cur == target_length:
            return seq
        if cur > target_length:
            return seq[:target_length]
        pad_count = target_length - cur
        repeat_args = (pad_count,) + (1,) * (seq.dim() - 1)
        return torch.cat((seq, seq[-1:].repeat(*repeat_args)), dim=0)

    @staticmethod
    def _apply_crossfade(
        seq1: torch.Tensor,
        seq2: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """Linearly blend the tail of *seq1* with the head of *seq2*."""
        if length <= 0:
            return torch.cat((seq1, seq2), dim=0)

        safe = min(length, seq1.shape[0], seq2.shape[0])
        if safe != length:
            WanHelper_WanFrameExtender._log(
                f"Crossfade length clamped to {safe} (sequence too short)."
            )

        base1 = seq1[:-safe]
        blend1 = seq1[-safe:]
        blend2 = seq2[:safe]
        base2 = seq2[safe:]

        view_shape = (safe,) + (1,) * (seq1.dim() - 1)
        ramp = torch.linspace(0, 1, safe, device=seq1.device).view(*view_shape)
        blended = (1.0 - ramp) * blend1 + ramp * blend2

        return torch.cat((base1, blended, base2), dim=0)

    @staticmethod
    def _generate_pad_segment(
        seq: torch.Tensor,
        mode: str,
        diff: int,
        position: str,
    ) -> torch.Tensor:
        """Create a tensor of *diff* frames according to *mode*."""
        count = seq.shape[0]
        device = seq.device
        repeat_args = (diff,) + (1,) * (seq.dim() - 1)

        if mode == "last_frame":
            return seq[-1:].repeat(*repeat_args)
        if mode == "first_frame":
            return seq[:1].repeat(*repeat_args)
        if mode == "repeat_last_n":
            indices = torch.arange(count - diff, count, device=device) % count
            return seq[indices]
        if mode == "repeat_first_n":
            indices = torch.arange(0, diff, device=device) % count
            return seq[indices]
        if mode == "ping_pong":
            if count == 1:
                return seq.repeat(*repeat_args)
            # period: [0, 1, ..., count-1, count-2, ..., 1] (length 2*(count-1))
            period_len = 2 * (count - 1)
            period = torch.cat(
                (
                    torch.arange(count, device=device),
                    torch.arange(count - 2, 0, -1, device=device),
                ),
                dim=0,
            )
            if position == "append":
                # frames after position count-1
                positions = torch.arange(count, count + diff, device=device)
            else:
                # frames before position 0
                positions = torch.arange(-diff, 0, device=device)
            indices = period[positions % period_len]
            return seq[indices]

        return seq[:1].repeat(*repeat_args)

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        target_length: int,
        padding_mode: str,
        position: str,
        crossfade_length: int,
        mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        current_count = images.shape[0]

        # Image is the ground truth: align the mask to the image batch length up front.
        if mask is not None and mask.shape[0] != current_count:
            cls._log(
                f"Mask aligned to image batch: {mask.shape[0]} -> {current_count} frames."
            )
            mask = cls._align_to(mask, current_count)

        if current_count >= target_length:
            cls._log(
                f"Input ({current_count}) >= target ({target_length}); passthrough, no trim."
            )
            info = (
                f"Action: passthrough\n"
                f"Input frames: {current_count}\n"
                f"Output frames: {current_count}\n"
                f"Target frames: {target_length}\n"
                f"Padding mode: {padding_mode}\n"
                f"Position: {position}\n"
                f"Crossfade length: {crossfade_length}"
            )
            return io.NodeOutput(images, current_count, info, mask)

        diff = target_length - current_count
        cls._log(
            f"Input: {current_count} frames.  Target: {target_length}.  Padding: +{diff}"
        )

        pad_segment = cls._generate_pad_segment(images, padding_mode, diff, position)
        mask_pad_segment = (
            cls._generate_pad_segment(mask, padding_mode, diff, position)
            if mask is not None
            else None
        )

        if position == "append":
            result = cls._apply_crossfade(images, pad_segment, crossfade_length)
            mask_out = (
                cls._apply_crossfade(mask, mask_pad_segment, crossfade_length)
                if mask is not None
                else None
            )
        else:
            result = cls._apply_crossfade(pad_segment, images, crossfade_length)
            mask_out = (
                cls._apply_crossfade(mask_pad_segment, mask, crossfade_length)
                if mask is not None
                else None
            )

        # Force exact target length.
        if result.shape[0] != target_length:
            result = cls._align_to(result, target_length)
        if mask_out is not None and mask_out.shape[0] != result.shape[0]:
            mask_out = cls._align_to(mask_out, result.shape[0])

        cls._log(f"Done. Output: {result.shape[0]} frames.")
        info = (
            f"Action: extend\n"
            f"Input frames: {current_count}\n"
            f"Output frames: {result.shape[0]}\n"
            f"Target frames: {target_length}\n"
            f"Frame delta: +{diff}\n"
            f"Padding mode: {padding_mode}\n"
            f"Position: {position}\n"
            f"Crossfade length: {crossfade_length}"
        )
        return io.NodeOutput(result, result.shape[0], info, mask_out)
