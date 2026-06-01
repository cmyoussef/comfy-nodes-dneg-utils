"""
Frame Padder & Trimmer node for ComfyUI.

Pads or trims an image batch so its frame count satisfies ``A * n + B``
for some integer *n*. Useful for fitting sequences to video-generation
model context lengths (e.g. Wan requires multiples of 4 + 1).
"""
from __future__ import annotations

import math

import torch
from comfy_api.latest import io


class WanHelper_WanFramePadder(io.ComfyNode):
    """
    Pad or trim an image sequence so its length equals ``A * n + B``.

    Padding modes control *what* frames are added, position controls
    *where* they go, and an optional crossfade blends the seam.
    In trim mode the extra frames are removed instead.
    """

    DESCRIPTION = (
        "Pad or trim an image batch to satisfy A*n + B frame count.\n\n"
        "Padding modes:\n"
        "  last_frame     - repeat the last frame\n"
        "  first_frame    - repeat the first frame\n"
        "  repeat_last_n  - cycle the last N frames\n"
        "  repeat_first_n - cycle the first N frames\n\n"
        "Trim mode removes padding frames added in a previous pass\n"
        "when original_frame_count is supplied."
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanHelper_WanFramePadder",
            display_name="WanHelper: Wan Frame Padder",
            category="WanHelper/video",
            description=cls.DESCRIPTION,
            inputs=[
                io.Image.Input("images", tooltip="Image batch [B,H,W,C] to pad or trim."),
                io.Combo.Input(
                    "trim_mode",
                    options=["OFF", "ON"],
                    default="OFF",
                    tooltip="ON = trim back to original_frame_count. OFF = pad up to nearest A*n + B.",
                ),
                io.Combo.Input(
                    "padding_mode",
                    options=["last_frame", "first_frame", "repeat_last_n", "repeat_first_n"],
                    default="last_frame",
                    tooltip="How to generate the padding frames.",
                ),
                io.Combo.Input(
                    "ping_pong",
                    options=["OFF", "ON"],
                    default="OFF",
                    tooltip="Reverse the pad segment before attaching (bounce effect).",
                ),
                io.Combo.Input(
                    "position",
                    options=["append", "prepend"],
                    default="append",
                    tooltip="Attach padding at the end or the beginning.",
                ),
                io.Int.Input(
                    "a_multiplier",
                    default=8,
                    min=1,
                    max=512,
                    step=1,
                    tooltip="A in target = A*n + B.",
                ),
                io.Int.Input(
                    "b_offset",
                    default=1,
                    min=-512,
                    max=512,
                    step=1,
                    tooltip="B in target = A*n + B.",
                ),
                io.Int.Input(
                    "crossfade_length",
                    default=0,
                    min=0,
                    max=128,
                    step=1,
                    tooltip="Number of frames to linearly blend at the seam (0 = hard cut).",
                ),
                io.Int.Input(
                    "original_frame_count",
                    default=0,
                    min=0,
                    max=10000,
                    step=1,
                    optional=True,
                    tooltip="Frame count before padding. Used by trim mode to restore the original length.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional mask [B,H,W]. Pre-aligned to the image batch length, "
                        "then padded or trimmed in lockstep with the images."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("images", tooltip="Padded (or trimmed) image batch [B,H,W,C]."),
                io.Int.Output("count", tooltip="Total frame count after processing."),
                io.String.Output("info", tooltip="Summary of the frame-count adjustment."),
                io.Mask.Output(
                    "mask",
                    tooltip="Mask aligned to the image output (same frame count). None passthrough if no mask was provided.",
                ),
            ],
        )

    @staticmethod
    def _log(message: str) -> None:
        print(f"[WanHelper_WanFramePadder] {message}")

    @staticmethod
    def _build_info(
        action: str,
        *,
        input_count: int,
        output_count: int,
        a_multiplier: int,
        b_offset: int,
        trim_mode: str,
        padding_mode: str,
        position: str,
        ping_pong: str,
        crossfade_length: int,
        target_count: int | None = None,
        delta: int | None = None,
        original_frame_count: int | None = None,
    ) -> str:
        lines = [
            f"Action: {action}",
            f"Input frames: {input_count}",
            f"Output frames: {output_count}",
            f"Trim mode: {trim_mode}",
            f"Padding mode: {padding_mode}",
            f"Position: {position}",
            f"Ping-pong: {ping_pong}",
            f"Crossfade length: {crossfade_length}",
            f"Target formula: {a_multiplier}*n + {b_offset}",
        ]

        if target_count is not None:
            lines.append(f"Target frames: {target_count}")
        if delta is not None:
            lines.append(f"Frame delta: {delta:+d}")
        if original_frame_count:
            lines.append(f"Original frame count: {original_frame_count}")

        return "\n".join(lines)

    @staticmethod
    def _apply_crossfade(
        seq1: torch.Tensor,
        seq2: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """Linearly blend the tail of *seq1* with the head of *seq2*. Works for 4D images and 3D masks."""
        if length <= 0:
            return torch.cat((seq1, seq2), dim=0)

        safe = min(length, seq1.shape[0], seq2.shape[0])
        if safe != length:
            WanHelper_WanFramePadder._log(
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
    ) -> torch.Tensor:
        """Create a tensor of *diff* frames according to *mode*. Works for 4D images and 3D masks."""
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

        return seq[:1].repeat(*repeat_args)

    @staticmethod
    def _align_to(seq: torch.Tensor, target_length: int) -> torch.Tensor:
        """Force *seq* to *target_length* by truncating or repeating its last frame. Image is the ground truth."""
        cur = seq.shape[0]
        if cur == target_length:
            return seq
        if cur > target_length:
            return seq[:target_length]
        pad_count = target_length - cur
        repeat_args = (pad_count,) + (1,) * (seq.dim() - 1)
        return torch.cat((seq, seq[-1:].repeat(*repeat_args)), dim=0)

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        trim_mode: str,
        padding_mode: str,
        ping_pong: str,
        position: str,
        a_multiplier: int,
        b_offset: int,
        crossfade_length: int,
        original_frame_count: int | None = 0,
        mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        original_frame_count = original_frame_count or 0
        current_count = images.shape[0]

        # Image is the ground truth: align the mask to the image batch length up front.
        # Shorter masks are extended by repeating their last frame; longer masks are truncated.
        # After this point, mask (if provided) walks through every code path in lockstep with images.
        if mask is not None:
            if mask.shape[0] != current_count:
                cls._log(
                    f"Mask aligned to image batch: {mask.shape[0]} -> {current_count} frames."
                )
                mask = cls._align_to(mask, current_count)

        if trim_mode == "ON":
            if original_frame_count <= 0 or original_frame_count >= current_count:
                cls._log(
                    f"Trim skipped: original_frame_count ({original_frame_count}) "
                    f"invalid or >= current ({current_count})."
                )
                info = cls._build_info(
                    "trim_skipped",
                    input_count=current_count,
                    output_count=current_count,
                    a_multiplier=a_multiplier,
                    b_offset=b_offset,
                    trim_mode=trim_mode,
                    padding_mode=padding_mode,
                    position=position,
                    ping_pong=ping_pong,
                    crossfade_length=crossfade_length,
                    target_count=current_count,
                    delta=0,
                    original_frame_count=original_frame_count,
                )
                return io.NodeOutput(images, current_count, info, mask)

            diff = current_count - original_frame_count
            cls._log(
                f"Trimming {diff} frames from {position} "
                f"to restore {original_frame_count} frames."
            )

            if position == "append":
                result = images[:original_frame_count]
                mask_out = mask[:original_frame_count] if mask is not None else None
            else:
                result = images[diff:]
                mask_out = mask[diff:] if mask is not None else None

            info = cls._build_info(
                "trim_applied",
                input_count=current_count,
                output_count=int(result.shape[0]),
                a_multiplier=a_multiplier,
                b_offset=b_offset,
                trim_mode=trim_mode,
                padding_mode=padding_mode,
                position=position,
                ping_pong=ping_pong,
                crossfade_length=crossfade_length,
                target_count=original_frame_count,
                delta=-diff,
                original_frame_count=original_frame_count,
            )
            return io.NodeOutput(result, result.shape[0], info, mask_out)

        a = max(1, a_multiplier)
        target = math.ceil((current_count - b_offset) / a) * a + b_offset

        if target <= current_count:
            if target == current_count:
                cls._log(f"{current_count} frames - already satisfies A*n + B, no padding needed.")
                info = cls._build_info(
                    "already_valid",
                    input_count=current_count,
                    output_count=current_count,
                    a_multiplier=a,
                    b_offset=b_offset,
                    trim_mode=trim_mode,
                    padding_mode=padding_mode,
                    position=position,
                    ping_pong=ping_pong,
                    crossfade_length=crossfade_length,
                    target_count=target,
                    delta=0,
                )
                return io.NodeOutput(images, current_count, info, mask)
            target += a

        diff = target - current_count
        cls._log(
            f"Input: {current_count} frames.  "
            f"Target (A={a}, B={b_offset}): {target}.  Padding: +{diff}"
        )

        pad_segment = cls._generate_pad_segment(images, padding_mode, diff)
        mask_pad_segment = (
            cls._generate_pad_segment(mask, padding_mode, diff) if mask is not None else None
        )

        if ping_pong == "ON":
            pad_segment = torch.flip(pad_segment, [0])
            if mask_pad_segment is not None:
                mask_pad_segment = torch.flip(mask_pad_segment, [0])

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

        if result.shape[0] != target:
            if result.shape[0] < target:
                extra = target - result.shape[0]
                result = torch.cat(
                    (result, result[-1:].repeat(extra, 1, 1, 1)), dim=0
                )
            else:
                result = result[:target]

        # Force mask length to match the final image length (image is ground truth).
        if mask_out is not None and mask_out.shape[0] != result.shape[0]:
            mask_out = cls._align_to(mask_out, result.shape[0])

        cls._log(f"Done. Output: {result.shape[0]} frames.")
        info = cls._build_info(
            "padding_applied",
            input_count=current_count,
            output_count=int(result.shape[0]),
            a_multiplier=a,
            b_offset=b_offset,
            trim_mode=trim_mode,
            padding_mode=padding_mode,
            position=position,
            ping_pong=ping_pong,
            crossfade_length=crossfade_length,
            target_count=target,
            delta=diff,
        )
        return io.NodeOutput(result, result.shape[0], info, mask_out)


