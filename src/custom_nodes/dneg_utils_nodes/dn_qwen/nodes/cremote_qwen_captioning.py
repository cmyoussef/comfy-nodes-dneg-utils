from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from comfy_api.latest import io

from .._progress import report_progress
from ..cremote_qwen_captioning.caption_service import CaptionService
from ..cremote_qwen_captioning.constants import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT_TEMPLATE,
)
from ..cremote_qwen_captioning.types import CaptionRequest
from ..cremote_qwen_captioning.utils import resolve_model_name_or_path

_caption_service = CaptionService()


class DN_QwenVideoCaptioner(io.ComfyNode):
    """Generate a caption for a single video using Qwen2.5-VL."""

    _ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DN_QwenVideoCaptioner",
            display_name="DN Qwen Video Captioner",
            category="DN/AI/Captioning",
            description=(
                "Generate a detailed one-paragraph caption for a video using Qwen2.5-VL. "
                "The node samples frames from the input video, injects the provided "
                "character name into the prompt template, and returns the generated caption, "
                "the resolved prompt sent to the model, and generation metadata as JSON."
            ),
            inputs=[
                io.String.Input(
                    "video_path",
                    tooltip=(
                        "Absolute path to the input video file. "
                        "Supported formats: .mp4, .mov, .avi, .mkv, .webm, .m4v."
                    ),
                ),
                io.String.Input(
                    "character_name",
                    tooltip=(
                        "Name or identity token for the main character or subject. "
                        "This value is injected into the prompt template via the "
                        '"{character_name}" placeholder and should be used consistently '
                        "in the generated caption."
                    ),
                ),
                io.String.Input(
                    "prompt_template",
                    default=DEFAULT_PROMPT_TEMPLATE,
                    multiline=True,
                    tooltip=(
                        "Prompt template used for caption generation. "
                        'Must contain the "{character_name}" placeholder. '
                        "Use this to control caption style, focus, level of detail, "
                        "and wording constraints."
                    ),
                ),
                io.String.Input(
                    "model_name_or_path",
                    default=DEFAULT_MODEL_NAME,
                    tooltip=(
                        "Model identifier or path. "
                        "If an absolute path is provided, it is used directly. "
                        "If a relative path or model name is provided, it is resolved "
                        "via the Comfy models root."
                    ),
                ),
                io.Float.Input(
                    "video_sampling_fps",
                    default=8.0,
                    min=0.1,
                    step=0.1,
                    tooltip=(
                        "Frame sampling rate in frames per second. "
                        "Higher values capture more visual detail but increase processing time "
                        "and memory usage."
                    ),
                ),
                io.Int.Input(
                    "max_new_tokens",
                    default=512,
                    min=1,
                    max=4096,
                    tooltip=(
                        "Maximum number of tokens the model may generate for the caption. "
                        "Increase for longer, more detailed outputs."
                    ),
                ),
                io.Float.Input(
                    "temperature",
                    default=0.3,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip=(
                        "Sampling temperature for text generation. "
                        "Lower values make output more deterministic and stable; "
                        "higher values increase variation and creativity."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=424242,
                    min=0,
                    tooltip=(
                        "Random seed used for reproducible generation settings. "
                        "Use the same seed and parameters to get more consistent results."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    "caption",
                    tooltip="Generated one-paragraph video caption."
                ),
                io.String.Output(
                    "resolved_prompt",
                    tooltip="Final prompt after substituting variables such as character_name."
                ),
                io.String.Output(
                    "generation_info_json",
                    tooltip="Generation metadata serialized as JSON, including model and runtime details."
                ),
            ],
            search_aliases=[
                "qwen video caption",
                "video caption",
                "qwen caption",
                "qwen2.5 vl",
            ],
        )

    @classmethod
    def _validate_video_path(cls, video_path: str) -> str | None:
        path = Path(video_path)

        suffix = path.suffix.lower()
        if suffix and suffix not in cls._ALLOWED_VIDEO_SUFFIXES:
            return f"unsupported video file extension: {suffix}"

        if not path.exists():
            return f"video_path does not exist: {video_path}"

        if not path.is_file():
            return f"video_path is not a file: {video_path}"

        return None

    @classmethod
    def validate_inputs(
        cls,
        video_path: str,
        character_name: str,
        prompt_template: str,
        **kwargs: Any,
    ) -> bool | str:
        if not character_name or not str(character_name).strip():
            return "character_name must be a non-empty string"

        if not prompt_template or "{character_name}" not in prompt_template:
            return 'prompt_template must contain "{character_name}"'

        # video_path may come from another node and may still be unresolved during validation,
        # so only lightweight suffix validation is done here.
        if isinstance(video_path, str) and video_path.strip():
            suffix = Path(video_path).suffix.lower()
            if suffix and suffix not in cls._ALLOWED_VIDEO_SUFFIXES:
                return f"unsupported video file extension: {suffix}"

        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        video_path: str | None = None,
        character_name: str | None = None,
        prompt_template: str | None = None,
        model_name_or_path: str | None = None,
        video_sampling_fps: float | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> str:
        # During graph analysis inputs may not be resolved yet.
        # Return a stable placeholder until video_path is available.
        if not video_path or not str(video_path).strip():
            return "pending:video_path"

        path = Path(video_path)
        if not path.exists():
            return f"missing:{video_path}"

        stat = path.stat()

        resolved_model_name_or_path = (
            resolve_model_name_or_path(model_name_or_path)
            if model_name_or_path
            else ""
        )

        return "|".join(
            [
                str(path.resolve()),
                str(int(stat.st_mtime)),
                str(stat.st_size),
                str(character_name or ""),
                str(prompt_template or ""),
                resolved_model_name_or_path,
                str(video_sampling_fps),
                str(max_new_tokens),
                str(temperature),
                str(seed),
            ]
        )

    @classmethod
    def execute(
        cls,
        video_path: str,
        character_name: str,
        prompt_template: str,
        model_name_or_path: str,
        video_sampling_fps: float,
        max_new_tokens: int,
        temperature: float,
        seed: int,
    ) -> io.NodeOutput:

        # Full video_path validation is done here after inputs are resolved at execution time.
        error = cls._validate_video_path(video_path)
        if error:
            return io.NodeOutput(block_execution=error)

        request = CaptionRequest(
            video_path=video_path,
            character_name=character_name,
            prompt_template=prompt_template,
            model_name_or_path=resolve_model_name_or_path(model_name_or_path),
            video_sampling_fps=video_sampling_fps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )

        report_progress(0, 3)
        result = _caption_service.caption_video(request)
        report_progress(3, 3)

        generation_info = result.generation_info
        if is_dataclass(generation_info):
            generation_info = asdict(generation_info)

        generation_info_json = json.dumps(
            generation_info,
            ensure_ascii=False,
            indent=2,
        )

        return io.NodeOutput(
            result.caption,
            result.resolved_prompt,
            generation_info_json,
        )