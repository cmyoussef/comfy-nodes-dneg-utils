from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass(slots=True)
class CaptionRequest:
    """Request for generating a video caption using a vision-language model.

    The request contains the video path, the character identity token, and
    generation parameters used to produce the caption.

    Args:
        video_path (str):
            Path to the video file for which the caption should be generated.

        character_name (str):
            Identity token of the main character in the video. This token is
            injected into the prompt template and must appear in the generated
            caption.

        prompt_template (str):
            Prompt template describing how the model should analyze the video.
            The template must contain the token "{character_name}", which will
            be replaced with the actual character name.

        model_name_or_path (str):
            Name or local path of the vision-language model used for caption
            generation.

        video_sampling_fps (float):
            Frame sampling rate used when extracting frames from the video.
            Higher values provide more visual context but increase processing
            time.

        max_new_tokens (int):
            Maximum number of tokens the model is allowed to generate.

        temperature (float):
            Sampling temperature used during generation. Higher values produce
            more diverse outputs, while lower values make results more
            deterministic.

        seed (int):
            Random seed used to ensure reproducible generation.
    """
    video_path: str
    character_name: str
    prompt_template: str
    model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    video_sampling_fps: float = 8.0
    max_new_tokens: int = 512
    temperature: float = 0.3
    seed: int = 424242


@dataclass(slots=True)
class GenerationInfo:
    """Metadata about the caption generation process, useful for debugging and analysis.

    Args:
        model_name_or_path (str):
            Name or local path of the vision-language model used for caption
            generation.
        video_path (str):
            Path to the video file for which the caption was generated.
        video_sampling_fps (float):
            Frame sampling rate used when extracting frames from the video.
        max_new_tokens (int):
            Maximum number of tokens the model was allowed to generate.
        temperature (float):
            Sampling temperature used during generation.
        seed (int):
            Random seed used to ensure reproducible generation.
    """
    model_name_or_path: str
    video_path: str
    video_sampling_fps: float
    max_new_tokens: int
    temperature: float
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CaptionResult:
    """
    Result of the caption generation process, including the generated caption and associated metadata.

    Args:
        caption (str):
            The generated caption describing the video content.
        resolved_prompt (str):
            The final prompt that was fed into the model after replacing the "{character_name}" token with
            the actual character name.
        generation_info (GenerationInfo):
            Metadata about the generation process, useful for debugging and analysis.
    """
    caption: str
    resolved_prompt: str
    generation_info: GenerationInfo
