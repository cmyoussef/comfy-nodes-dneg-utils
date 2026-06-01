from __future__ import annotations

import gc
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .types import CaptionRequest, CaptionResult, GenerationInfo
from .utils import resolve_prompt

# Global generation semaphore.
#
# This prevents multiple caption nodes from running GPU inference concurrently
# within the same ComfyUI process. Running several large VL models in parallel
# can easily cause CUDA OOM, especially because each node loads the model
# independently (the model is intentionally not cached to allow VRAM to be
# released after each run).
#
# By serializing inference we ensure:
#   - only one model occupies GPU memory at a time
#   - VRAM spikes from parallel model loads are avoided
#   - workflows remain stable even when several caption nodes are triggered.
#
_GENERATION_SEMAPHORE = threading.Semaphore(1)


@dataclass(slots=True)
class LoadedModel:
    """Container for a loaded Qwen VL model and its processor."""

    model_name_or_path: str
    model: Qwen2_5_VLForConditionalGeneration
    processor: AutoProcessor


class CaptionService:
    """
    Single-run caption service for DN_QwenVideoCaptioner.

    Important design choice:
    The model is loaded for one request and explicitly released afterwards
    to avoid keeping VRAM occupied between workflow runs.
    """

    def caption_video(self, request: CaptionRequest) -> CaptionResult:
        resolved_prompt = resolve_prompt(
            prompt_template=request.prompt_template,
            character_name=request.character_name,
        )

        with _GENERATION_SEMAPHORE:
            loaded = self._load_model(request.model_name_or_path)

            try:
                caption_text = self._run_inference(
                    loaded_model=loaded.model,
                    processor=loaded.processor,
                    video_path=request.video_path,
                    resolved_prompt=resolved_prompt,
                    video_sampling_fps=request.video_sampling_fps,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    seed=request.seed,
                    max_pixels=getattr(request, "max_pixels", None),
                )
            finally:
                # Explicitly release model resources and clear GPU memory after inference to prevent OOM in subsequent runs.
                del loaded
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        caption_text = self._ensure_character_token(
            caption=caption_text,
            character_name=request.character_name,
        )

        generation_info = GenerationInfo(
            model_name_or_path=request.model_name_or_path,
            video_path=request.video_path,
            video_sampling_fps=request.video_sampling_fps,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            seed=request.seed,
        )

        return CaptionResult(
            caption=caption_text,
            resolved_prompt=resolved_prompt,
            generation_info=generation_info,
        )

    def _load_model(self, model_name_or_path: str) -> LoadedModel:
        """Load model and processor for a single request."""
        use_cuda = torch.cuda.is_available()
        torch_dtype = torch.bfloat16 if use_cuda else torch.float32

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto" if use_cuda else None,
        )
        processor = AutoProcessor.from_pretrained(model_name_or_path)

        if not use_cuda:
            model = model.to("cpu")

        model.eval()

        return LoadedModel(
            model_name_or_path=model_name_or_path,
            model=model,
            processor=processor,
        )

    def _run_inference(
        self,
        loaded_model: Any,
        processor: Any,
        video_path: str,
        resolved_prompt: str,
        video_sampling_fps: float,
        max_new_tokens: int,
        temperature: float,
        seed: int,
        max_pixels: int | None = None,
    ) -> str:
        self._set_seed(seed)

        video_uri = Path(video_path).expanduser().resolve().as_uri()

        video_item: Dict[str, Any] = {
            "type": "video",
            "video": video_uri,
            "fps": video_sampling_fps,
        }
        if max_pixels is not None:
            video_item["max_pixels"] = max_pixels

        messages = [
            {
                "role": "user",
                "content": [
                    video_item,
                    {"type": "text", "text": resolved_prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        device = next(loaded_model.parameters()).device
        inputs = inputs.to(device)

        do_sample = temperature > 0.0
        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature

        with torch.inference_mode():
            generated_ids = loaded_model.generate(
                **inputs,
                **generate_kwargs,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        del image_inputs
        del video_inputs
        del inputs
        del generated_ids
        del generated_ids_trimmed

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_text

    @staticmethod
    def _ensure_character_token(caption: str, character_name: str) -> str:
        """Ensure the final caption contains the character token."""
        if character_name in caption:
            return caption

        if caption:
            return f"{character_name}. {caption}"

        return character_name

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def result_to_json_dict(result: CaptionResult) -> Dict[str, Any]:
        return {
            "caption": result.caption,
            "resolved_prompt": result.resolved_prompt,
            "generation_info": result.generation_info.to_dict(),
        }

    @staticmethod
    def result_to_json_string(result: CaptionResult) -> str:
        return json.dumps(
            CaptionService.result_to_json_dict(result),
            ensure_ascii=False,
            indent=2,
        )
