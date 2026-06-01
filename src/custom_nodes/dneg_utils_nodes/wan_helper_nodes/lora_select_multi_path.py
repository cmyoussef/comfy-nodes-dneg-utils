"""
LoRA Select Multi (Path) node for ComfyUI.

A variant of WanVideoLoraSelectMulti that accepts full file paths
instead of selecting from ComfyUI/models/loras. Delegates the heavy
lifting to the original WanVideoWrapper node.
"""
from __future__ import annotations

import os
import logging

from comfy_api.latest import io

logger = logging.getLogger(__name__)

try:
    from ComfyUI.custom_nodes.ComfyUI_WanVideoWrapper.nodes_model_loading import (
        WanVideoLoraSelectMulti as _WanVideoLoraSelectMulti,
    )
except Exception:
    _WanVideoLoraSelectMulti = None
    logger.debug(
        "WanVideoLoraSelectMulti is not available at import time; "
        "will retry lazily during execution if needed."
    )


def _get_original_node_class():
    """Best-effort lazy import for WanVideoWrapper load-order issues."""
    global _WanVideoLoraSelectMulti
    if _WanVideoLoraSelectMulti is not None:
        return _WanVideoLoraSelectMulti

    try:
        from ComfyUI.custom_nodes.ComfyUI_WanVideoWrapper.nodes_model_loading import (
            WanVideoLoraSelectMulti as _LoadedWanVideoLoraSelectMulti,
        )
    except Exception:
        return None

    _WanVideoLoraSelectMulti = _LoadedWanVideoLoraSelectMulti
    return _WanVideoLoraSelectMulti

_STRENGTH_KWARGS = dict(default=1.0, min=-10.0, max=10.0, step=0.0001)
_PATH_TOOLTIP = "Full path to a LoRA .safetensors file, or leave empty to skip"


class WanHelper_LoraSelectMultiPath(io.ComfyNode):
    """Select multiple LoRA models by full file path."""

    DESCRIPTION = (
        "Select up to 5 LoRA models by full file path.\n"
        "Works identically to WanVideo Lora Select Multi but accepts\n"
        "arbitrary paths instead of the ComfyUI/models/loras dropdown."
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        inputs = []
        for i in range(5):
            inputs.append(
                io.String.Input(
                    f"lora_{i}",
                    default="",
                    multiline=False,
                    tooltip=_PATH_TOOLTIP,
                )
            )
            inputs.append(
                io.Float.Input(
                    f"strength_{i}",
                    tooltip="LORA strength, set to 0.0 to unmerge the LORA",
                    **_STRENGTH_KWARGS,
                )
            )

        inputs.extend([
            io.Custom("WANVIDLORA").Input(
                "prev_lora",
                optional=True,
                tooltip="For loading multiple LoRAs",
            ),
            io.Custom("SELECTEDBLOCKS").Input(
                "blocks",
                optional=True,
            ),
            io.Boolean.Input(
                "low_mem_load",
                default=False,
                optional=True,
                tooltip="Load the LORA model with less VRAM usage, slower loading. No effect if merge_loras is False",
            ),
            io.Boolean.Input(
                "merge_loras",
                default=True,
                optional=True,
                tooltip="Merge LoRAs into the model, otherwise they are loaded on the fly. Always disabled for GGUF and scaled fp8 models. This affects ALL LoRAs, not just the current one",
            ),
        ])

        return io.Schema(
            node_id="WanHelper_LoraSelectMultiPath",
            display_name="WanHelper: Lora Select Multi (Path)",
            category="WanHelper/lora",
            description=cls.DESCRIPTION,
            inputs=inputs,
            outputs=[
                io.Custom("WANVIDLORA").Output(
                    "lora",
                    tooltip="LoRA configuration list",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        lora_0: str = "",
        strength_0: float = 1.0,
        lora_1: str = "",
        strength_1: float = 1.0,
        lora_2: str = "",
        strength_2: float = 1.0,
        lora_3: str = "",
        strength_3: float = 1.0,
        lora_4: str = "",
        strength_4: float = 1.0,
        prev_lora=None,
        blocks=None,
        low_mem_load: bool = False,
        merge_loras: bool = True,
    ) -> io.NodeOutput:
        if blocks is None:
            blocks = {}

        # Resolve paths: replace each non-empty path string with the full
        # path so the original node's getlorapath receives valid values.
        # We override the path resolution that normally goes through
        # folder_paths by feeding absolute paths and then patching the
        # result afterwards.
        if not merge_loras:
            low_mem_load = False

        loras_list = list(prev_lora) if prev_lora else []

        lora_inputs = [
            (lora_0, strength_0),
            (lora_1, strength_1),
            (lora_2, strength_2),
            (lora_3, strength_3),
            (lora_4, strength_4),
        ]

        for lora_path, strength in lora_inputs:
            s = round(strength, 4) if not isinstance(strength, list) else strength
            if not lora_path or not lora_path.strip() or s == 0.0:
                continue
            lora_path = lora_path.strip()
            if not os.path.isfile(lora_path):
                raise FileNotFoundError(f"LoRA file not found: {lora_path}")
            loras_list.append({
                "path": lora_path,
                "strength": s,
                "name": os.path.splitext(os.path.basename(lora_path))[0],
                "blocks": blocks.get("selected_blocks", {}),
                "layer_filter": blocks.get("layer_filter", ""),
                "low_mem_load": low_mem_load,
                "merge_loras": merge_loras,
            })

        if len(loras_list) == 0:
            return io.NodeOutput(None)
        return io.NodeOutput(loras_list)
