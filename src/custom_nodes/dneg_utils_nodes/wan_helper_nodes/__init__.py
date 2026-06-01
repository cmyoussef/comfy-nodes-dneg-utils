"""
Wan Helper Nodes

Custom nodes for the Wan video generation model, including Alembic camera loading,
face landmark preview, frame padding utilities, and position-pass helpers.
"""
import logging

from typing_extensions import override

from comfy_api.latest import ComfyAPI, ComfyExtension, io

from .wan_alembic_camera import WanHelper_WanAlembicCamera
from .landmark_preview import WanHelper_LandmarkPreview
from .frame_padder import WanHelper_WanFramePadder
from .frame_extender import WanHelper_WanFrameExtender
from .mask_ramp import WanHelper_MaskRamp
from .depth_normalize import WanHelper_DepthNormalize
from .position_relative import (
    WanHelper_GetLocatorPosition,
    WanHelper_ListAlembicLocators,
    WanHelper_NormalizePositionPass,
    WanHelper_WorldPositionToHeadRelative,
)
from .lora_select_multi_path import WanHelper_LoraSelectMultiPath

logger = logging.getLogger(__name__)

try:
    from .delayed_masking import WanHelper_DelayedMasking
except Exception:  # pragma: no cover - depends on Comfy runtime environment
    WanHelper_DelayedMasking = None
    logger.exception("Failed to import WanHelper_DelayedMasking; continuing without this node.")

api = ComfyAPI()

_NODE_REPLACEMENTS = [
    ("WanHelper_WanAlembicCamera", "WanHelper_WanAlembicCamera"),
    ("WanHelper_LandmarkPreview", "WanHelper_LandmarkPreview"),
    ("WanHelper_WanFramePadder", "WanHelper_WanFramePadder"),
    ("WanHelper_WanFrameExtender", "WanHelper_WanFrameExtender"),
    ("WanHelper_MaskRamp", "WanHelper_MaskRamp"),
    ("WanHelper_DepthNormalize", "WanHelper_DepthNormalize"),
    ("WanHelper_WorldPositionToHeadRelative", "WanHelper_WorldPositionToHeadRelative"),
    ("WanHelper_ListAlembicLocators", "WanHelper_ListAlembicLocators"),
    ("WanHelper_GetLocatorPosition", "WanHelper_GetLocatorPosition"),
    ("WanHelper_NormalizePositionPass", "WanHelper_NormalizePositionPass"),
    ("WanHelper_LoraSelectMultiPath", "WanHelper_LoraSelectMultiPath"),
]


class WanHelperExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        for old_node_id, new_node_id in _NODE_REPLACEMENTS:
            await api.node_replacement.register(
                io.NodeReplace(new_node_id=new_node_id, old_node_id=old_node_id)
            )

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        node_list: list[type[io.ComfyNode]] = [
            WanHelper_WanAlembicCamera,
            WanHelper_LandmarkPreview,
            WanHelper_WanFramePadder,
            WanHelper_WanFrameExtender,
            WanHelper_MaskRamp,
            WanHelper_DepthNormalize,
            WanHelper_WorldPositionToHeadRelative,
            WanHelper_ListAlembicLocators,
            WanHelper_GetLocatorPosition,
            WanHelper_NormalizePositionPass,
            WanHelper_LoraSelectMultiPath,
        ]
        if WanHelper_DelayedMasking is not None:
            node_list.append(WanHelper_DelayedMasking)
        return node_list


async def comfy_entrypoint() -> WanHelperExtension:
    return WanHelperExtension()


__all__ = ["comfy_entrypoint"]
