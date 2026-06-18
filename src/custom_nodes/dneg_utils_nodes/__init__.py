"""DNEG Utils Nodes — ComfyUI custom node pack.

Small general utility nodes that do not yet justify dedicated repositories.
Add new node modules under ``_nodes/`` and register them in ``_ALL_NODES``.
"""
import logging

from typing_extensions import override

from comfy_api.latest import ComfyAPI, ComfyExtension, io

from ._nodes._list_files import DN_ListFiles
from .ddcoloring.node_single import DDColorNode
from .ddcoloring.node_sequence import DDColorSequenceNode
from .fpt_lens_distort.nodes import (
    FPTLensDistortUndistort,
    FPTSTMapDistort,
    FPTLensCalibInfo,
    FPTGenerateSTMap,
    FPTUnbulge,
)
from .dn_qwen.nodes.cremote_qwen_captioning import DN_QwenVideoCaptioner

from .wan_helper_nodes.wan_alembic_camera import WanHelper_WanAlembicCamera
from .wan_helper_nodes.landmark_preview import WanHelper_LandmarkPreview
from .wan_helper_nodes.frame_padder import WanHelper_WanFramePadder
from .wan_helper_nodes.frame_extender import WanHelper_WanFrameExtender
from .wan_helper_nodes.mask_ramp import WanHelper_MaskRamp
from .wan_helper_nodes.depth_normalize import WanHelper_DepthNormalize
from .wan_helper_nodes.position_relative import (
    WanHelper_GetLocatorPosition,
    WanHelper_ListAlembicLocators,
    WanHelper_NormalizePositionPass,
    WanHelper_WorldPositionToHeadRelative,
)
from .wan_helper_nodes.lora_select_multi_path import WanHelper_LoraSelectMultiPath

_logger = logging.getLogger(__name__)
try:
    from .wan_helper_nodes.delayed_masking import WanHelper_DelayedMasking
except Exception:  # pragma: no cover - depends on Comfy runtime environment
    WanHelper_DelayedMasking = None
    _logger.exception("Failed to import WanHelper_DelayedMasking; continuing without this node.")

try:
    from .deep_hdr.node import CREMOTE_DeepHDR
except Exception:  # pragma: no cover - depends on deep_hdr bob package
    CREMOTE_DeepHDR = None
    _logger.exception("Failed to import CREMOTE_DeepHDR; continuing without this node. "
                      "Ensure 'deep_hdr' is deployed with target 'ml_cv' in dneg.json.")

api = ComfyAPI()

_ALL_NODES = [
    DN_ListFiles,
    DDColorNode,
    DDColorSequenceNode,
    FPTLensDistortUndistort,
    FPTSTMapDistort,
    FPTLensCalibInfo,
    FPTGenerateSTMap,
    FPTUnbulge,
    DN_QwenVideoCaptioner,
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

_FPT_LENS_NODE_REPLACEMENTS = [
    ("fpt_LensDistortUndistort", "FPT_LensDistortUndistort"),
    ("fpt_STMapDistort", "FPT_STMapDistort"),
    ("fpt_LensCalibInfo", "FPT_LensCalibInfo"),
    ("fpt_GenerateSTMap", "FPT_GenerateSTMap"),
    ("fpt_Unbulge", "FPT_Unbulge"),
]

_WAN_HELPER_NODE_REPLACEMENTS = [
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


class _DNEGUtilsExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        for old_node_id, new_node_id in _FPT_LENS_NODE_REPLACEMENTS:
            await api.node_replacement.register(
                io.NodeReplace(new_node_id=new_node_id, old_node_id=old_node_id)
            )
        for old_node_id, new_node_id in _WAN_HELPER_NODE_REPLACEMENTS:
            await api.node_replacement.register(
                io.NodeReplace(new_node_id=new_node_id, old_node_id=old_node_id)
            )

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        node_list = list(_ALL_NODES)
        if WanHelper_DelayedMasking is not None:
            node_list.append(WanHelper_DelayedMasking)
        if CREMOTE_DeepHDR is not None:
            node_list.append(CREMOTE_DeepHDR)
        return node_list


async def comfy_entrypoint() -> _DNEGUtilsExtension:
    """ComfyUI V3 extension entrypoint — called by the ComfyUI loader."""
    return _DNEGUtilsExtension()
