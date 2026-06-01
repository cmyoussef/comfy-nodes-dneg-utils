"""DNEG Utils Nodes — ComfyUI custom node pack.

Small general utility nodes that do not yet justify dedicated repositories.
Add new node modules under ``_nodes/`` and register them in ``_ALL_NODES``.
"""

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
]

_FPT_LENS_NODE_REPLACEMENTS = [
    ("fpt_LensDistortUndistort", "FPT_LensDistortUndistort"),
    ("fpt_STMapDistort", "FPT_STMapDistort"),
    ("fpt_LensCalibInfo", "FPT_LensCalibInfo"),
    ("fpt_GenerateSTMap", "FPT_GenerateSTMap"),
    ("fpt_Unbulge", "FPT_Unbulge"),
]


class _DNEGUtilsExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        for old_node_id, new_node_id in _FPT_LENS_NODE_REPLACEMENTS:
            await api.node_replacement.register(
                io.NodeReplace(new_node_id=new_node_id, old_node_id=old_node_id)
            )

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return _ALL_NODES


async def comfy_entrypoint() -> _DNEGUtilsExtension:
    """ComfyUI V3 extension entrypoint — called by the ComfyUI loader."""
    return _DNEGUtilsExtension()
