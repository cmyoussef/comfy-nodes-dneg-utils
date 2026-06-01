# src\comfyui_remote\custom_nodes\fpt_lens_distort\__init__.py
"""
FPT Lens Distortion Nodes for ComfyUI (API2 / V3)
Lens distortion and undistortion using OpenCV calibration or STMaps.
"""
from typing_extensions import override

from comfy_api.latest import ComfyAPI, ComfyExtension, io

from .nodes import (
    FPTLensDistortUndistort,
    FPTSTMapDistort,
    FPTLensCalibInfo,
    FPTGenerateSTMap,
    FPTUnbulge,
)

api = ComfyAPI()

_NODE_REPLACEMENTS = [
    ("fpt_LensDistortUndistort", "FPT_LensDistortUndistort"),
    ("fpt_STMapDistort", "FPT_STMapDistort"),
    ("fpt_LensCalibInfo", "FPT_LensCalibInfo"),
    ("fpt_GenerateSTMap", "FPT_GenerateSTMap"),
    ("fpt_Unbulge", "FPT_Unbulge"),
]


class FPTLensDistortExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        for old_node_id, new_node_id in _NODE_REPLACEMENTS:
            await api.node_replacement.register(
                io.NodeReplace(new_node_id=new_node_id, old_node_id=old_node_id)
            )

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FPTLensDistortUndistort,
            FPTSTMapDistort,
            FPTLensCalibInfo,
            FPTGenerateSTMap,
            FPTUnbulge,
        ]


async def comfy_entrypoint() -> FPTLensDistortExtension:
    return FPTLensDistortExtension()


__all__ = ["comfy_entrypoint"]