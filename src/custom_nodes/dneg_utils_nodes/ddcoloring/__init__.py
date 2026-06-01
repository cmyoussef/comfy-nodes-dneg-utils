# src\comfyui_remote\custom_nodes\ddcoloring\__init__.py
"""ddcoloring package — ComfyUI node registration (API2 / V3)."""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .node_single import DDColorNode
from .node_sequence import DDColorSequenceNode


class DDColorExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DDColorNode, DDColorSequenceNode]


async def comfy_entrypoint() -> DDColorExtension:
    return DDColorExtension()


__all__ = ["comfy_entrypoint"]
