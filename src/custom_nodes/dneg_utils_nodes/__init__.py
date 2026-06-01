"""DNEG Utils Nodes — ComfyUI custom node pack.

Small general utility nodes that do not yet justify dedicated repositories.
Add new node modules under ``_nodes/`` and register them in ``_ALL_NODES``.
"""

from comfy_api.latest import ComfyExtension, io

from ._nodes._list_files import DN_ListFiles

_ALL_NODES = [
    DN_ListFiles,
]


class _DNEGUtilsExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return _ALL_NODES


async def comfy_entrypoint() -> _DNEGUtilsExtension:
    """ComfyUI V3 extension entrypoint — called by the ComfyUI loader."""
    return _DNEGUtilsExtension()
