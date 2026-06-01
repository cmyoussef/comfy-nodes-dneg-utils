from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .nodes.cremote_qwen_captioning import DN_QwenVideoCaptioner


class DnQwenExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DN_QwenVideoCaptioner,
        ]


async def comfy_entrypoint() -> DnQwenExtension:
    return DnQwenExtension()


__all__ = ["comfy_entrypoint"]
