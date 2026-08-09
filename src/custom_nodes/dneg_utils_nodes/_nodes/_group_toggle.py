"""DN Group Toggle — server-side definition for a UI-only control node.

All of this node's behaviour lives in the frontend (``web/dneg_group_toggle.js``):
it lists the graph's groups and mutes or bypasses them. Nothing is computed here.

It is declared as a real node anyway, rather than a purely client-side virtual
node, for two reasons:

1. **It is never "missing".** ComfyUI decides a workflow references an unknown
   node by looking the type up in ``LiteGraph.registered_node_types``. A
   client-side-only node is added to that registry by an extension at runtime, so
   it is absent whenever the extension is late, disabled, or unsupported by the
   active renderer — which is exactly how rgthree's frontend-only nodes end up in
   the "Missing Node Packs" dialog under Nodes 2.0. A node that ships a schema is
   registered from ``object_info`` before any workflow loads, in every renderer.

2. **The settings stay reachable.** Declaring them as inputs makes them ordinary
   widgets, drawn and edited by whichever renderer is active. The alternative —
   LiteGraph node *properties* — is only reachable through the properties panel,
   which Nodes 2.0 does not provide.

The node declares no outputs, so nothing can connect to it and it is never
reachable from an output node. ``execute`` is therefore unreachable in practice;
it returns an empty result for completeness.
"""
from __future__ import annotations

from comfy_api.latest import io


ACTIONS = ["mute", "bypass"]
RESTRICTIONS = ["none", "max one", "always one"]
SORTS = ["position", "alphanumeric", "custom alphabet"]


class DN_GroupToggle(io.ComfyNode):
    """Mute or bypass whole graph groups from a single node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DN_GroupToggle",
            display_name="DN Group Toggle",
            category="DN/utils/graph",
            description=(
                "Collect the workflow's groups and switch them on or off from one node. "
                "Groups are matched by colour and/or title, and a restriction can enforce "
                "that at most one — or exactly one — is ever active, which makes a set of "
                "coloured groups behave like a radio-button switch. "
                "This node performs no computation and never takes part in a prompt."
            ),
            inputs=[
                io.Combo.Input(
                    "action",
                    socketless=True,
                    options=ACTIONS,
                    default="mute",
                    tooltip=(
                        "What a switched-off group is set to. 'mute' skips its nodes "
                        "entirely; 'bypass' passes inputs through to outputs where the "
                        "types allow it."
                    ),
                ),
                io.Combo.Input(
                    "restriction",
                    socketless=True,
                    options=RESTRICTIONS,
                    default="none",
                    tooltip=(
                        "'none' toggles each group independently. "
                        "'max one' switches the others off when you switch one on. "
                        "'always one' also refuses to switch off the last active group, "
                        "so the set behaves like radio buttons."
                    ),
                ),
                io.String.Input(
                    "match_colors",
                    socketless=True,
                    default="",
                    multiline=False,
                    tooltip=(
                        "Comma-separated group colours to include. Accepts ComfyUI colour "
                        "names (green, pale_blue, black) or hex (#8A8, #3f789e). "
                        "Leave empty to list every group. "
                        "Set a group's colour by right-clicking its title bar."
                    ),
                ),
                io.String.Input(
                    "match_title",
                    socketless=True,
                    advanced=True,
                    default="",
                    multiline=False,
                    tooltip=(
                        "Case-insensitive regular expression the group title must match, "
                        "applied on top of match_colors. Example: ^sdxl "
                        "Leave empty to accept any title."
                    ),
                ),
                io.Combo.Input(
                    "sort",
                    socketless=True,
                    options=SORTS,
                    default="position",
                    tooltip=(
                        "Row order: 'position' follows the groups' layout on the canvas, "
                        "'alphanumeric' sorts by title, 'custom alphabet' uses the "
                        "custom_sort_alphabet prefixes."
                    ),
                ),
                io.String.Input(
                    "custom_sort_alphabet",
                    socketless=True,
                    advanced=True,
                    default="",
                    multiline=False,
                    tooltip=(
                        "Used when sort is 'custom alphabet'. Comma-separated title "
                        "prefixes, for example: sdxl,flux,wan "
                        "Groups matching an earlier prefix sort first; anything unmatched "
                        "follows, ordered alphanumerically."
                    ),
                ),
                io.Boolean.Input(
                    "show_nav",
                    socketless=True,
                    advanced=True,
                    default=True,
                    tooltip=(
                        "Draw a navigation arrow on each row that centres the canvas on "
                        "that group. Only available in the classic canvas renderer."
                    ),
                ),
                io.Boolean.Input(
                    "include_subgraphs",
                    socketless=True,
                    advanced=True,
                    default=True,
                    tooltip="Also list groups that live inside subgraphs.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        """Unreachable — the node has no outputs, so no prompt ever includes it."""
        return io.NodeOutput()
