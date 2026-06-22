/**
 * CREMOTE_DeepHDR — frontend companion extension.
 *
 * Greys out the `saturation_threshold` widget whenever a MASK input is
 * connected, because the threshold is only used for auto-mask computation
 * and is ignored when an explicit mask is provided.
 *
 * Uses LiteGraph's onConnectionsChange hook; no server round-trips needed.
 */
import { app } from "/scripts/app.js";

const NODE_TYPE = "CREMOTE_DeepHDR";
const MASK_INPUT_NAME = "mask";
const THRESHOLD_WIDGET = "saturation_threshold";

// LiteGraph constant: connection type 1 = INPUT slot.
const LITEGRAPH_INPUT = 1;

function isDeepHDRNode(node) {
    return node && (node.type === NODE_TYPE || node.comfyClass === NODE_TYPE);
}

function findWidget(node, name) {
    return (node.widgets || []).find(w => w.name === name) ?? null;
}

/** Return the slot index of the "mask" input, or -1 if not found. */
function getMaskSlotIndex(node) {
    const inputs = node.inputs || [];
    for (let i = 0; i < inputs.length; i++) {
        if (inputs[i].name === MASK_INPUT_NAME) return i;
    }
    return -1;
}

function isMaskConnected(node) {
    const idx = getMaskSlotIndex(node);
    if (idx < 0) return false;
    const input = (node.inputs || [])[idx];
    return input != null && input.link != null;
}

/**
 * Disable saturation_threshold when mask is connected; enable otherwise.
 * widget.disabled = true is the same pattern used by the IvyStyler
 * version_number widget.
 */
function syncThresholdWidget(node) {
    const widget = findWidget(node, THRESHOLD_WIDGET);
    if (!widget) return;
    widget.disabled = isMaskConnected(node);
}

app.registerExtension({
    name: "dneg.deep_hdr.mask_aware_ui",

    nodeCreated(node) {
        if (!isDeepHDRNode(node)) return;

        // Defer initial sync so ComfyUI has finished wiring restored connections.
        requestAnimationFrame(() => syncThresholdWidget(node));

        const origOnConnectionsChange = node.onConnectionsChange;

        node.onConnectionsChange = function(type, slotIndex, connected, link_info, input_info) {
            if (typeof origOnConnectionsChange === "function") {
                origOnConnectionsChange.call(this, type, slotIndex, connected, link_info, input_info);
            }

            if (type !== LITEGRAPH_INPUT) return;

            const maskIdx = getMaskSlotIndex(node);
            if (maskIdx >= 0 && slotIndex === maskIdx) {
                syncThresholdWidget(node);
                app.graph?.setDirtyCanvas(true, false);
            }
        };
    },
});
