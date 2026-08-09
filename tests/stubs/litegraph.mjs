/**
 * Minimal LiteGraph stand-ins, installed as globals before the module under test
 * is imported. Only the surface DN_GroupToggle actually touches is modelled.
 */

export class LGraphNode {
  constructor(title) {
    this.title = title;
    this.widgets = [];
    this.properties = {};
    this.size = [240, 60];
  }

  addWidget(type, name, value, callback) {
    const widget = { type, name, label: name, value, callback };
    this.widgets.push(widget);
    return widget;
  }

  addCustomWidget(widget) {
    this.widgets.push(widget);
    return widget;
  }

  computeSize() {
    return [240, 30 + this.widgets.length * 20];
  }

  setSize(size) {
    this.size = size;
  }

  setDirtyCanvas() {}
}

export function installGlobals() {
  globalThis.LGraphNode = LGraphNode;

  globalThis.LiteGraph = {
    ALWAYS: 0,
    NEVER: 2,
    NODE_WIDGET_HEIGHT: 20,
    WIDGET_BGCOLOR: "#222",
    WIDGET_OUTLINE_COLOR: "#666",
    WIDGET_TEXT_COLOR: "#DDD",
    WIDGET_SECONDARY_TEXT_COLOR: "#999",
    ContextMenu: class {},
  };
  globalThis.Path2D = class {
    constructor(d) {
      this.d = d;
    }
  };

  // The real palette values, so colour-name matching is tested against reality.
  globalThis.LGraphCanvas = {
    node_colors: {
      red: { groupcolor: "#A88" },
      green: { groupcolor: "#8A8" },
      blue: { groupcolor: "#88A" },
      pale_blue: { groupcolor: "#3f789e" },
      black: { groupcolor: "#444" },
    },
  };
}
