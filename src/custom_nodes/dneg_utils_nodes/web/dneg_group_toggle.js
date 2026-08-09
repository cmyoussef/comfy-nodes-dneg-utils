/**
 * DN Group Toggle — mute or bypass whole graph groups from a single node.
 *
 * The node itself is declared server-side in ``_nodes/_group_toggle.py`` and does
 * nothing there; this file is all of its behaviour. It scans the current graph
 * for groups, appends one toggle per matched group below the settings widgets,
 * and writes `mode` onto the nodes inside a group when a toggle is clicked.
 *
 * This replaces rgthree's "Fast Groups Muter/Bypasser" for our workflows. Three
 * things are deliberately done differently, each fixing a failure we hit:
 *
 * 1. **The node has a schema, so it is never "missing".** ComfyUI resolves a
 *    workflow's node types against `LiteGraph.registered_node_types`. Nodes that
 *    only exist client-side are put there by an extension at runtime, so they
 *    vanish whenever the extension is late, disabled, or unsupported by the
 *    active renderer — which is how rgthree's Label and Fast Groups nodes land in
 *    the "Missing Node Packs" dialog under Nodes 2.0.
 *
 * 2. **Settings are widgets, not LiteGraph properties.** Widgets are drawn and
 *    edited by whichever renderer is active. Properties need the properties
 *    panel, which Nodes 2.0 does not provide.
 *
 * 3. **Group state is derived, never cached.** Whether a group is active is read
 *    from the modes of the nodes inside it (`isGroupActive`) every single time.
 *    rgthree cached it on the widget, and when their widget `value` changed from
 *    a boolean to a `{toggled}` object their "always one" guard silently became a
 *    no-op (`!w.value` on an object is always false) — the last enabled group
 *    could be switched off and the workflow ended up fully muted.
 *
 * Settings (widgets on the node):
 *   action                "mute" | "bypass" — what a switched-off group becomes.
 *   restriction           "none" | "max one" | "always one".
 *   match_colors          Comma-separated group colours to include. ComfyUI colour
 *                         names ("green", "pale_blue") or hex ("#3f789e").
 *                         Empty = every group.
 *   match_title           Case-insensitive regex the group title must match.
 *   sort                  "position" | "alphanumeric" | "custom alphabet".
 *   custom_sort_alphabet  Comma-separated title prefixes, e.g. "sdxl,flux,wan".
 *   include_subgraphs     Also list groups inside subgraphs.
 */
import { app } from "/scripts/app.js";

const EXTENSION_NAME = "dneg.utils.group_toggle";
const NODE_TYPE = "DN_GroupToggle";

/** ComfyUI's "bypass" mode. LiteGraph only defines ALWAYS (0) and NEVER (2). */
const MODE_BYPASS = 4;

/** How often the graph is re-scanned for group / membership changes. */
const SCAN_INTERVAL_MS = 400;

/** Marks a widget as a generated group row rather than a settings widget. */
const ROW_FLAG = "__dnGroupRow";

/* -------------------------------------------------------------------------- */
/* Graph helpers                                                              */
/* -------------------------------------------------------------------------- */

/** The graph the canvas is currently showing (a subgraph, if the user drilled in). */
function currentGraph() {
  return app.canvas?.getCurrentGraph?.() ?? app.graph ?? null;
}

/** Every group we could offer a toggle for; per-node filtering happens later. */
function collectGroups(includeSubgraphs) {
  const graph = currentGraph();
  if (!graph) return [];

  const groups = [...(graph.groups ?? graph._groups ?? [])];
  if (includeSubgraphs && graph.subgraphs) {
    for (const subgraph of graph.subgraphs.values()) {
      groups.push(...(subgraph.groups ?? subgraph._groups ?? []));
    }
  }
  return groups;
}

/**
 * The nodes a group contains. Newer frontends keep a `_children` set that also
 * holds reroutes and nested groups, so filter down to real nodes; older ones
 * expose a plain array.
 */
function groupNodes(group) {
  if (group?._children) {
    return Array.from(group._children).filter((child) => child instanceof LGraphNode);
  }
  return group?.nodes ?? group?._nodes ?? [];
}

/**
 * Nodes a toggle may write `mode` onto. Controller nodes are excluded so a
 * DN Group Toggle sitting inside a group it manages can never mute itself into
 * an unrecoverable state.
 */
function targetNodes(group) {
  return groupNodes(group).filter((node) => node.type !== NODE_TYPE);
}

/**
 * A group is active when anything inside it still executes. Single source of
 * truth for toggle state — nothing is cached on the widget.
 */
function isGroupActive(group) {
  return targetNodes(group).some((node) => node.mode === LiteGraph.ALWAYS);
}

/** Stable identity for a group across workflow reloads, used to match widgets. */
function groupKey(group) {
  const graphId = group?.graph?.id ?? "root";
  return `${graphId}:${group?.id ?? group?.title ?? ""}`;
}

function groupPos(group) {
  return group?.pos ?? group?._pos ?? [0, 0];
}

/**
 * Expand a colour to a comparable `#rrggbb`. Accepts ComfyUI palette names,
 * 3-digit hex and 6-digit hex, so `match_colors: "green"` matches a group saved
 * as "#8A8".
 */
function normalizeColor(value) {
  if (!value) return "";

  let color = String(value).trim().toLowerCase();
  const preset = LGraphCanvas.node_colors?.[color];
  if (preset) color = preset.groupcolor;

  color = color.replace("#", "").toLowerCase();
  if (color.length === 3) color = color.replace(/(.)(.)(.)/, "$1$1$2$2$3$3");
  return color ? `#${color}` : "";
}

function splitList(value) {
  return String(value ?? "")
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

/**
 * Parse `custom_sort_alphabet`, matching rgthree: a comma-separated string is a
 * list of title prefixes ("sdxl,flux,wan"); a string with no comma is split into
 * single characters ("zyxw"), which is the terse form for reversing the alphabet.
 */
function parseAlphabet(value) {
  const raw = String(value ?? "").replace(/\n/g, "").trim();
  if (!raw) return [];
  return raw.includes(",")
    ? raw.toLowerCase().split(",").map((entry) => entry.trim()).filter(Boolean)
    : raw.toLowerCase().split("");
}

/**
 * True when ComfyUI is rendering nodes with the Vue "Nodes 2.0" renderer, which
 * does not draw canvas widgets. Rows fall back to plain toggles there.
 */
function vueNodesEnabled() {
  try {
    return app.ui?.settings?.getSettingValue?.("Comfy.VueNodes.Enabled") === true;
  } catch (err) {
    return false;
  }
}

/** Canvas is zoomed far enough out that detail is not worth drawing. */
function isLowQuality() {
  return (app.canvas?.ds?.scale ?? 1) <= 0.5;
}

/** Truncate `text` with an ellipsis so it fits `maxWidth`. */
function fitString(ctx, text, maxWidth) {
  if (ctx.measureText(text).width <= maxWidth) return text;

  const ellipsis = "…";
  const ellipsisWidth = ctx.measureText(ellipsis).width;
  if (maxWidth <= ellipsisWidth) return text;

  let low = 0;
  let high = text.length;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if (ctx.measureText(text.substring(0, mid)).width <= maxWidth - ellipsisWidth) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }
  return text.substring(0, low) + ellipsis;
}

/** Chain a prototype hook without clobbering whatever is already there. */
function chain(prototype, name, handler) {
  const previous = prototype[name];
  prototype[name] = function (...args) {
    const result = previous?.apply(this, args);
    return handler.apply(this, args) ?? result;
  };
}

/* -------------------------------------------------------------------------- */
/* Scanner                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * One shared timer drives every DN Group Toggle on the canvas: recompute group
 * membership once, then let each node re-read the graph. It only runs while at
 * least one toggle node exists.
 */
const scanner = {
  nodes: new Set(),
  timer: null,
  pendingDelayMs: Infinity,

  register(node) {
    this.nodes.add(node);
    // Widgets are populated after construction, so wait a tick before the first read.
    this.schedule(16);
  },

  unregister(node) {
    this.nodes.delete(node);
    if (!this.nodes.size) this.stop();
  },

  stop() {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    this.pendingDelayMs = Infinity;
  },

  schedule(delayMs = SCAN_INTERVAL_MS) {
    if (!this.nodes.size) return;
    if (this.timer !== null) {
      if (delayMs >= this.pendingDelayMs) return; // something sooner is already queued
      clearTimeout(this.timer);
    }
    this.pendingDelayMs = delayMs;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.pendingDelayMs = Infinity;
      this.scan();
      this.schedule();
    }, delayMs);
  },

  scan() {
    // Group bounds are in flux mid-drag; membership computed now would be wrong.
    if (!app.canvas || app.canvas.isDragging) return;

    for (const group of collectGroups(true)) {
      try {
        group.recomputeInsideNodes();
      } catch (err) {
        console.error(`[${NODE_TYPE}] recomputeInsideNodes failed`, err);
      }
    }

    for (const node of this.nodes) {
      try {
        node.dnRefresh();
      } catch (err) {
        console.error(`[${NODE_TYPE}] refresh failed`, err);
      }
    }
  },
};

/* -------------------------------------------------------------------------- */
/* Row widget                                                                  */
/* -------------------------------------------------------------------------- */

/** Horizontal band on the right of a row reserved for the nav arrow. */
const NAV_HIT_WIDTH = 15 + 28 + 1;

/**
 * One group row, drawn on the canvas: title on the left, then a yes/no label, a
 * round toggle, and an optional navigation arrow. Geometry mirrors rgthree's
 * Fast Groups rows so the node reads identically.
 *
 * `value` is a plain boolean, and it is re-derived from the graph on every scan —
 * it is never the authority for a restriction decision.
 */
class GroupRowWidget {
  constructor(node, row) {
    this.type = "custom";
    this.name = "";
    this.label = "";
    this.value = false;
    this.options = { on: "yes", off: "no", serialize: false };
    this.node = node;
    this.row = row;
    this[ROW_FLAG] = true;
  }

  computeSize(width) {
    return [width, LiteGraph.NODE_WIDGET_HEIGHT];
  }

  serializeValue() {
    return this.value;
  }

  /** Uniform entry point so callers do not care which widget kind a row uses. */
  callback(value) {
    this.node.dnRowToggled(this.row, value);
  }

  draw(ctx, node, width, posY, height) {
    const lowQuality = isLowQuality();
    const margin = 15;
    const showNav = node.dnSetting("show_nav", true) !== false;

    // Background pill.
    ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
    ctx.beginPath();
    ctx.roundRect(margin, posY, width - margin * 2, height, lowQuality ? [0] : [height * 0.5]);
    ctx.fill();
    if (!lowQuality) ctx.stroke();

    // Right to left: the title on the left takes whatever space is left over.
    let currentX = width - margin;

    if (showNav && !lowQuality) {
      currentX -= 7;
      const midY = posY + height * 0.5;
      ctx.fillStyle = ctx.strokeStyle = "#89A";
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      const arrow = new Path2D(`M${currentX} ${midY} l -7 6 v -3 h -7 v -6 h 7 v -3 z`);
      ctx.fill(arrow);
      ctx.stroke(arrow);
      currentX -= 14;

      currentX -= 7;
      ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
      ctx.stroke(new Path2D(`M ${currentX} ${posY} v ${height}`));
    } else if (showNav && lowQuality) {
      currentX -= 28;
    }

    // The toggle dot.
    currentX -= 7;
    ctx.fillStyle = this.value ? "#89A" : "#333";
    ctx.beginPath();
    const toggleRadius = height * 0.36;
    ctx.arc(currentX - toggleRadius, posY + height * 0.5, toggleRadius, 0, Math.PI * 2);
    ctx.fill();
    currentX -= toggleRadius * 2;

    if (lowQuality) return;

    currentX -= 4;
    ctx.textAlign = "right";
    ctx.fillStyle = this.value ? LiteGraph.WIDGET_TEXT_COLOR : LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
    const labelOn = this.options.on ?? "true";
    const labelOff = this.options.off ?? "false";
    ctx.fillText(this.value ? labelOn : labelOff, currentX, posY + height * 0.7);
    currentX -= Math.max(ctx.measureText(labelOn).width, ctx.measureText(labelOff).width);

    currentX -= 7;
    ctx.textAlign = "left";
    const maxLabelWidth = width - margin - 10 - (width - currentX);
    ctx.fillText(fitString(ctx, this.label ?? "", maxLabelWidth), margin + 10, posY + height * 0.7);
  }

  mouse(event, pos, node) {
    if (event?.type !== "pointerdown") return true;

    const showNav = node.dnSetting("show_nav", true) !== false;
    if (showNav && pos[0] >= node.size[0] - NAV_HIT_WIDTH) {
      // The arrow is not drawn when zoomed out, so do not act on its hit area.
      if (!isLowQuality()) node.dnNavigateTo(this.row.group);
    } else {
      this.callback(!this.value);
    }
    return true;
  }
}

/* -------------------------------------------------------------------------- */
/* Node behaviour                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Mixed into the registered node's prototype. Everything is `dn`-prefixed
 * because this shares a prototype with LGraphNode and ComfyUI's own additions.
 */
const groupToggleMethods = {
  /** Read a settings widget by schema input name. */
  dnSetting(name, fallback) {
    const widget = this.widgets?.find((w) => !w[ROW_FLAG] && w.name === name);
    return widget ? widget.value : fallback;
  },

  dnSettingWidgets() {
    return (this.widgets ?? []).filter((widget) => !widget[ROW_FLAG]);
  },

  /* -- matching -- */

  /** Groups this node offers a toggle for, filtered and sorted by its settings. */
  dnMatchedGroups() {
    const groups = collectGroups(this.dnSetting("include_subgraphs", true) !== false);
    return this.dnSortGroups(groups.filter((group) => this.dnGroupMatches(group)));
  },

  dnGroupMatches(group) {
    const wanted = splitList(this.dnSetting("match_colors", "")).map(normalizeColor).filter(Boolean);
    if (wanted.length) {
      const actual = normalizeColor(group.color);
      if (!actual || !wanted.includes(actual)) return false;
    }

    const pattern = String(this.dnSetting("match_title", "") ?? "").trim();
    if (pattern) {
      try {
        if (!new RegExp(pattern, "i").test(group.title ?? "")) return false;
      } catch (err) {
        // An in-progress regex while typing should not blank the node out.
        console.warn(`[${NODE_TYPE}] invalid match_title regex: ${pattern}`, err);
      }
    }

    return true;
  },

  dnSortGroups(groups) {
    const alphabet = parseAlphabet(this.dnSetting("custom_sort_alphabet", ""));
    const sort = this.dnSetting("sort", "position");

    if (sort === "custom alphabet" && alphabet.length) {
      const rank = (group) => {
        const title = (group.title ?? "").toLowerCase();
        const index = alphabet.findIndex((prefix) => title.startsWith(prefix));
        return index < 0 ? alphabet.length : index;
      };
      return groups.sort(
        (a, b) => rank(a) - rank(b) || (a.title ?? "").localeCompare(b.title ?? ""),
      );
    }

    if (sort === "position") {
      // Snap to a 30px grid first so a roughly-aligned row of groups reads
      // left-to-right instead of by pixel-exact y.
      return groups.sort((a, b) => {
        const [ax, ay] = groupPos(a);
        const [bx, by] = groupPos(b);
        const rowDelta = Math.floor(ay / 30) - Math.floor(by / 30);
        return rowDelta || Math.floor(ax / 30) - Math.floor(bx / 30);
      });
    }

    return groups.sort((a, b) => (a.title ?? "").localeCompare(b.title ?? ""));
  },

  /* -- rendering -- */

  /**
   * Build the widget for one row. The canvas renderer gets rgthree-style rows
   * with an inline nav arrow; Nodes 2.0 cannot draw those, so it gets a plain
   * toggle and reaches navigation through the node's context menu instead.
   */
  dnCreateRowWidget(row) {
    if (vueNodesEnabled()) {
      const node = this;
      const widget = this.addWidget("toggle", "", false, (value) => node.dnRowToggled(row, value));
      widget[ROW_FLAG] = true;
      widget.options = { ...(widget.options ?? {}), on: "yes", off: "no", serialize: false };
      return widget;
    }

    const widget = new GroupRowWidget(this, row);
    if (typeof this.addCustomWidget === "function") {
      this.addCustomWidget(widget);
    } else {
      this.widgets.push(widget);
    }
    return widget;
  },

  /** Reconcile one toggle widget per matched group. Called by the shared scanner. */
  dnRefresh() {
    this.dnRows = this.dnRows ?? [];

    const rows = [];
    let structureChanged = false;
    let valuesChanged = false;

    for (const group of this.dnMatchedGroups()) {
      const key = groupKey(group);
      let row = this.dnRows.find((candidate) => candidate.key === key);

      if (!row) {
        row = { key, group, widget: null };
        row.widget = this.dnCreateRowWidget(row);
        structureChanged = true;
      }

      // The group object is rebuilt on workflow load, and titles can be renamed.
      row.group = group;
      const label = group.title ?? "";
      if (row.widget.label !== label || row.widget.name !== label) {
        row.widget.name = label;
        row.widget.label = label;
        structureChanged = true;
      }

      const active = isGroupActive(group);
      if (row.widget.value !== active) {
        row.widget.value = active;
        valuesChanged = true;
      }

      rows.push(row);
    }

    structureChanged =
      structureChanged ||
      rows.length !== this.dnRows.length ||
      rows.some((row, index) => row !== this.dnRows[index]);

    this.dnRows = rows;
    // Settings stay on top; generated rows follow, in match order.
    this.widgets = [...this.dnSettingWidgets(), ...rows.map((row) => row.widget)];

    if (structureChanged) {
      const [minWidth, minHeight] = this.computeSize();
      this.setSize([Math.max(this.size?.[0] ?? 0, minWidth), minHeight]);
    }
    if (structureChanged || valuesChanged) {
      this.setDirtyCanvas(true, structureChanged);
    }
  },

  /* -- toggling -- */

  /** Write the on/off mode onto every node in a group. */
  dnApplyToGroup(group, active, action) {
    const off = (action ?? this.dnSetting("action", "mute")) === "bypass" ? MODE_BYPASS : LiteGraph.NEVER;
    const mode = active ? LiteGraph.ALWAYS : off;
    for (const node of targetNodes(group)) {
      node.mode = mode;
    }
  },

  /**
   * A row was clicked. The toggle widget has already flipped its own `value`,
   * so a refused change has to write it back.
   */
  dnRowToggled(row, value) {
    const restriction = this.dnSetting("restriction", "none");

    if (!value && restriction === "always one") {
      // Refuse to switch off the last group standing. Read the graph rather than
      // sibling widget values, so an out-of-band mute cannot fool the check.
      const otherActive = this.dnRows.some(
        (other) => other !== row && isGroupActive(other.group),
      );
      if (!otherActive) {
        row.widget.value = true;
        this.setDirtyCanvas(true, false);
        return;
      }
    }

    if (value && restriction !== "none") {
      for (const other of this.dnRows) {
        if (other === row) continue;
        this.dnApplyToGroup(other.group, false);
        other.widget.value = false;
      }
    }

    this.dnApplyToGroup(row.group, value);
    row.widget.value = value;
    this.graph?.setDirtyCanvas(true, false);
  },

  /** Menu actions. Honours the restriction rather than fighting it. */
  dnSetAll(enable) {
    if (!this.dnRows?.length) return;

    const restriction = this.dnSetting("restriction", "none");
    const onlyFirst =
      (enable && restriction !== "none") || (!enable && restriction === "always one");

    this.dnRows.forEach((row, index) => {
      this.dnApplyToGroup(row.group, onlyFirst ? index === 0 : enable);
    });
    this.dnRefresh();
  },

  /**
   * Invert every row. Under a "one" restriction only the first row that would
   * turn on is allowed to, and "always one" force-enables the last row if the
   * inversion would otherwise leave nothing active.
   */
  dnToggleAll() {
    if (!this.dnRows?.length) return;

    const restriction = this.dnSetting("restriction", "none");
    const onlyOne = restriction.includes(" one");
    let foundOne = false;

    for (const row of this.dnRows) {
      const wanted = onlyOne && foundOne ? false : !isGroupActive(row.group);
      foundOne = foundOne || wanted;
      this.dnApplyToGroup(row.group, wanted);
    }
    if (!foundOne && restriction === "always one") {
      this.dnApplyToGroup(this.dnRows[this.dnRows.length - 1].group, true);
    }
    this.dnRefresh();
  },

  /** Centre the canvas on a group, zooming out only as far as needed to fit it. */
  dnNavigateTo(group) {
    const canvas = app.canvas;
    if (!canvas || !group) return;

    canvas.centerOnNode(group);

    const size = group._size ?? group.size;
    if (size && canvas.canvas) {
      const current = canvas.ds?.scale ?? 1;
      const fitX = canvas.canvas.width / size[0] - 0.02;
      const fitY = canvas.canvas.height / size[1] - 0.02;
      canvas.setZoom?.(Math.min(current, fitX, fitY), [
        canvas.canvas.width / 2,
        canvas.canvas.height / 2,
      ]);
    }
    canvas.setDirty(true, true);
  },

  /** A settings widget changed. */
  dnSettingChanged(name, value) {
    if (name === "action") {
      // Re-stamp the groups that are currently off so they carry the new
      // off-mode (a muted group must become bypassed, and vice versa).
      for (const row of this.dnRows ?? []) {
        if (!isGroupActive(row.group)) this.dnApplyToGroup(row.group, false, value);
      }
    }
    scanner.schedule(16);
  },
};

/* -------------------------------------------------------------------------- */
/* Registration                                                                */
/* -------------------------------------------------------------------------- */

app.registerExtension({
  name: EXTENSION_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_TYPE) return;

    Object.assign(nodeType.prototype, groupToggleMethods);

    chain(nodeType.prototype, "onNodeCreated", function () {
      this.dnRows = [];
      // Settings must persist; the generated rows are excluded in onSerialize.
      this.serialize_widgets = true;

      for (const widget of this.dnSettingWidgets()) {
        const node = this;
        const previous = widget.callback;
        widget.callback = function (value, ...rest) {
          const result = previous?.call(this, value, ...rest);
          node.dnSettingChanged(widget.name, value);
          return result;
        };
      }
    });

    chain(nodeType.prototype, "onAdded", function () {
      scanner.register(this);
    });

    chain(nodeType.prototype, "onRemoved", function () {
      scanner.unregister(this);
    });

    chain(nodeType.prototype, "onSerialize", function (info) {
      // Drop the generated rows so a saved workflow only carries the settings.
      if (Array.isArray(info?.widgets_values)) {
        info.widgets_values = info.widgets_values.slice(0, this.dnSettingWidgets().length);
      }
    });

    chain(nodeType.prototype, "getExtraMenuOptions", function (canvas, options) {
      const node = this;
      const offLabel = node.dnSetting("action", "mute") === "bypass" ? "Bypass all" : "Mute all";
      options.push(
        null,
        { content: "Enable all", callback: () => node.dnSetAll(true) },
        { content: offLabel, callback: () => node.dnSetAll(false) },
        { content: "Toggle all", callback: () => node.dnToggleAll() },
        {
          content: "Navigate to group",
          has_submenu: true,
          callback: (value, menuOptions, event, parentMenu) => {
            const entries = (node.dnRows ?? []).map((row) => ({
              content: row.group.title ?? "",
              callback: () => node.dnNavigateTo(row.group),
            }));
            if (!entries.length) return;
            new LiteGraph.ContextMenu(entries, { event, parentMenu, title: "Groups" });
          },
        },
      );
      return options;
    });
  },
});
