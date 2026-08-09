/**
 * Tests for DN_GroupToggle — web/dneg_group_toggle.js driving the node declared
 * in _nodes/_group_toggle.py.
 *
 *   node --test "tests/*.test.mjs"
 *
 * (Use the glob, not `node --test tests` — a bare directory name gets resolved
 * as a module on Windows and fails before any test runs.)
 *
 * No dependencies and no build step: LiteGraph is stubbed in tests/stubs/, and
 * the module's `import ... from "/scripts/app.js"` — an absolute URL Node cannot
 * resolve — is rewritten to the stub before the source is imported.
 *
 * The bulk of these cover the restriction rules, because that is precisely what
 * regressed in rgthree's Fast Groups Muter: its "always one" guard tested a
 * widget value that had quietly become an object, so the guard never fired and
 * the last enabled group could be switched off. Our state is derived from node
 * modes instead, and these tests pin that down.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

import { LGraphNode, installGlobals } from "./stubs/litegraph.mjs";
import { makeContext } from "./stubs/canvas2d.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const PACKAGE = join(HERE, "..", "src", "custom_nodes", "dneg_utils_nodes");
const SOURCE = join(PACKAGE, "web", "dneg_group_toggle.js");
const SCHEMA = join(PACKAGE, "_nodes", "_group_toggle.py");
const APP_STUB = join(HERE, "stubs", "comfy_app.mjs");

const NODE_TYPE = "DN_GroupToggle";

const MODE_ALWAYS = 0;
const MODE_NEVER = 2;
const MODE_BYPASS = 4;

/** Settings widgets, in schema order, with their declared defaults. */
const SETTINGS = [
  ["action", "combo", "mute"],
  ["restriction", "combo", "none"],
  ["match_colors", "text", ""],
  ["match_title", "text", ""],
  ["sort", "combo", "position"],
  ["custom_sort_alphabet", "text", ""],
  ["show_nav", "toggle", true],
  ["include_subgraphs", "toggle", true],
];

/** Import the extension with globals installed and its app import redirected. */
async function loadGroupToggle() {
  installGlobals();
  const source = readFileSync(SOURCE, "utf8").replace(
    '"/scripts/app.js"',
    JSON.stringify(pathToFileURL(APP_STUB).href),
  );
  await import(`data:text/javascript,${encodeURIComponent(source)}`);
  const { app, extensions } = await import(pathToFileURL(APP_STUB).href);
  return { app, extension: extensions.at(-1) };
}

const { app, extension } = await loadGroupToggle();
assert.ok(extension?.beforeRegisterNodeDef, "extension should expose beforeRegisterNodeDef");

/**
 * Stand in for the class ComfyUI builds from the schema: settings widgets are
 * created first, then `onNodeCreated` fires — and `beforeRegisterNodeDef` has
 * already patched the prototype by then.
 */
class GroupToggleNode extends LGraphNode {
  constructor() {
    super("DN Group Toggle");
    this.type = NODE_TYPE;
    for (const [name, widgetType, value] of SETTINGS) {
      this.addWidget(widgetType, name, value, () => {});
    }
    this.onNodeCreated?.();
  }
}
extension.beforeRegisterNodeDef(GroupToggleNode, { name: NODE_TYPE });

/**
 * Build a graph from group specs and point the stub app at it.
 * @param {Array<{title: string, color?: string, active?: boolean, count?: number, pos?: number[]}>} specs
 */
function buildGraph(specs) {
  const graph = { id: "root", groups: [], setDirtyCanvas() {} };

  graph.groups = specs.map((spec, index) => {
    const members = Array.from({ length: spec.count ?? 2 }, () => {
      const member = new LGraphNode("member");
      member.type = "SomeNode";
      member.mode = spec.active ? MODE_ALWAYS : MODE_NEVER;
      return member;
    });

    return {
      id: index + 1,
      title: spec.title,
      color: spec.color,
      pos: spec.pos ?? [0, index * 100],
      graph,
      _children: new Set(members),
      recomputeInsideNodes() {},
    };
  });

  app.graph = graph;
  app.canvas = { getCurrentGraph: () => graph, isDragging: false, setDirty() {} };
  return graph;
}

function makeToggleNode(settings = {}) {
  const node = new GroupToggleNode();
  for (const [name, value] of Object.entries(settings)) {
    const widget = node.widgets.find((w) => w.name === name);
    assert.ok(widget, `unknown setting: ${name}`);
    widget.value = value;
  }
  node.graph = app.graph;
  node.dnRefresh();
  return node;
}

const activeTitles = (node) =>
  node.dnRows.filter((row) => row.widget.value).map((row) => row.group.title);

const titles = (node) => node.dnRows.map((row) => row.group.title);

const modesOf = (group) => [...group._children].map((member) => member.mode);

/** Click a row the way the canvas does: the widget flips itself, then calls back. */
function click(node, title, value) {
  const row = node.dnRows.find((candidate) => candidate.group.title === title);
  assert.ok(row, `no row titled ${title}`);
  row.widget.value = value;
  row.widget.callback(value);
}

/* -------------------------------------------------------------------------- */

test("settings widgets survive and stay above the generated rows", () => {
  buildGraph([{ title: "A", color: "#8A8", active: true }, { title: "B", color: "#8A8" }]);
  const node = makeToggleNode({ match_colors: "green" });

  assert.deepEqual(
    node.dnSettingWidgets().map((w) => w.name),
    SETTINGS.map(([name]) => name),
    "all seven settings present, in schema order",
  );
  assert.deepEqual(
    node.widgets.map((w) => w.label ?? w.name),
    [...SETTINGS.map(([name]) => name), "A", "B"],
    "rows are appended after the settings",
  );
});

test("the schema's input names match the names the frontend reads", () => {
  const schema = readFileSync(SCHEMA, "utf8");
  const declared = [...schema.matchAll(/io\.\w+\.Input\(\s*"([a-z_]+)"/g)].map((m) => m[1]);
  const read = new Set(
    [...readFileSync(SOURCE, "utf8").matchAll(/dnSetting\(\s*"([a-z_]+)"/g)].map((m) => m[1]),
  );

  assert.deepEqual(declared, SETTINGS.map(([name]) => name), "schema order matches this test");
  for (const name of read) {
    assert.ok(declared.includes(name), `frontend reads "${name}" but the schema does not declare it`);
  }
});

test("generated rows are excluded from the saved widget values", () => {
  buildGraph([{ title: "A", color: "#8A8", active: true }, { title: "B", color: "#8A8" }]);
  const node = makeToggleNode({ match_colors: "green", restriction: "max one" });

  const info = { widgets_values: node.widgets.map((w) => w.value) };
  assert.equal(info.widgets_values.length, SETTINGS.length + 2, "rows are present before serialize");

  node.onSerialize(info);
  assert.deepEqual(info.widgets_values, [
    "mute", "max one", "green", "", "position", "", true, true,
  ]);
});

test('"always one" refuses to switch off the last active group', () => {
  buildGraph([
    { title: "Text to Image", color: "#3f789e", active: true },
    { title: "Image to Image", color: "#3f789e", active: false },
    { title: "Refine Image", color: "#3f789e", active: false },
  ]);
  const node = makeToggleNode({ match_colors: "pale_blue", restriction: "always one" });
  assert.deepEqual(activeTitles(node), ["Text to Image"]);

  click(node, "Text to Image", false);
  node.dnRefresh();

  assert.deepEqual(activeTitles(node), ["Text to Image"], "toggle should snap back on");
  assert.deepEqual(modesOf(node.dnRows[0].group), [MODE_ALWAYS, MODE_ALWAYS]);
});

test('"always one" moves the active slot when another group is switched on', () => {
  buildGraph([
    { title: "Text to Image", color: "#3f789e", active: true },
    { title: "Image to Image", color: "#3f789e", active: false },
    { title: "Refine Image", color: "#3f789e", active: false },
  ]);
  const node = makeToggleNode({ match_colors: "pale_blue", restriction: "always one" });

  click(node, "Image to Image", true);
  node.dnRefresh();

  assert.deepEqual(activeTitles(node), ["Image to Image"]);
});

test('"always one" holds across an arbitrary click sequence', () => {
  buildGraph([
    { title: "A", color: "#8A8", active: true },
    { title: "B", color: "#8A8", active: false },
    { title: "C", color: "#8A8", active: false },
  ]);
  const node = makeToggleNode({ match_colors: "green", restriction: "always one" });

  for (const title of ["A", "B", "C", "B", "A", "C", "C", "A", "A"]) {
    const row = node.dnRows.find((candidate) => candidate.group.title === title);
    click(node, title, !row.widget.value);
    node.dnRefresh();
    assert.equal(activeTitles(node).length, 1, `exactly one active after clicking ${title}`);
  }
});

test('"always one" survives a group muted outside the node', () => {
  const graph = buildGraph([
    { title: "A", color: "#8A8", active: true },
    { title: "B", color: "#8A8", active: false },
  ]);
  const node = makeToggleNode({ match_colors: "green", restriction: "always one" });

  // Someone mutes A by hand (Ctrl+M on the nodes), leaving nothing active.
  for (const member of graph.groups[0]._children) member.mode = MODE_NEVER;
  node.dnRefresh();
  assert.deepEqual(activeTitles(node), []);

  // B can still be switched on — the guard reads modes, not stale widget values.
  click(node, "B", true);
  node.dnRefresh();
  assert.deepEqual(activeTitles(node), ["B"]);
});

test('"max one" allows zero active but never two', () => {
  buildGraph([
    { title: "A", color: "#8A8", active: true },
    { title: "B", color: "#8A8", active: false },
  ]);
  const node = makeToggleNode({ match_colors: "green", restriction: "max one" });

  click(node, "B", true);
  node.dnRefresh();
  assert.deepEqual(activeTitles(node), ["B"]);

  click(node, "B", false);
  node.dnRefresh();
  assert.deepEqual(activeTitles(node), []);
});

test('"none" leaves rows independent', () => {
  buildGraph([
    { title: "A", color: "#8A8", active: true },
    { title: "B", color: "#8A8", active: false },
  ]);
  const node = makeToggleNode({ match_colors: "green", restriction: "none" });

  click(node, "B", true);
  node.dnRefresh();
  assert.deepEqual(activeTitles(node).sort(), ["A", "B"]);
});

test("Enable all / Disable all honour the restriction", () => {
  buildGraph([
    { title: "A", color: "#8A8", active: true },
    { title: "B", color: "#8A8", active: false },
    { title: "C", color: "#8A8", active: false },
  ]);
  const alwaysOne = makeToggleNode({ match_colors: "green", restriction: "always one" });
  alwaysOne.dnSetAll(false);
  assert.equal(activeTitles(alwaysOne).length, 1, "always one keeps a survivor");
  alwaysOne.dnSetAll(true);
  assert.equal(activeTitles(alwaysOne).length, 1, "always one never enables everything");

  const unrestricted = makeToggleNode({ match_colors: "green", restriction: "none" });
  unrestricted.dnSetAll(true);
  assert.equal(activeTitles(unrestricted).length, 3);
  unrestricted.dnSetAll(false);
  assert.equal(activeTitles(unrestricted).length, 0);
});

test("match_colors resolves palette names, 3-digit hex and mixed case", () => {
  buildGraph([
    { title: "Lora Snow", color: "#8A8", active: true },
    { title: "Model SDXL", color: "#3f789e", active: false },
    { title: "Core Prompt", color: "#444", active: false },
  ]);

  assert.deepEqual(titles(makeToggleNode({ match_colors: "green" })), ["Lora Snow"]);
  assert.deepEqual(titles(makeToggleNode({ match_colors: "pale_blue" })), ["Model SDXL"]);
  assert.deepEqual(titles(makeToggleNode({ match_colors: "#8a8" })), ["Lora Snow"]);
  assert.equal(makeToggleNode({ match_colors: "green, #3F789E" }).dnRows.length, 2);
  assert.equal(makeToggleNode({ match_colors: "" }).dnRows.length, 3, "empty means all groups");
});

test("match_title filters by case-insensitive regex", () => {
  buildGraph([
    { title: "SDXL LoRA - Snow", color: "#8A8" },
    { title: "SDXL LoRA - Underwater", color: "#8A8" },
    { title: "No LoRA / IPAdapter", color: "#8A8" },
  ]);
  assert.equal(
    makeToggleNode({ match_colors: "green", match_title: "^sdxl" }).dnRows.length,
    2,
  );
  assert.equal(makeToggleNode({ match_title: "ipadapter" }).dnRows.length, 1);
});

test("an invalid match_title regex does not blank the node", () => {
  buildGraph([{ title: "A", color: "#8A8" }, { title: "B", color: "#8A8" }]);

  // The node warns about the bad pattern; that is expected here, so keep it out
  // of the test output.
  const warn = console.warn;
  console.warn = () => {};
  try {
    // "(" is what you have mid-typing; the node should keep listing groups.
    assert.equal(makeToggleNode({ match_title: "(" }).dnRows.length, 2);
  } finally {
    console.warn = warn;
  }
});

test('action "bypass" stamps mode 4 instead of mode 2', () => {
  buildGraph([
    { title: "A", color: "#8A8", active: true },
    { title: "B", color: "#8A8", active: false },
  ]);
  const node = makeToggleNode({
    match_colors: "green",
    restriction: "max one",
    action: "bypass",
  });

  click(node, "B", true);
  node.dnRefresh();
  assert.deepEqual(modesOf(node.dnRows[0].group), [MODE_BYPASS, MODE_BYPASS]);
  assert.deepEqual(modesOf(node.dnRows[1].group), [MODE_ALWAYS, MODE_ALWAYS]);
});

test("switching the action widget re-stamps the groups that are already off", () => {
  buildGraph([
    { title: "A", color: "#8A8", active: true },
    { title: "B", color: "#8A8", active: false },
  ]);
  const node = makeToggleNode({ match_colors: "green", restriction: "none" });
  assert.deepEqual(modesOf(node.dnRows[1].group), [MODE_NEVER, MODE_NEVER]);

  // Drive it through the widget, the way a click on the combo does.
  const widget = node.widgets.find((w) => w.name === "action");
  widget.value = "bypass";
  widget.callback("bypass");

  assert.deepEqual(modesOf(node.dnRows[1].group), [MODE_BYPASS, MODE_BYPASS]);
  assert.deepEqual(modesOf(node.dnRows[0].group), [MODE_ALWAYS, MODE_ALWAYS], "active untouched");
});

test("sort orders rows by alphabet, position and custom prefixes", () => {
  buildGraph([
    { title: "Zulu", color: "#8A8", pos: [0, 0] },
    { title: "Alpha", color: "#8A8", pos: [0, 200] },
    { title: "Mike", color: "#8A8", pos: [500, 5] },
  ]);

  assert.deepEqual(titles(makeToggleNode({ sort: "alphanumeric" })), ["Alpha", "Mike", "Zulu"]);
  // Zulu and Mike share a 30px row band, so Zulu (x=0) precedes Mike (x=500).
  assert.deepEqual(titles(makeToggleNode({ sort: "position" })), ["Zulu", "Mike", "Alpha"]);
  assert.deepEqual(
    titles(makeToggleNode({ sort: "custom alphabet", custom_sort_alphabet: "mike,zulu" })),
    ["Mike", "Zulu", "Alpha"],
    "unmatched groups sort last, alphanumerically",
  );
});

test("rows track renames and drop groups that disappear", () => {
  const graph = buildGraph([
    { title: "A", color: "#8A8", active: true },
    { title: "B", color: "#8A8", active: false },
  ]);
  const node = makeToggleNode({ match_colors: "green", restriction: "none" });
  const widgetForA = node.dnRows[0].widget;

  graph.groups[0].title = "A renamed";
  node.dnRefresh();
  assert.equal(node.dnRows[0].widget, widgetForA, "widget is reused, not rebuilt");
  assert.equal(node.dnRows[0].widget.label, "A renamed");

  graph.groups.pop();
  node.dnRefresh();
  assert.equal(node.dnRows.length, 1);
  assert.equal(node.widgets.length, SETTINGS.length + 1, "settings kept, one row left");
});

test("a controller sitting inside a group it manages never mutes itself", () => {
  const graph = buildGraph([{ title: "A", color: "#8A8", active: true }]);
  const node = makeToggleNode({ match_colors: "green", restriction: "none" });

  const controller = new LGraphNode("controller");
  controller.type = NODE_TYPE;
  controller.mode = MODE_ALWAYS;
  graph.groups[0]._children.add(controller);

  click(node, "A", false);
  assert.equal(controller.mode, MODE_ALWAYS);
});

test("getExtraMenuOptions adds the actions without dropping existing entries", () => {
  buildGraph([{ title: "A", color: "#8A8", active: true }]);

  const muter = makeToggleNode({ match_colors: "green" });
  const options = [{ content: "Pre-existing" }];
  muter.getExtraMenuOptions(null, options);
  const labels = options.filter(Boolean).map((entry) => entry.content);

  assert.ok(labels.includes("Pre-existing"), "existing menu entries survive");
  assert.deepEqual(labels.slice(1), [
    "Enable all", "Mute all", "Toggle all", "Navigate to group",
  ]);

  // The off-action is named for what it actually does, like rgthree's.
  const bypasser = makeToggleNode({ match_colors: "green", action: "bypass" });
  const bypassOptions = [];
  bypasser.getExtraMenuOptions(null, bypassOptions);
  assert.ok(
    bypassOptions.filter(Boolean).map((e) => e.content).includes("Bypass all"),
    "bypass action renames the entry",
  );
});

test('"Toggle all" inverts every row, and respects the restriction', () => {
  buildGraph([
    { title: "A", color: "#8A8", active: true },
    { title: "B", color: "#8A8", active: false },
    { title: "C", color: "#8A8", active: false },
  ]);

  const unrestricted = makeToggleNode({ match_colors: "green", restriction: "none" });
  unrestricted.dnToggleAll();
  assert.deepEqual(activeTitles(unrestricted), ["B", "C"], "every row flipped");

  // Under "max one" only the first row that wants to turn on is allowed to.
  buildGraph([
    { title: "A", color: "#8A8", active: true },
    { title: "B", color: "#8A8", active: false },
    { title: "C", color: "#8A8", active: false },
  ]);
  const maxOne = makeToggleNode({ match_colors: "green", restriction: "max one" });
  maxOne.dnToggleAll();
  assert.deepEqual(activeTitles(maxOne), ["B"]);

  // Inverting a single active row would leave nothing on, so "always one" saves it.
  buildGraph([{ title: "Only", color: "#8A8", active: true }]);
  const alwaysOne = makeToggleNode({ match_colors: "green", restriction: "always one" });
  alwaysOne.dnToggleAll();
  assert.deepEqual(activeTitles(alwaysOne), ["Only"]);
});

test("custom_sort_alphabet splits on characters when it has no comma", () => {
  buildGraph([
    { title: "Alpha", color: "#8A8" },
    { title: "Mike", color: "#8A8" },
    { title: "Zulu", color: "#8A8" },
  ]);

  // rgthree's terse form: "zma" reverses the usual order.
  assert.deepEqual(
    titles(makeToggleNode({ sort: "custom alphabet", custom_sort_alphabet: "zma" })),
    ["Zulu", "Mike", "Alpha"],
  );
  // A comma means whole prefixes, so "z,m,a" behaves the same here...
  assert.deepEqual(
    titles(makeToggleNode({ sort: "custom alphabet", custom_sort_alphabet: "z,m,a" })),
    ["Zulu", "Mike", "Alpha"],
  );
  // ...but multi-character prefixes only work in the comma form.
  assert.deepEqual(
    titles(makeToggleNode({ sort: "custom alphabet", custom_sort_alphabet: "mike,zulu" })),
    ["Mike", "Zulu", "Alpha"],
  );
});

test("a row draws its title, yes/no state, toggle dot and nav arrow", () => {
  buildGraph([{ title: "SDXL LoRA - Snow Mountain", color: "#8A8", active: true }]);
  const node = makeToggleNode({ match_colors: "green" });
  const widget = node.dnRows[0].widget;

  const ctx = makeContext();
  widget.draw(ctx, node, 300, 40, 20);

  assert.equal(ctx.texts.find((t) => t.align === "right").text, "yes", "active row reads 'yes'");
  const title = ctx.texts.find((t) => t.align === "left");
  assert.ok(title.text.startsWith("SDXL LoRA"), "title is drawn");
  assert.equal(ctx.arcs.length, 1, "one toggle dot");
  assert.equal(ctx.arcs[0].fill, "#89A", "dot is lit while active");
  assert.ok(
    ctx.paths.some((p) => p.kind === "fill" && p.d.includes("l -7 6")),
    "nav arrow is drawn",
  );

  // Switching it off changes the dot colour and the label.
  widget.value = false;
  const off = makeContext();
  widget.draw(off, node, 300, 40, 20);
  assert.equal(off.texts.find((t) => t.align === "right").text, "no");
  assert.equal(off.arcs[0].fill, "#333");
});

test("show_nav removes the arrow, and a long title is ellipsised", () => {
  buildGraph([{ title: "A very long group title that will not fit", color: "#8A8" }]);

  const withNav = makeToggleNode({ match_colors: "green" });
  const navCtx = makeContext();
  withNav.dnRows[0].widget.draw(navCtx, withNav, 200, 0, 20);
  assert.ok(navCtx.paths.some((p) => p.d?.includes("l -7 6")), "arrow present by default");
  assert.ok(navCtx.texts.find((t) => t.align === "left").text.endsWith("…"), "title truncated");

  const noNav = makeToggleNode({ match_colors: "green", show_nav: false });
  const plainCtx = makeContext();
  noNav.dnRows[0].widget.draw(plainCtx, noNav, 200, 0, 20);
  assert.ok(!plainCtx.paths.some((p) => p.d?.includes("l -7 6")), "arrow suppressed");
});

test("clicking the nav band navigates; clicking elsewhere toggles", () => {
  const graph = buildGraph([
    { title: "A", color: "#8A8", active: true },
    { title: "B", color: "#8A8", active: false },
  ]);
  const node = makeToggleNode({ match_colors: "green", restriction: "none" });
  node.size = [300, 200];

  let centred = null;
  app.canvas.centerOnNode = (target) => {
    centred = target;
  };

  // Right-hand band -> navigate, state untouched.
  node.dnRows[0].widget.mouse({ type: "pointerdown" }, [300 - 10, 5], node);
  assert.equal(centred, graph.groups[0], "canvas centred on the group");
  assert.deepEqual(activeTitles(node), ["A"], "navigation does not toggle");

  // Anywhere left of the band -> toggle.
  node.dnRows[1].widget.mouse({ type: "pointerdown" }, [40, 5], node);
  node.dnRefresh();
  assert.deepEqual(activeTitles(node).sort(), ["A", "B"]);
});
