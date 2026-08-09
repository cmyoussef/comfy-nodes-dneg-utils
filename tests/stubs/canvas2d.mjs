/**
 * Recording stand-in for CanvasRenderingContext2D, enough to exercise a widget's
 * draw() and assert on the geometry it produced.
 */
export function makeContext({ charWidth = 6 } = {}) {
  const calls = [];
  const record = (op) => (...args) => calls.push({ op, args });

  return {
    calls,
    texts: [],
    paths: [],
    arcs: [],

    // Style slots the widget writes to.
    fillStyle: "",
    strokeStyle: "",
    textAlign: "",
    lineJoin: "",
    lineCap: "",

    measureText(text) {
      return { width: String(text).length * charWidth };
    },
    fillText(text, x, y) {
      this.texts.push({ text, x, y, align: this.textAlign, fill: this.fillStyle });
    },
    arc(x, y, radius) {
      this.arcs.push({ x, y, radius, fill: this.fillStyle });
    },
    fill(path) {
      if (path) this.paths.push({ kind: "fill", d: path.d });
      calls.push({ op: "fill", args: [] });
    },
    stroke(path) {
      if (path) this.paths.push({ kind: "stroke", d: path.d });
      calls.push({ op: "stroke", args: [] });
    },
    roundRect(x, y, w, h, radii) {
      calls.push({ op: "roundRect", args: [x, y, w, h, radii] });
    },
    beginPath: record("beginPath"),
    save: record("save"),
    restore: record("restore"),
  };
}
