/**
 * Stand-in for ComfyUI's `/scripts/app.js`, substituted into the module under
 * test by `loadGroupToggle()` in ../group_toggle.test.mjs.
 */
/** Extensions registered by the module under test, in registration order. */
export const extensions = [];

export const app = {
  canvas: null,
  graph: null,
  registerExtension(extension) {
    extensions.push(extension);
  },
};
