const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Element {
  constructor(tag) {
    this.tag = tag;
    this.style = {};
    this.children = [];
    this.handlers = {};
    this.textContent = "";
    this.innerHTML = "";
    this.scrollHeight = 100;
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  addEventListener(name, handler) {
    this.handlers[name] = handler;
  }
}

const extensions = [];
const copied = [];
const requests = [];
const context = {
  app: { registerExtension(extension) { extensions.push(extension); } },
  api: {
    async fetchApi(_url, options) {
      requests.push(JSON.parse(options.body));
      return { async json() { return { report: "# Report" }; } };
    },
  },
  document: { createElement(tag) { return new Element(tag); } },
  navigator: { clipboard: { async writeText(value) { copied.push(value); } } },
  window: {},
  setTimeout() {},
  clearTimeout() {},
  console,
};
const source = fs.readFileSync(path.join(__dirname, "..", "web", "regional_multichar.js"), "utf8")
  .replace(/^import .*;\s*$/gm, "");
vm.runInNewContext(source, context);

async function main() {
  const extension = extensions.find((item) => item.name === "Regional.MultiCharComposeLive");
  const defaults = {
    global_positive: "A studio scene", global_negative: "blur",
    subject_count_lock: true, use_names: "handle", bind_interactions: true,
    cast_roster: true, order_and_group: true, auto_scale_hints: true,
    spatial_detail: "fine", output_format: "prose", auto_framing: false,
    negative_mode: "global_dedup", prompt_profile: "flux2", negative_term_cap: 0,
    negative_cap_priority: "global_first", spatial_cues: "auto", scale_cues: "off",
    layout_json_override: "",
  };
  class ComposeNode {
    constructor() {
      this.widgets = Object.entries(defaults).map(([name, value]) => ({ name, value }));
      this.inputs = [];
      this.size = [500, 660];
    }

    addDOMWidget(_name, _type, element) {
      this.element = element;
      return {};
    }

    setDirtyCanvas() {}
  }
  await extension.beforeRegisterNodeDef(ComposeNode, { name: "MultiCharPromptCompose" });
  const node = new ComposeNode();
  const widget = (name) => node.widgets.find((item) => item.name === name);
  node.onNodeCreated();
  const editor = node._composeSettingsEditor;
  const initial = JSON.parse(editor.ta.value);
  assert.equal(initial.global_positive, "A studio scene");
  assert.equal(initial.global_negative, "blur");
  assert.equal(initial.prompt_profile, "flux2");
  assert.equal(Object.keys(initial).length, Object.keys(defaults).length - 1);
  for (const name of ["grid_cols", "grid_rows", "characters", "links", "aspect", "batch_size", "layout_json_override"]) {
    assert.equal(Object.hasOwn(initial, name), false, name);
  }

  widget("subject_count_lock").value = false;
  widget("subject_count_lock").callback();
  assert.equal(JSON.parse(editor.ta.value).subject_count_lock, false);
  node._taRefs.global_positive.ta.value = "An outdoor scene";
  node._taRefs.global_positive.ta.handlers.input();
  assert.equal(JSON.parse(editor.ta.value).global_positive, "An outdoor scene");

  editor.ta.value = JSON.stringify({ global_negative: "bad hands", negative_term_cap: 80 });
  editor.ta.handlers.input();
  editor.apply.handlers.click();
  assert.equal(widget("global_negative").value, "bad hands");
  assert.equal(widget("negative_term_cap").value, 80);
  assert.equal(widget("subject_count_lock").value, false);
  assert.equal(node._taRefs.global_negative.ta.value, "bad hands");
  assert.equal(JSON.parse(editor.ta.value).global_negative, "bad hands");
  assert.equal(requests.at(-1).opts.negative_term_cap, 80);

  editor.ta.value = JSON.stringify({ global_negative: "invalid change", negative_term_cap: "80" });
  editor.ta.handlers.input();
  editor.apply.handlers.click();
  assert.equal(widget("global_negative").value, "bad hands");
  assert.equal(widget("negative_term_cap").value, 80);

  editor.ta.value = JSON.stringify({ global_negative: "wrong change", characters: [] });
  editor.ta.handlers.input();
  editor.apply.handlers.click();
  assert.equal(widget("global_negative").value, "bad hands");

  widget("layout_json_override").value = '{"characters":[]}';
  node.onConfigure();
  assert.equal(editor.legacyRow.style.display, "flex");
  editor.ta.value = JSON.stringify({ widgets_values_named: {
    global_positive: "Copied scene", layout_json_override: "",
  } });
  editor.ta.handlers.input();
  editor.apply.handlers.click();
  assert.equal(widget("global_positive").value, "Copied scene");
  assert.equal(widget("layout_json_override").value, "");
  assert.equal(editor.legacyRow.style.display, "none");

  const settingsRow = node.element.children[4].children[2];
  await settingsRow.children[1].handlers.click();
  assert.equal(JSON.parse(copied.at(-1)).global_positive, "Copied scene");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
