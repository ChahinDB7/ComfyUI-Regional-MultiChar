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
    this.scrollHeight = 100;
  }

  set innerHTML(value) {
    this._innerHTML = value;
    this.children = [];
  }

  get innerHTML() {
    return this._innerHTML;
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
const context = {
  app: { registerExtension(extension) { extensions.push(extension); } },
  api: {},
  document: { createElement(tag) { return new Element(tag); } },
  window: { requestAnimationFrame() {} },
  setTimeout() {},
  console,
};
const source = fs.readFileSync(path.join(__dirname, "..", "web", "regional_multichar.js"), "utf8")
  .replace(/^import .*;\s*$/gm, "");
vm.runInNewContext(source, context);

function find(element, predicate) {
  if (predicate(element)) return element;
  for (const child of element.children || []) {
    const match = find(child, predicate);
    if (match) return match;
  }
  return null;
}

function collect(element, predicate) {
  return [...(predicate(element) ? [element] : []),
    ...(element.children || []).flatMap((child) => collect(child, predicate))];
}

async function main() {
  const extension = extensions.find((item) => item.name === "Regional.MultiChar");
  const oldLayout = {
    characters: [{ name: "A", cells: [0], positive: "same face", negative: "extra arm" }],
    links: [{ between: [1], positive: "waving", negative: "apart" }],
  };
  class LayoutNode {
    constructor() {
      this.widgets = [
        { name: "aspect", value: "rectangle 1216x832", options: { values: ["rectangle 1216x832"] } },
        { name: "grid_cols", value: 1 }, { name: "grid_rows", value: 1 },
        { name: "batch_size", value: 1 },
        { name: "layout_json", value: JSON.stringify(oldLayout) },
      ];
      this.size = [440, 600];
    }

    addDOMWidget() { return {}; }
    setDirtyCanvas() {}
  }
  await extension.beforeRegisterNodeDef(LayoutNode, { name: "RegionalCharacterLayout" });
  const node = new LayoutNode();
  const widget = (name) => node.widgets.find((item) => item.name === name);
  node.onNodeCreated();

  assert.equal(node.k2.structured_characters, true);
  assert.equal(node.k2.characters[0].generic_positive, "same face");
  assert.equal(node.k2.characters[0].generic_negative, "extra arm");
  assert.equal(node.k2.characters[0].looks_positive, "");
  assert.equal(Object.hasOwn(node.k2.characters[0], "positive"), false);
  assert.equal(node.k2.links[0].positive, "waving");
  assert.equal(JSON.parse(widget("layout_json").value).structured_characters, true);
  const textareas = collect(node.k2_root, (item) => item.tag === "textarea");
  for (const label of ["generic positive", "generic negative", "looks positive",
    "looks negative", "pose / action positive", "pose / action negative"]) {
    assert.ok(textareas.some((item) => item.placeholder === label), label);
  }
  assert.ok(textareas.some((item) => String(item.placeholder || "").startsWith("interaction positive")));

  let checkbox = find(node.k2_root, (item) => item.tag === "input" && item.type === "checkbox");
  checkbox.checked = false;
  checkbox.handlers.change();
  assert.equal(node.k2.structured_characters, false);
  assert.equal(node.k2.characters[0].positive, "same face");
  assert.equal(node.k2.characters[0].negative, "extra arm");
  assert.equal(Object.hasOwn(node.k2.characters[0], "generic_positive"), false);
  assert.equal(JSON.parse(widget("layout_json").value).structured_characters, false);
  assert.ok(collect(node.k2_root, (item) => item.tag === "textarea")
    .some((item) => String(item.placeholder || "").startsWith("positive — looks")));

  checkbox = find(node.k2_root, (item) => item.tag === "input" && item.type === "checkbox");
  checkbox.checked = true;
  checkbox.handlers.change();
  assert.equal(node.k2.characters[0].generic_positive, "same face");
  assert.equal(node.k2.characters[0].pose_action_positive, "");

  const pasted = {
    structured_characters: true,
    characters: [{ name: "A", cells: [0], generic_positive: "person",
      looks_positive: "black hair", pose_action_positive: "running",
      generic_negative: "blur", looks_negative: "red hair", pose_action_negative: "sitting" }],
    links: [{ between: [1], positive: "holding a bag", negative: "apart" }],
  };
  node._jsonTA.value = JSON.stringify(pasted);
  node._jsonTA.handlers.input();
  const applyButton = node.k2_root.children[2].children[0];
  applyButton.handlers.click();
  assert.equal(node.k2.characters[0].looks_positive, "black hair");
  assert.equal(node.k2.characters[0].pose_action_negative, "sitting");
  assert.equal(node.k2.links[0].positive, "holding a bag");
  const saved = JSON.parse(widget("layout_json").value);
  assert.equal(saved.structured_characters, true);
  assert.equal(Object.hasOwn(saved.characters[0], "positive"), false);

  checkbox = find(node.k2_root, (item) => item.tag === "input" && item.type === "checkbox");
  checkbox.checked = false;
  checkbox.handlers.change();
  assert.equal(node.k2.characters[0].positive, "person, black hair, running");
  assert.equal(node.k2.characters[0].negative, "blur, red hair, sitting");
  assert.equal(Object.hasOwn(node.k2.characters[0], "looks_positive"), false);
  assert.equal(node.k2.links[0].positive, "holding a bag");
  checkbox = find(node.k2_root, (item) => item.tag === "input" && item.type === "checkbox");
  checkbox.checked = true;
  checkbox.handlers.change();
  assert.equal(node.k2.characters[0].generic_positive, "person, black hair, running");
  assert.equal(node.k2.characters[0].looks_positive, "");

  widget("layout_json").value = JSON.stringify({
    structured_characters: false,
    characters: [{ name: "B", cells: [], positive: "old pose", negative: "old guard" }],
    links: [{ between: [1], positive: "old interaction", negative: "old interaction guard" }],
  });
  node.onConfigure();
  assert.equal(node.k2.structured_characters, false);
  assert.equal(node.k2.characters[0].positive, "old pose");
  assert.equal(node.k2.links[0].negative, "old interaction guard");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
