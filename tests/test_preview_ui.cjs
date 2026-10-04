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
const context = {
  app: { registerExtension(extension) { extensions.push(extension); } },
  api: {},
  document: { createElement(tag) { return new Element(tag); } },
  navigator: { clipboard: { async writeText(value) { copied.push(value); } } },
  setTimeout() {},
  console,
};
const source = fs.readFileSync(path.join(__dirname, "..", "web", "regional_multichar.js"), "utf8")
  .replace(/^import .*;\s*$/gm, "");
vm.runInNewContext(source, context);

async function main() {
  const extension = extensions.find((item) => item.name === "Regional.MultiCharPreview");
  class PreviewNode {
    constructor() {
      this.size = [460, 650];
    }

    addDOMWidget(name, type, element) {
      this.element = element;
      return {};
    }

    setDirtyCanvas() {}
  }
  await extension.beforeRegisterNodeDef(PreviewNode, { name: "MultiCharPromptPreview" });
  const node = new PreviewNode();
  node.onNodeCreated();
  assert.equal(node.mc_prompt_blocks.positive.raw.textContent, "");
  assert.equal(node.mc_prompt_blocks.negative.raw.textContent, "");

  node.onExecuted({
    has_prompts: [true], positive_text: ["line 1\n**literal** <tag>"],
    negative_text: ["blur, extra arm"], text: ["report"],
  });
  assert.equal(node.mc_prompt_blocks.positive.raw.textContent, "line 1\n**literal** <tag>");
  assert.equal(node.mc_prompt_blocks.negative.raw.textContent, "blur, extra arm");
  assert.equal(node.mc_prompt_blocks.fallback.style.display, "none");

  const positiveHeader = node.mc_prompt_blocks.positive.section.children[0];
  await positiveHeader.children[2].handlers.click();
  assert.equal(copied[0], "line 1\n**literal** <tag>");
  positiveHeader.children[1].handlers.click();
  assert.equal(node.mc_prompt_blocks.positive.raw.style.display, "none");
  assert.match(node.mc_prompt_blocks.positive.markdown.innerHTML, /<b>literal<\/b>/);

  node.onExecuted({ has_prompts: [false], text: ["# Arbitrary markdown"] });
  assert.equal(node.mc_prompt_blocks.positive.section.style.display, "none");
  assert.match(node.mc_prompt_blocks.fallback.innerHTML, /<h1/);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
