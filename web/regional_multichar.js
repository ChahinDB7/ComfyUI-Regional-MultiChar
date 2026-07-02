// Dynamic editor for the RegionalCharacterLayout node.
//
// Renders an add/remove list of character cards (each with a clickable grid to
// place the character + its own positive/negative text) and interaction cards
// (link two or more characters + shared positive/negative text). All of it is
// serialized into the node's hidden `layout_json` widget, which is what the
// Python side reads. If this script fails to load, that widget stays visible as
// a plain JSON textarea so the node still works -- nothing here is load-bearing
// for generation, only for editing comfort.

import { app } from "../../scripts/app.js";

const NODE = "RegionalCharacterLayout";
const PALETTE = ["#e74c3c", "#2ecc71", "#3498db", "#f1c40f", "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6"];

function el(tag, style, props) {
  const e = document.createElement(tag);
  if (style) Object.assign(e.style, style);
  if (props) Object.assign(e, props);
  return e;
}

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function widgetVal(node, name, fallback) {
  const w = getWidget(node, name);
  const v = w ? w.value : undefined;
  return v === undefined || v === null || v === "" ? fallback : v;
}

function grid(node) {
  const cols = Math.max(1, Math.min(8, parseInt(widgetVal(node, "grid_cols", 3)) || 3));
  const rows = Math.max(1, Math.min(8, parseInt(widgetVal(node, "grid_rows", 3)) || 3));
  return { cols, rows };
}

// image aspect from the "aspect" combo (e.g. "square 832x832") so the editor
// grid is drawn in the real canvas shape, not the grid's own shape
function imageAspect(node) {
  const s = String(widgetVal(node, "aspect", "1216x832"));
  const m = s.match(/(\d+)\s*[x×]\s*(\d+)/i);
  return m ? { w: parseInt(m[1]), h: parseInt(m[2]) } : { w: 1216, h: 832 };
}

function blankChar() {
  return { name: "", cells: [], positive: "", negative: "" };
}

function loadState(node) {
  let data = { characters: [], links: [] };
  const w = getWidget(node, "layout_json");
  if (w && w.value) {
    try {
      const p = JSON.parse(w.value);
      // normalize so a malformed entry (e.g. a null in characters) can't make
      // render() throw later and strand the user with a hidden raw widget
      data.characters = (Array.isArray(p.characters) ? p.characters : [])
        .filter((c) => c && typeof c === "object")
        .map((c) => ({
          name: c.name || "",
          cells: Array.isArray(c.cells) ? c.cells.filter(Number.isInteger) : [],
          positive: c.positive || "",
          negative: c.negative || "",
        }));
      data.links = (Array.isArray(p.links) ? p.links : [])
        .filter((l) => l && typeof l === "object")
        .map((l) => ({
          between: Array.isArray(l.between) ? l.between.filter(Number.isInteger) : [],
          positive: l.positive || "",
          negative: l.negative || "",
        }));
    } catch (e) {
      /* keep empty, raw widget still holds the bad text */
    }
  }
  if (!data.characters.length) data.characters = [blankChar()];
  node.k2 = data;
}

function save(node) {
  const w = getWidget(node, "layout_json");
  if (w) w.value = JSON.stringify(node.k2);
  node.setDirtyCanvas(true, true);
}

function dropInvalidCells(node) {
  const { cols, rows } = grid(node);
  const max = cols * rows;
  for (const ch of node.k2.characters) {
    ch.cells = (ch.cells || []).filter((c) => c >= 0 && c < max);
  }
}

function render(node) {
  const root = node.k2_root;
  if (!root) return;
  root.innerHTML = "";
  const { cols, rows } = grid(node);

  const heading = (text) =>
    el("div", { fontWeight: "600", margin: "8px 0 4px", color: "#cfd2d6", fontSize: "11px", letterSpacing: ".04em", textTransform: "uppercase" }, { textContent: text });

  const button = (text, bg) =>
    el("button", { background: bg || "#3a3f4b", color: "#fff", border: "none", borderRadius: "5px", padding: "5px 9px", cursor: "pointer", fontSize: "12px" }, { textContent: text });

  const textArea = (val, placeholder, onInput) => {
    const t = el("textarea", { width: "100%", boxSizing: "border-box", background: "#1c1e24", color: "#e6e6e6", border: "1px solid #333", borderRadius: "5px", padding: "5px", fontSize: "12px", resize: "vertical", minHeight: "72px", marginTop: "3px" }, { value: val || "", placeholder: placeholder || "" });
    t.addEventListener("input", () => onInput(t.value));
    // keep canvas keyboard shortcuts from stealing typing
    t.addEventListener("pointerdown", (e) => e.stopPropagation());
    return t;
  };

  // ---- characters ----
  root.appendChild(heading("Characters"));
  node.k2.characters.forEach((ch, ci) => {
    const color = PALETTE[ci % PALETTE.length];
    const card = el("div", { border: `1px solid #333`, borderLeft: `4px solid ${color}`, borderRadius: "6px", padding: "7px", marginBottom: "7px", background: "#23262e" });

    const head = el("div", { display: "flex", alignItems: "center", gap: "6px", marginBottom: "5px" });
    head.appendChild(el("span", { color: color, fontWeight: "700" }, { textContent: `#${ci + 1}` }));
    const nameInput = el("input", { flex: "1", background: "#1c1e24", color: "#e6e6e6", border: "1px solid #333", borderRadius: "5px", padding: "4px 6px", fontSize: "12px" }, { value: ch.name || "", placeholder: "label (optional, e.g. mother)" });
    nameInput.addEventListener("input", () => { ch.name = nameInput.value; save(node); });
    nameInput.addEventListener("pointerdown", (e) => e.stopPropagation());
    head.appendChild(nameInput);
    const del = button("✕", "#5b2b2b");
    del.title = "remove character";
    del.addEventListener("click", () => {
      node.k2.characters.splice(ci, 1);
      // shift link references
      for (const ln of node.k2.links) {
        ln.between = (ln.between || []).filter((b) => b !== ci + 1).map((b) => (b > ci + 1 ? b - 1 : b));
      }
      if (!node.k2.characters.length) node.k2.characters = [blankChar()];
      save(node); render(node);
    });
    head.appendChild(del);
    card.appendChild(head);

    // clickable grid, drawn in the real canvas shape (box aspect = image aspect)
    const gw = grid(node);
    const asp = imageAspect(node);
    const BOX_W = 180;
    const boxH = Math.max(40, Math.round(BOX_W * (asp.h / asp.w)));
    const cellW = BOX_W / gw.cols;
    const cellH = boxH / gw.rows;
    const gridBox = el("div", { display: "grid", gridTemplateColumns: `repeat(${gw.cols}, ${cellW}px)`, gridAutoRows: `${cellH}px`, gap: "2px", margin: "2px 0 5px" });
    for (let i = 0; i < gw.cols * gw.rows; i++) {
      const on = (ch.cells || []).includes(i);
      const cell = el("div", { background: on ? color : "#161821", border: `1px solid ${on ? color : "#333"}`, borderRadius: "3px", cursor: "pointer" });
      cell.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        const idx = (ch.cells || []).indexOf(i);
        if (idx >= 0) ch.cells.splice(idx, 1);
        else (ch.cells = ch.cells || []).push(i);
        save(node); render(node);
      });
      gridBox.appendChild(cell);
    }
    card.appendChild(el("div", { fontSize: "10px", color: "#8b8f98", marginBottom: "2px" }, { textContent: "where on canvas (click cells; none = whole image)" }));
    card.appendChild(gridBox);

    card.appendChild(textArea(ch.positive, "positive — looks + what they do", (v) => { ch.positive = v; save(node); }));
    card.appendChild(textArea(ch.negative, "negative — what to avoid for this character", (v) => { ch.negative = v; save(node); }));
    root.appendChild(card);
  });

  const addChar = button("+ Add character");
  addChar.addEventListener("click", () => { node.k2.characters.push(blankChar()); save(node); render(node); });
  root.appendChild(addChar);

  // ---- interactions ----
  root.appendChild(heading("Interactions"));
  node.k2.links.forEach((ln, li) => {
    const card = el("div", { border: "1px solid #333", borderRadius: "6px", padding: "7px", marginBottom: "7px", background: "#23262e" });
    const head = el("div", { display: "flex", alignItems: "center", gap: "6px", marginBottom: "5px", flexWrap: "wrap" });
    head.appendChild(el("span", { color: "#cfd2d6", fontWeight: "600" }, { textContent: "between:" }));
    node.k2.characters.forEach((ch, ci) => {
      const n = ci + 1;
      const on = (ln.between || []).includes(n);
      const chip = el("span", { background: on ? PALETTE[ci % PALETTE.length] : "#161821", color: on ? "#fff" : "#9aa0aa", border: "1px solid #333", borderRadius: "10px", padding: "2px 8px", cursor: "pointer", fontSize: "11px" }, { textContent: ch.name ? `${n}:${ch.name}` : `#${n}` });
      chip.addEventListener("click", () => {
        ln.between = ln.between || [];
        const idx = ln.between.indexOf(n);
        if (idx >= 0) ln.between.splice(idx, 1);
        else ln.between.push(n);
        ln.between.sort((a, b) => a - b);
        save(node); render(node);
      });
      head.appendChild(chip);
    });
    const del = button("✕", "#5b2b2b");
    del.addEventListener("click", () => { node.k2.links.splice(li, 1); save(node); render(node); });
    head.appendChild(del);
    card.appendChild(head);
    card.appendChild(textArea(ln.positive, "interaction positive — e.g. kissing, arms wrapped around each other", (v) => { ln.positive = v; save(node); }));
    card.appendChild(textArea(ln.negative, "interaction negative — e.g. not touching, apart", (v) => { ln.negative = v; save(node); }));
    root.appendChild(card);
  });

  const addLink = button("+ Add interaction");
  addLink.addEventListener("click", () => { node.k2.links.push({ between: [], positive: "", negative: "" }); save(node); render(node); });
  root.appendChild(addLink);

  // size the editor to fit. Read scrollHeight live each call (not frozen): the
  // DOM element is mounted by the frontend on a later tick, so at first render
  // scrollHeight is 0 - remeasure() forces a re-fit once it is attached.
  if (node.k2_widget) {
    node.k2_widget.computeSize = () => [node.size ? node.size[0] : 360, (node.k2_root.scrollHeight || 0) + 12];
  }
  node.setDirtyCanvas(true, true);
}

function remeasure(node) {
  // the DOM widget's element attaches on a later Vue tick; re-render after two
  // frames so scrollHeight is real and the node resizes to fit.
  const raf = window.requestAnimationFrame || ((f) => setTimeout(f, 16));
  raf(() => raf(() => { try { render(node); } catch (e) { /* ignore */ } }));
}

function hideRaw(node) {
  const w = getWidget(node, "layout_json");
  if (!w) return;
  w.hidden = true;
  w.computeSize = () => [0, -4];
  if (w.element) w.element.style.display = "none";
  if (w.inputEl) w.inputEl.style.display = "none";
}

function setup(node) {
  if (node.k2_root) return;
  const root = el("div", { width: "100%", padding: "2px 4px", boxSizing: "border-box", overflowY: "auto", fontFamily: "sans-serif" });
  node.k2_root = root;
  node.k2_widget = node.addDOMWidget("krea2_editor", "krea2", root, { hideOnZoom: false });
  node.k2_widget.serialize = false;   // data lives in layout_json; don't double-save it
  // re-render when the aspect or grid dimensions change
  for (const name of ["aspect", "grid_cols", "grid_rows"]) {
    const w = getWidget(node, name);
    if (w) {
      const prev = w.callback;
      w.callback = function () {
        const r = prev ? prev.apply(this, arguments) : undefined;
        dropInvalidCells(node);
        save(node);
        render(node);
        return r;
      };
    }
  }
  loadState(node);
  render(node);
  hideRaw(node);     // hide the raw JSON widget only after a clean render, so a
                     // render failure leaves it visible as the editable fallback
  remeasure(node);
}

app.registerExtension({
  name: "Regional.MultiChar",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      try { setup(this); } catch (e) { console.error("[RegionalMultiChar] setup failed", e); }
      return r;
    };

    // when a saved workflow loads, widget values (incl. layout_json) arrive here
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      try {
        if (!this.k2_root) setup(this);
        loadState(this);
        render(this);
        remeasure(this);
      } catch (e) { console.error("[RegionalMultiChar] configure failed", e); }
      return r;
    };
  },
});
