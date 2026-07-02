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
import { api } from "../../scripts/api.js";

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
  // live preview: nudge any connected compose nodes to rebuild (debounced)
  try { pokeComposeConsumers(node); } catch (e) { /* ignore */ }
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
    const t = el("textarea", { width: "100%", boxSizing: "border-box", background: "#1c1e24", color: "#e6e6e6", border: "1px solid #333", borderRadius: "5px", padding: "5px", fontSize: "12px", resize: "vertical", minHeight: "104px", marginTop: "3px" }, { value: val || "", placeholder: placeholder || "" });
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

// ---------------------------------------------------------------------------
// Read-only markdown viewer for the MultiCharPromptPreview node. Wire the
// compose node's "prompt_report" (or any STRING) in to see the assembled prompt.
// Nothing here is load-bearing for generation - only for debugging comfort.
// ---------------------------------------------------------------------------
const PREVIEW_NODE = "MultiCharPromptPreview";

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// tiny, safe markdown subset -> HTML (escapes first, so prompt text can't inject)
function mdToHtml(md) {
  const blocks = [];
  md = String(md || "").replace(/```([\s\S]*?)```/g, (m, code) => {
    blocks.push(
      '<pre style="background:#0d0f14;border:1px solid #2a2d36;border-radius:5px;padding:7px;' +
      'overflow-x:auto;white-space:pre-wrap;word-break:break-word;margin:4px 0">' +
      escapeHtml(code.replace(/^\n/, "").replace(/\n$/, "")) + "</pre>");
    return "@@@" + (blocks.length - 1) + "@@@";
  });
  let html = escapeHtml(md);
  html = html.replace(/`([^`]+)`/g, '<code style="background:#0d0f14;padding:1px 4px;border-radius:3px">$1</code>');
  html = html.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  const lines = html.split("\n");
  const out = [];
  let inList = false;
  const closeList = () => { if (inList) { out.push("</ul>"); inList = false; } };
  for (const ln of lines) {
    const h = ln.match(/^(#{1,4})\s+(.*)$/);
    if (h) { closeList(); const lvl = h[1].length; out.push('<h' + lvl + ' style="margin:8px 0 3px;color:#8ab4f8">' + h[2] + "</h" + lvl + ">"); continue; }
    if (/^---+$/.test(ln.trim())) { closeList(); out.push('<hr style="border:none;border-top:1px solid #2a2d36;margin:6px 0">'); continue; }
    const li = ln.match(/^\s*[-*]\s+(.*)$/);
    if (li) { if (!inList) { out.push('<ul style="margin:2px 0;padding-left:18px">'); inList = true; } out.push("<li>" + li[1] + "</li>"); continue; }
    const bq = ln.match(/^>\s?(.*)$/);
    if (bq) { closeList(); out.push('<div style="border-left:3px solid #2a2d36;padding-left:8px;color:#c7ccd6;margin:2px 0">' + bq[1] + "</div>"); continue; }
    if (ln.trim() === "") { closeList(); out.push('<div style="height:6px"></div>'); continue; }
    out.push("<div>" + ln + "</div>");
  }
  closeList();
  return out.join("\n").replace(/@@@(\d+)@@@/g, (m, i) => blocks[+i]);
}

app.registerExtension({
  name: "Regional.MultiCharPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== PREVIEW_NODE) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      const div = el("div", {
        width: "100%", minHeight: "80px", maxHeight: "620px", overflowY: "auto",
        padding: "6px 10px", boxSizing: "border-box", background: "#14161c",
        color: "#dfe3ea", border: "1px solid #2a2d36", borderRadius: "6px",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
        fontSize: "11.5px", lineHeight: "1.5", wordBreak: "break-word",
      });
      div.innerHTML = '<div style="color:#6b7280">(run the graph to see the assembled prompt)</div>';
      this.mc_preview = div;
      this.mc_widget = this.addDOMWidget("mc_report", "mc_report", div, { hideOnZoom: false });
      this.mc_widget.serialize = false;
      this.mc_widget.computeSize = () => [this.size ? this.size[0] : 420, Math.min(620, (div.scrollHeight || 90) + 14)];
      if (!this.size || this.size[0] < 380) this.size = [440, 340];
      return r;
    };

    const onExec = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExec ? onExec.apply(this, arguments) : undefined;
      try {
        let txt = "";
        if (message && message.text) txt = Array.isArray(message.text) ? message.text.join("") : String(message.text);
        if (this.mc_preview) {
          this.mc_preview.innerHTML = mdToHtml(txt);
          this.setDirtyCanvas(true, true);
        }
      } catch (e) { console.error("[MultiCharPreview] render failed", e); }
      return r;
    };
  },
});

// ---------------------------------------------------------------------------
// LIVE in-editor preview on the MultiCharPromptCompose node. Gathers the node's
// own settings + the connected RegionalCharacterLayout, POSTs them to the
// backend /multichar/preview route (which runs the SAME python assembler used
// for generation), and renders the returned markdown. 3s debounce so it stays
// responsive while typing. No run needed.
// ---------------------------------------------------------------------------
const COMPOSE_NODE = "MultiCharPromptCompose";
const COMPOSE_SETTINGS = ["subject_count_lock", "use_names", "bind_interactions",
  "cast_roster", "order_and_group", "auto_scale_hints", "spatial_detail",
  "output_format", "auto_framing", "negative_mode", "global_positive", "global_negative"];

// in-node options guide (rendered with mdToHtml). No apostrophes so it stays
// safe inside single-quoted JS strings; double quotes + backticks are literal.
const HELP_MD = [
  "## Multi-Char Prompt Compose — options",
  "",
  "Builds ONE natural-language prompt from the connected **Regional Characters** editor. Best on strong text encoders (Chroma / Flux / SD3). Every aid OFF = plain old concatenation.",
  "",
  "**global_positive / global_negative** — scene-wide style + environment, and things to avoid everywhere. Put your art style and setting here.",
  "",
  "**subject_count_lock** (on/off) — ON adds \"There are exactly N people and no one else\". Stops dropped or extra characters. Turn OFF only if you want the model free to add background people.",
  "",
  "**use_names** — how each character is referred to:",
  "- `handle` (default) — a natural referent like \"the blonde woman\", derived from the name/description and reused in the roster + interactions. Best coherence.",
  "- `label` — \"Woman: ...\" prefix style. Cleanest to read.",
  "- `off` — no subject prefix (old behavior). Good for a single character.",
  "",
  "**bind_interactions** (on/off) — ON rewrites an interaction to name its two characters (\"The woman and the man are kissing...\") using the between chips, so you type only the action (\"kissing...\"). OFF uses the interaction text verbatim.",
  "",
  "**cast_roster** (on/off) — ON adds a \"Cast: ...\" line up front so the model registers distinct people before the detailed sentences. Reduces two characters merging. Turn off for 1 to 2 simple characters.",
  "",
  "**order_and_group** (on/off) — ON sorts characters left to right and merges same-cell characters into one clause. OFF keeps card order, each on its own line. Keep ON for multi-character scenes.",
  "",
  "**auto_scale_hints** (on/off) — ON adds size words from grid depth: top row becomes small/distant, bottom row becomes large/close. Needs a grid with more than 1 row. Fixes a background character rendering too big.",
  "",
  "**spatial_detail** — how grid cells become words:",
  "- `fine` (default) — left / center / right plus upper / lower areas.",
  "- `coarse` — just left / center / right (plus fore/background). Simplest.",
  "- `grid_coords` — fine wording PLUS spreadsheet tags (A1, B2). Experimental extra cue; do not rely on it alone.",
  "",
  "**output_format** — sentence style:",
  "- `prose` (default) — flowing sentences. Best for the model.",
  "- `labeled` — \"Name (location): ...\". Most readable for you.",
  "- `numbered` — \"1) ...\". Most explicit structure.",
  "",
  "**auto_framing** (on/off) — ON derives a shot: wide establishing shot when characters are spread out, otherwise medium shot. Turn OFF if you set the shot yourself in global_positive.",
  "",
  "**negative_mode** — per-character negatives (negatives are global on Chroma):",
  "- `global_dedup` (default) — merge and de-duplicate all negatives.",
  "- `to_positive_assertion` — convert a character negative into a positive counter-trait (\"old\" becomes \"young\") on that character. Often obeyed better than a negation.",
].join("\n");

// build the exact payload the backend expects from a compose node
function composeGather(node) {
  const layout = { grid_cols: 3, grid_rows: 1, characters: [], links: [] };
  try {
    const slot = (node.inputs || []).findIndex((i) => i.name === "layout");
    if (slot >= 0 && node.inputs[slot].link != null && node.graph) {
      const lnk = node.graph.links[node.inputs[slot].link];
      const src = lnk && node.graph.getNodeById(lnk.origin_id);
      if (src) {
        let parsed = {};
        const gj = getWidget(src, "layout_json");
        try { parsed = JSON.parse(gj && gj.value ? gj.value : "{}"); } catch (e) { /* */ }
        layout.characters = parsed.characters || [];
        layout.links = parsed.links || [];
        layout.grid_cols = parseInt(widgetVal(src, "grid_cols", 3)) || 3;
        layout.grid_rows = parseInt(widgetVal(src, "grid_rows", 1)) || 1;
      }
    }
  } catch (e) { /* leave defaults */ }
  const opts = {};
  for (const name of COMPOSE_SETTINGS) {
    const w = getWidget(node, name);
    if (w) opts[name] = w.value;
  }
  return { layout, opts };
}

async function composeRefresh(node) {
  if (!node.mc_preview) return;
  try {
    const resp = await api.fetchApi("/multichar/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(composeGather(node)),
    });
    const data = await resp.json();
    node._lastReport = data.report || "";
    node.mc_preview.innerHTML = mdToHtml(node._lastReport || "(no preview)");
  } catch (e) {
    node.mc_preview.innerHTML = '<div style="color:#c0616b">preview unavailable: ' + escapeHtml(String(e)) + "</div>";
  }
  node.setDirtyCanvas(true, true);
}

function composeScheduleRefresh(node) {
  if (!node) return;
  if (node.mc_timer) clearTimeout(node.mc_timer);
  node.mc_timer = setTimeout(() => composeRefresh(node), 3000);  // 3s debounce
}

// called from the layout editor's save(): refresh compose nodes fed by this layout
function pokeComposeConsumers(layoutNode) {
  const out = (layoutNode.outputs || [])[0];
  if (!out || !out.links || !layoutNode.graph) return;
  for (const lid of out.links) {
    const lnk = layoutNode.graph.links[lid];
    if (!lnk) continue;
    const tgt = layoutNode.graph.getNodeById(lnk.target_id);
    if (tgt && tgt.type === COMPOSE_NODE) composeScheduleRefresh(tgt);
  }
}

// hide a standard multiline widget and edit its value through a big, resizable
// textarea instead (bigger start height than the default widget).
function bigTextarea(self, widgetName, label, minH) {
  const w = getWidget(self, widgetName);
  const wrap = el("div", { width: "100%", marginTop: "5px" });
  wrap.appendChild(el("div", { fontSize: "10px", color: "#8b8f98", margin: "2px 0", textTransform: "uppercase", letterSpacing: ".04em" }, { textContent: label }));
  const ta = el("textarea", {
    width: "100%", boxSizing: "border-box", background: "#1c1e24", color: "#e6e6e6",
    border: "1px solid #333", borderRadius: "5px", padding: "6px", fontSize: "12px",
    resize: "vertical", minHeight: minH + "px", lineHeight: "1.4",
  }, { value: (w && w.value) || "" });
  ta.addEventListener("pointerdown", (e) => e.stopPropagation());
  ta.addEventListener("input", () => { if (w) w.value = ta.value; composeScheduleRefresh(self); });
  wrap.appendChild(ta);
  if (w) {  // hide the original widget; this textarea now edits its value
    w.hidden = true;
    w.computeSize = () => [0, -4];
    if (w.inputEl) w.inputEl.style.display = "none";
    if (w.element) w.element.style.display = "none";
  }
  return { wrap, ta, w };
}

app.registerExtension({
  name: "Regional.MultiCharComposeLive",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== COMPOSE_NODE) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      const self = this;
      const root = el("div", { width: "100%", padding: "2px 4px", boxSizing: "border-box", fontFamily: "sans-serif" });

      // header: label + help / copy / refresh buttons
      const head = el("div", { display: "flex", alignItems: "center", gap: "6px", margin: "2px 0 4px" });
      head.appendChild(el("span", { fontSize: "10px", color: "#8b8f98", textTransform: "uppercase", letterSpacing: ".04em", flex: "1" }, { textContent: "live prompt preview (3s after edits)" }));
      const mkBtn = (txt) => {
        const b = el("button", { background: "#3a3f4b", color: "#fff", border: "none", borderRadius: "4px", padding: "2px 8px", cursor: "pointer", fontSize: "11px" }, { textContent: txt });
        b.addEventListener("pointerdown", (e) => e.stopPropagation());
        return b;
      };
      const helpBtn = mkBtn("ⓘ help");
      const copyBtn = mkBtn("📋 copy");
      const nowBtn = mkBtn("↻ now");
      head.appendChild(helpBtn);
      head.appendChild(copyBtn);
      head.appendChild(nowBtn);

      // collapsible options help panel
      const help = el("div", {
        display: "none", margin: "2px 0 4px", padding: "6px 10px", background: "#0f1116",
        color: "#c7ccd6", border: "1px solid #2a2d36", borderRadius: "6px",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: "11px",
        lineHeight: "1.5", maxHeight: "360px", overflowY: "auto", wordBreak: "break-word",
      });
      help.innerHTML = mdToHtml(HELP_MD);
      helpBtn.addEventListener("click", () => {
        help.style.display = help.style.display === "none" ? "block" : "none";
        self.setDirtyCanvas(true, true);
      });

      nowBtn.addEventListener("click", () => composeRefresh(self));
      copyBtn.addEventListener("click", async () => {
        const md = self._lastReport || (self.mc_preview ? self.mc_preview.innerText : "") || "";
        const done = () => { copyBtn.textContent = "✓ copied"; setTimeout(() => (copyBtn.textContent = "📋 copy"), 1200); };
        try { await navigator.clipboard.writeText(md); done(); }
        catch (e) {
          const t = document.createElement("textarea"); t.value = md; document.body.appendChild(t); t.select();
          try { document.execCommand("copy"); } catch (_) { /* */ }
          document.body.removeChild(t); done();
        }
      });

      // bigger, resizable global positive/negative text areas
      const posTA = bigTextarea(self, "global_positive", "Global positive (scene style + environment)", 130);
      const negTA = bigTextarea(self, "global_negative", "Global negative (avoid everywhere)", 90);
      this._taRefs = { global_positive: posTA, global_negative: negTA };

      // rendered live-preview panel
      const div = el("div", {
        width: "100%", minHeight: "70px", maxHeight: "520px", overflowY: "auto",
        padding: "6px 10px", boxSizing: "border-box", background: "#14161c",
        color: "#dfe3ea", border: "1px solid #2a2d36", borderRadius: "6px", marginTop: "5px",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
        fontSize: "11px", lineHeight: "1.45", wordBreak: "break-word",
      });
      div.innerHTML = '<div style="color:#6b7280">(edit the scene — preview builds 3s after you stop typing)</div>';
      this.mc_preview = div;

      root.appendChild(head);
      root.appendChild(help);
      root.appendChild(posTA.wrap);
      root.appendChild(negTA.wrap);
      root.appendChild(div);

      this.mc_widget = this.addDOMWidget("mc_ui", "mc_ui", root, { hideOnZoom: false });
      this.mc_widget.serialize = false;
      this.mc_widget.computeSize = () => [this.size ? this.size[0] : 500, (root.scrollHeight || 120) + 12];
      if (!this.size || this.size[0] < 440) this.size = [500, 660];

      // rebuild 3s after any setting widget changes (text areas handle their own)
      for (const w of (this.widgets || [])) {
        if (w.name === "mc_ui" || w.name === "global_positive" || w.name === "global_negative") continue;
        const prev = w.callback;
        w.callback = function () {
          const rr = prev ? prev.apply(this, arguments) : undefined;
          composeScheduleRefresh(self);
          return rr;
        };
      }
      // resize/help toggles -> recompute node height
      if (window.ResizeObserver) {
        try { new ResizeObserver(() => self.setDirtyCanvas(true, true)).observe(root); } catch (e) { /* */ }
      }
      composeScheduleRefresh(this);
      return r;
    };

    const onConn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConn ? onConn.apply(this, arguments) : undefined;
      composeScheduleRefresh(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      try {  // re-sync the text areas from the loaded widget values + re-hide originals
        if (this._taRefs) {
          for (const name in this._taRefs) {
            const ref = this._taRefs[name];
            const w = getWidget(this, name);
            if (ref && w) {
              ref.ta.value = w.value || "";
              w.hidden = true;
              w.computeSize = () => [0, -4];
              if (w.inputEl) w.inputEl.style.display = "none";
              if (w.element) w.element.style.display = "none";
            }
          }
        }
      } catch (e) { /* */ }
      composeScheduleRefresh(this);
      return r;
    };
  },
});
