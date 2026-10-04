# ComfyUI Regional MultiChar

**Model-agnostic regional multi-character conditioning for ANY SDXL / SD1.5 / Flux / Chroma / SD3 checkpoint.**

Place characters on a grid, give each one their own prompt, link two for an interaction — and get plain masked CONDITIONING that feeds a standard KSampler. No Krea-2, RES4LYF, or ClownsharKSampler dependency required.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Node Reference](#node-reference)
   - [Regional Characters (grid layout)](#1-regional-characters-grid-layout)
   - [Regional Multi-Char Conditioning](#2-regional-multi-char-conditioning)
   - [Multi-Char Prompt Compose (wording)](#3-multi-char-prompt-compose-wording)
   - [Multi-Char Prompt Preview (read-only)](#4-multi-char-prompt-preview-read-only)
   - [Multi-Char Layout Enhancer (LLM, optional)](#5-multi-char-layout-enhancer-llm-optional)
   - [Regional FaceDetailer Toggle](#6-regional-facedetailer-toggle)
   - [Regional Hires Toggle](#7-regional-hires-toggle)
   - [Grayscale Filter (optional)](#8-grayscale-filter-optional)
   - [Seed Label (optional)](#9-seed-label-optional)
4. [Technical Deep-Dive](#technical-deep-dive)
   - [Layout JSON Schema](#layout-json-schema)
   - [Mask Building Pipeline](#mask-building-pipeline)
   - [Conditioning Pipeline](#conditioning-pipeline)
   - [Composition Fractions](#composition-fractions)
   - [Wording-Based Composition (no masks)](#wording-based-composition-no-masks)
   - [LLM Layout Enhancer](#llm-layout-enhancer)
   - [Live Preview HTTP Route](#live-preview-http-route)
   - [Frontend Editor](#frontend-editor)
5. [Typical Workflows](#typical-workflows)
6. [Tips & Gotchas](#tips--gotchas)

---

## Overview

Regional MultiChar solves the core problem of multi-character diffusion: getting each character to stay in their own part of the canvas without fighting over conditioning. It works at two levels:

| Approach | Best for | Key nodes |
|---|---|---|
| **Masked regional conditioning** | SDXL / SD1.5 / Pony / Illustrious / NoobAI | `RegionalCharacterLayout` → `RegionalMultiCharConditioning` |
| **Wording-based composition** | Flux / Chroma / SD3 (strong text encoders, no cross-attention masks) | `RegionalCharacterLayout` → `MultiCharPromptCompose` |

Both approaches share the same visual grid editor.

---

## Installation

1. Clone or drop the folder into `ComfyUI/custom_nodes/`:
   ```
   ComfyUI/custom_nodes/ComfyUI-Regional-MultiChar/
   ```
2. Restart ComfyUI. Conditioning nodes appear under **Regional/conditioning**; the grayscale filter and seed label appear under **Regional/image**.

No additional Python packages are required for the core nodes. The optional `MultiCharLayoutEnhancer` node also requires `transformers` (already present in most ComfyUI environments).

---

## Node Reference

### 1. Regional Characters (grid layout)

**Internal name:** `RegionalCharacterLayout`

The entry point for both pipelines. Defines the canvas aspect, the grid dimensions, and all character / interaction data.

| Widget | Type | Description |
|---|---|---|
| `aspect` | Combo | One of five preset resolutions (`wide 1536x832`, `rectangle 1216x832`, `square 1024x1024`, `portrait 832x1216`, `tall 832x1536`) |
| `grid_cols` | INT 1–8 | Number of grid columns |
| `grid_rows` | INT 1–8 | Number of grid rows |
| `batch_size` | INT 1–64 | Latent batch size |
| `layout_json` | STRING (hidden) | Serialized JSON; the visual editor writes here automatically |

**Outputs:**

| Name | Type | Content |
|---|---|---|
| `layout` | `REGIONAL_LAYOUT` | Python dict: `{grid_cols, grid_rows, width, height, characters, links}` |
| `latent` | `LATENT` | Empty zero-filled latent matching the chosen aspect and batch size |

The `REGIONAL_LAYOUT` type is a plain Python `dict` — not a tensor — passed between nodes as a structured bundle.

The **Structured character prompts** checkbox above the character cards is on by default. Each character then has six text fields: Generic, Looks, and Pose / action, each with a positive and negative field. Keep recurring appearance details in Looks and change Pose / action for each scene. Interactions keep their existing positive and negative fields. Turn the checkbox off to edit characters with the older two-field layout. Switching off combines the three positive fields into `positive` and the three negative fields into `negative`; switching on again puts the combined text in Generic, so copy the JSON first if you need to preserve the separation.

---

### 2. Regional Multi-Char Conditioning

**Internal name:** `RegionalMultiCharConditioning`

Converts a `REGIONAL_LAYOUT` into masked `CONDITIONING` tensors ready for any standard KSampler.

| Widget | Type | Description |
|---|---|---|
| `clip` | CLIP | The model's text encoder |
| `latent` | LATENT | Used to read the spatial dimensions (height × width in latent space) |
| `feather` | INT 0–512 | Gaussian blur radius **in pixel space** applied to each region mask (divided by 8 for latent space) |
| `region_strength` | FLOAT 0–10 | `mask_strength` passed to `ConditioningSetMask`; values above 1.0 amplify the region signal |
| `global_positive` | STRING | Full-canvas positive that every pixel receives regardless of regions |
| `global_negative` | STRING | Global negative (or base for folded-negative mode) |
| `auto_split` | BOOLEAN | When ON, a grid cell claimed by two characters is split into vertical strips so they each get a distinct mask strip rather than identical masks |
| `regional_negatives` | BOOLEAN | ON = encode each character's negative separately with its own mask (slower, more precise). OFF = collect all per-character negatives and encode one combined global negative (faster) |
| `merge_linked` | BOOLEAN | ON = characters joined by an interaction link are merged into one region (union of cells, concatenated prompts). OFF = each character keeps its own region, and each link is an additional region |
| `composition_frac` | FLOAT 0.05–1.0 | Fraction of the denoising schedule that uses per-region conditioning. After this point a single merged full-scene prompt takes over |
| `layout` | REGIONAL_LAYOUT (optional) | The bundle from `RegionalCharacterLayout` |

**Outputs:**

| Name | Type |
|---|---|
| `positive` | CONDITIONING |
| `negative` | CONDITIONING |
| `latent` | LATENT (pass-through) |
| `mask_preview` | IMAGE (coloured overlay, one hue per region) |

---

### 3. Multi-Char Prompt Compose (wording)

**Internal name:** `MultiCharPromptCompose`

Wording-based alternative to masked conditioning. Assembles **one** natural-language prompt from the layout — no masks, no per-region encoding. Best on strong text encoders (Flux, Chroma, SD3).

| Widget | Default | Description |
|---|---|---|
| `global_positive` | — | Scene-wide style, environment, lighting |
| `global_negative` | — | Things to avoid everywhere |
| `subject_count_lock` | ON | Adds "There are exactly N people and no one else" — stops dropped or extra characters |
| `use_names` | `handle` | `handle` = natural referent like "the blonde woman"; `label` = "Woman: ..."; `off` = no prefix |
| `bind_interactions` | ON | Rewrites interaction text to name its characters: "The woman and the man are kissing..." |
| `cast_roster` | ON | Adds a "Cast: ..." line before the detailed sentences |
| `order_and_group` | ON | Sorts characters left-to-right and merges same-cell characters into one clause |
| `auto_scale_hints` | ON | Adds depth words from grid row: top row = small/distant, bottom row = large/close |
| `spatial_detail` | `fine` | `fine` = "in the upper left area"; `coarse` = "on the left side"; `grid_coords` = fine + spreadsheet tags (A1, B2) |
| `output_format` | `prose` | `prose` = flowing sentences; `labeled` = "Name (loc): ..."; `numbered` = "1) ..." |
| `auto_framing` | OFF | Derives shot type from character spread (wide / medium) |
| `negative_mode` | `global_dedup` | `global_dedup` = merge + deduplicate all negatives; `to_positive_assertion` = convert negatives to positive counter-traits ("old" → "young") |
| `layout` | optional | From `RegionalCharacterLayout` |
| `prompt_profile` | `default` | `default` keeps the current positive and negative text; `flux2` uses the cleanup rules below |
| `negative_term_cap` | `0` | Maximum number of deduplicated negative terms; `0` keeps all terms |
| `negative_cap_priority` | `global_first` | `global_first` keeps global terms before character and interaction terms; `interaction_first` puts interaction guards first in `flux2` |
| `spatial_cues` | `auto` | In `flux2`, `auto` skips grid wording for a 1x1 grid or when all characters share the same cells; `always` forces it; `off` removes it |
| `scale_cues` | `off` | In `flux2`, `row_based` allows the existing top=far/bottom=near scale hint; `off` avoids assuming that vertical position means depth |
| `layout_json_override` | empty | Legacy layout override retained for saved workflows; the composer hides this widget in new workflows |

**Outputs:** `positive` (CONDITIONING), `negative` (CONDITIONING), `positive_text` (STRING), `negative_text` (STRING), `prompt_report` (STRING markdown).

For Flux 2 Klein, start with `prompt_profile=flux2` and leave `negative_term_cap=0` for the first comparison. Set a cap only after checking which terms it drops in the report. `interaction_first` is useful when a long global negative would otherwise push contact or adjacency guards past the cap. The cap changes the negative prompt in either profile when you set it above zero.

Check the sampler before tuning negatives. ComfyUI [skips the unconditional pass at CFG 1](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/samplers.py), so a workflow at that setting may ignore this node's negative output. The [reference Flux 2 Klein pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py) also skips classifier-free guidance for distilled models. If negatives are inactive in your workflow, changing the cap cannot improve gaze or anatomy. Keep the negative text output for workflows that do use it.

In `flux2`, the composer also skips a count-lock sentence if the global positive already states the same headcount, removes exact comma-separated clauses repeated between a character and its interaction, and places each interaction after the last group containing one of its members. It keeps unique interaction details. The wording is cleaned up at sentence joins. A plain role name such as `Mature Woman` becomes `the mature woman` in prose handles, while labeled and numbered character lines keep the name as entered.

The composer has a **Prompt settings JSON (edit / paste -> Apply)** box. It shows this node's `global_positive`, `global_negative`, and all prompt assembly settings. Copy the JSON to edit it elsewhere, then paste it and click **Apply settings JSON**. A partial object updates only the keys it contains. You can also paste a `widgets_values_named` object from a workflow. The box stays in sync when you change a widget or either global prompt, unless you have an un-applied JSON edit in progress. Character positions and interactions belong in the **Layout JSON** box on `RegionalCharacterLayout`; the composer settings box does not include `grid_cols`, `grid_rows`, `characters`, `links`, `aspect`, or `batch_size`.

Older workflows with a nonempty `layout_json_override` still use that override. The composer shows a notice and a **Clear old override** button when one is active. New settings JSON does not create a layout override.

Saved workflows without these widgets still use the old defaults. Existing node names and port order are unchanged. Old two-field character JSON remains valid alongside the new structured format.

---

### 4. Multi-Char Prompt Preview (read-only)

**Internal name:** `MultiCharPromptPreview`

Connect `MultiCharPromptCompose.prompt_report` to the preview's existing `text` input. After a graph run, the preview extracts the final positive and negative strings from that report and shows them in separate **POSITIVE** and **NEGATIVE** blocks. These are the strings the composer passed to `clip.tokenize()`. Each block starts in raw-text mode, has its own **Copy** button, and can switch to a rendered Markdown view. An empty string stays blank; the preview does not put an `(empty)` placeholder in the encoder text.

The preview also has optional `positive_text` and `negative_text` inputs. Connect the composer's matching outputs to them when you want the raw values passed directly, including prompts that contain Markdown code fences. The original `text` input and passthrough `text` output keep their positions, so saved workflows with only the `prompt_report` link still work. If `text` receives an unrelated Markdown string, the node renders it as before.

The separate live preview panel on `MultiCharPromptCompose` still POSTs to `/multichar/preview` and updates 3 seconds after edits, without running the graph.

---

### 5. Multi-Char Layout Enhancer (LLM, optional)

**Internal name:** `MultiCharLayoutEnhancer`

Enriches each character's combined positive prompt (and each interaction's `positive`) in a layout using a local HuggingFace instruct LLM. For structured characters, its output is a runtime `positive` override; the upstream editor's six fields stay unchanged. Drop an HF model directory under `models/LLM/` and it appears in the dropdown.

| Widget | Description |
|---|---|
| `enable` | OFF = passthrough (safe to leave in graph) |
| `model` | HF model directory found under `models/LLM/` |
| `enrich_characters` / `enrich_interactions` | Which parts to rewrite |
| `max_words` | Soft upper bound for character descriptions |
| `smart_wording` | ON = adaptive length; OFF = always aim for the cap |
| `temperature` / `top_p` / `seed` | Standard LLM sampling controls |

The node frees ComfyUI's GPU models before loading the LLM and frees the LLM before returning, so it fits alongside large image models. Any failure returns the layout unchanged (never breaks a run).

---

### 6. Regional FaceDetailer Toggle

**Internal name:** `RegionalFaceDetailerSwitch`

A single boolean that selects between a base image (no FaceDetailer) and an image processed with FaceDetailer. Both inputs are **lazy** — when OFF, the FaceDetailer branch never executes and costs no compute.

---

### 7. Regional Hires Toggle

**Internal name:** `RegionalHiresSwitch`

Same pattern for hires second-pass latents. When OFF, the upscale + second KSampler branch is completely skipped.

---

### 8. Grayscale Filter (optional)

**Internal name:** `RegionalGrayscaleFilter`

Connect the final decoded `IMAGE` to this node, then connect its output to Save Image. Put it after FaceDetailer or any other image pass that might add color. With `enabled` on (the default), it converts RGB pixels to grayscale using weighted luminance and returns a three-channel image. It preserves an alpha channel if one is present. Inputs with fewer than three channels pass through unchanged. With `enabled` off, it passes the image through unchanged.

| Tweak | Default | Effect |
|---|---:|---|
| `grayscale_strength` | `1.0` | `1.0` removes all color; `0.0` leaves the input unchanged. Values between them mix some original color back in. |
| `brightness` | `0.0` | Adds or subtracts light after conversion. Range: `-0.5` to `0.5`. |
| `contrast` | `1.0` | Below `1.0` softens dark and light tones; above `1.0` separates them more. Range: `0.0` to `2.0`. |
| `black_lift` | `0.0` | Raises black toward gray without turning white darker. Range: `0.0` to `1.0`. |

The tone controls run before the strength mix. For a lighter manga print, keep `grayscale_strength=1.0` and try `black_lift=0.05` to `0.15`. Increase `brightness` slightly if the whole image is too dark. The new tweaks are optional inputs; saved workflows without them use the defaults and produce the same grayscale as before.

This is a final image filter. It catches colored pixels left in a manga image without changing the prompt, sampler, or other nodes. It cannot repair line art or anatomy.

---

### 9. Seed Label (optional)

**Internal name:** `RegionalSeedLabel`

Connect a decoded `IMAGE` to `image` and an `INT` to `seed`. The node writes `Seed: 42` (or your chosen prefix and seed) on every image in the batch, then returns an `IMAGE`. Put it after the grayscale filter if you use one. With `enabled=false`, it returns the input image unchanged.

For an exact label, use ComfyUI's built-in `PrimitiveInt` as a shared seed source: connect its `INT` output to both `KSampler.seed` and `RegionalSeedLabel.seed`. Set the seed source's **control after generate** to `fixed`. KSampler has no seed output, so typing the same number into two separate widgets does not keep them synchronized if either one changes.

| Tweak | Default | Effect |
|---|---|---|
| `position` | `bottom_right` | Choose a corner, or `custom` for percentage coordinates. |
| `optimize` | `none` | `none` uses your text and shadow colors; `grayscale` uses white text with a black outline; `photo_realistic` uses white text on a dark translucent plate. |
| `font` / `font_file` | `sans` / empty | Choose sans, mono, serif, or the Pillow default. A local `.ttf` or `.otf` path takes priority. |
| `font_size` / `margin` | `28` / `24` | Text size and distance from the edge, in pixels. The node shrinks text if it would exceed the image. |
| `text_color` | `#FFFFFF` | RGB hex or a Pillow color name. Used by `optimize=none`. |
| `shadow` / `shadow_color` / `shadow_offset` | on / `#000000` / `2` | Draw a second copy behind the label. Grayscale mode forces its shadow to black. |
| `prefix` | `Seed: ` | Text before the numeric seed. Use an empty prefix for digits only. |
| `x_percent` / `y_percent` | `50` / `50` | Used only at `position=custom`; `0` is left or top and `100` is right or bottom. |

The label is drawn after sampling and decoding. `optimize=grayscale` changes the label styling, not the image colors. Existing workflows without this node run as before. Pillow and NumPy are provided by a standard ComfyUI install.

---

## Technical Deep-Dive

### Layout JSON Schema

The visual editor in the node panel serializes everything into a single hidden widget called `layout_json`. The schema:

```json
{
  "structured_characters": true,
  "characters": [
    {
      "name": "mother",
      "cells": [0, 3, 6],
      "generic_positive": "1girl",
      "looks_positive": "long hair, red dress",
      "pose_action_positive": "walking forward",
      "generic_negative": "extra person",
      "looks_negative": "wrong outfit",
      "pose_action_negative": "sitting"
    }
  ],
  "links": [
    {
      "between": [1, 2],
      "positive": "hugging each other",
      "negative": "apart"
    }
  ]
}
```

- **`cells`** — zero-based flat cell indices (`row * cols + col`). Cell 0 is top-left; cell `cols-1` is top-right.
- **`between`** — 1-based character indices matching the card order in the editor.
- A character with no cells (`[]`) receives a full-canvas mask (conditioned everywhere).
- **`structured_characters`** — defaults to `true` when missing. In that mode, the editor moves old character `positive` and `negative` text into `generic_positive` and `generic_negative` without changing the prompt. Set it to `false` to keep character `positive` and `negative` keys instead.
- Character text is combined as Generic, Looks, then Pose / action. The runtime layout bundle also provides combined `positive` and `negative` values for nodes that read the old keys. Links are unchanged.
- Use one character format at a time in pasted JSON. If old and structured fields are both present, a nonempty old `positive` or `negative` wins for that side and moves to Generic in the editor.

A full-layout JSON (including `aspect`, `grid_cols`, `grid_rows`, `batch_size`) can be pasted into the "Layout JSON (edit / paste -> Apply)" textarea in the `RegionalCharacterLayout` editor to load the whole scene in one shot. The composer JSON box accepts prompt settings instead of layout fields.

---

### Mask Building Pipeline

**File: `regional_multichar.py` → `_build_region_masks()`**

1. **Cell → pixel box** (`_cell_box`): converts a flat cell index to `(y0, y1, x0, x1)` in latent-space pixels using `round()` to avoid off-by-one gaps.
2. **Claim accounting**: builds a `claims` dict mapping each cell to the list of region indices that selected it.
3. **Auto-split** (when `auto_split=True`): if a cell is claimed by *k* regions, it is divided into *k* equal-width vertical strips (one per claiming region, in region order). This ensures overlapping selections still produce distinct masks instead of identical ones.
4. **Fallback**: a region with no valid cells gets a full-canvas mask of all ones so it is never silently dropped.
5. **Feathering** (`_gaussian_blur`): a separable Gaussian convolution is applied to the `[1,1,H,W]` accumulator tensor. The kernel is built from `torch.exp`, the radius is clamped to half the smaller canvas dimension, and replication padding prevents edge darkening.

Output is a list of `[1,H,W]` float32 tensors (one per region) in latent space.

---

### Conditioning Pipeline

**File: `regional_multichar.py` → `RegionalMultiCharConditioning.build()`**

1. **Region list** (`_regions()`): the layout's `characters` + `links` are flattened into a list of `{cells, positive, negative}` dicts.
   - With `merge_linked=False` (default): one region per character, one region per link.
   - With `merge_linked=True`: a Union-Find (path-compressed) merges characters that share any link into one region; their cells are unioned and their prompts concatenated.
2. **Global base**: the global positive is encoded unconditionally (no mask). The global negative is encoded similarly if `regional_negatives=True`.
3. **Per-region encoding**: each region's positive is encoded and then masked via `_apply_mask()`, which calls `node_helpers.conditioning_set_values` to attach `{mask, set_area_to_bounds: False, mask_strength}`. This is the same mechanism as ComfyUI's built-in `ConditioningSetMask` node.
4. **Negative handling**:
   - `regional_negatives=False` (default): all per-character negatives are collected into a list and encoded as a single comma-joined string alongside the global negative.
   - `regional_negatives=True`: each character's negative is encoded and masked separately; the global negative is the unmasked base.
5. **Final conditioning lists** (`pos`, `neg`) are returned as standard ComfyUI CONDITIONING (a list of `[tensor, dict]` pairs).

---

### Composition Fractions

When `composition_frac < 1.0`:

- The existing `pos` conditioning list has `{start_percent: 0.0, end_percent: composition_frac}` attached to it.
- A second, unmasked encoding of all prompts merged into one sentence is created with `{start_percent: composition_frac, end_percent: 1.0}`.
- Both are concatenated: regional guidance drives the early denoising steps (structure / placement), the merged full-scene prompt drives the later steps (detail / coherence).

---

### Wording-Based Composition (no masks)

**File: `regional_multichar.py` → `assemble_multichar()`**

This is a pure Python string-assembly function (no tensors, no CLIP, no masks). The key stages:

1. **Handle derivation** (`_derive_handle`): each character gets a stable, natural referent ("the blonde woman") extracted from their name or positive. Duplicate handles are disambiguated with ordinals ("the first blonde woman", "the second blonde woman").
2. **Placement phrases** (`_loc_phrase`, `_scale_phrase`): cell indices → centroid fraction → spatial words ("in the upper left area", "large and prominent, close to the viewer").
3. **Negative pre-pass**: if `negative_mode=to_positive_assertion`, known antonyms are looked up in `_ANTONYMS` and moved into the character's positive additions ("old" → "young"); unknowns remain in the negative.
4. **Grouping + ordering** (`order_and_group`): characters sharing the same cell signature are grouped into one clause; groups are sorted left-to-right by centroid column fraction.
5. **Sentence rendering**: the selected `output_format` (prose / labeled / numbered) determines how each character or group is serialised into a sentence.
6. **Interactions**: if `bind_interactions`, action fragments are prepended with "The X and Y are ..."; otherwise used verbatim.
7. **Deduplication** (`_dedup_terms`): the final negative collects all per-character and global negatives, deduplicating by lowercased term.

The `flux2` profile adds count-lock checks, exact clause deduplication, interaction placement after the last member group, and optional negative term limits. Its default scale setting does not infer depth from grid row because a character higher in the image is not necessarily farther away. `spatial_cues=always` and `scale_cues=row_based` restore those cues when they fit a scene.

The report shows repeated clauses across blocks, profile adjustments, dropped negative terms, and rough length estimates. The token estimate uses character count divided by four; it is not the Flux tokenizer's actual count. A warning appears when that estimate passes 450 tokens so you can check the active encoder's context limit. The reference Klein pipeline uses a [512-token maximum by default](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py); your ComfyUI path may differ.

The output `{positive, negative, report}` dict is shared between the compose node and the live HTTP preview route.

---

### LLM Layout Enhancer

Calls `AutoTokenizer` / `AutoModelForCausalLM` from HuggingFace `transformers`. Before loading:
- `mm.unload_all_models()` + `mm.soft_empty_cache()` frees ComfyUI's VRAM.

After generation, `del mdl, tok` + `gc.collect()` + `torch.cuda.empty_cache()` frees it again. Any exception at any stage returns the original layout unchanged via a `try/finally`.

Two system prompts are used:
- **`SYS_C`** (character): rewrites one character's description for a manga/grayscale context with explicit pose wording.
- **`SYS_L`** (interaction/link): rewrites interaction text with concrete physical contact details.

`smart_wording=True` instructs the model to use only as many words as genuinely needed; `smart_wording=False` always aims for the `max_words` cap.

---

### Live Preview HTTP Route

**File: `regional_multichar.py` (bottom)**

At module load time a POST route `/multichar/preview` is registered on `PromptServer.instance`. The JavaScript editor POSTs `{layout, opts}` here; the server calls `assemble_multichar()` (the same function used during actual generation) and returns `{positive, negative, report}` as JSON. This guarantees the live preview is never stale relative to what the sampler will actually receive.

The registration is wrapped in a bare `try/except` so the module still loads in environments without a running ComfyUI server (e.g. unit tests).

---

### Frontend Editor

**File: `web/regional_multichar.js`**

Registered via `WEB_DIRECTORY = "./web"` in `__init__.py`. Three ComfyUI extensions are registered:

| Extension name | Node | What it does |
|---|---|---|
| `Regional.MultiChar` | `RegionalCharacterLayout` | Injects the visual character/interaction/grid editor as a DOM widget |
| `Regional.MultiCharPreview` | `MultiCharPromptPreview` | Shows exact positive and negative strings in separate copyable raw-text blocks, with optional Markdown rendering |
| `Regional.MultiCharComposeLive` | `MultiCharPromptCompose` | Adds the live preview, larger textareas, and a prompt settings JSON box; POSTs to `/multichar/preview` with a 3-second debounce |

The editor stores its state in `node.k2` (a `{structured_characters, characters, links}` object). Every interactive change calls `save(node)`, which serialises `node.k2` to the hidden `layout_json` widget, then calls `node.setDirtyCanvas()` to mark the graph dirty. The raw `layout_json` textarea is hidden after a successful first render; if JavaScript fails, it remains visible as a plain editable fallback so the node always works.

The grid cells inside each character card are drawn as CSS grid divs sized to the real image aspect ratio (not the grid's own aspect), so the editor preview matches what the image will look like.

---

## Typical Workflows

### SDXL / Pony / Illustrious — masked regional conditioning

```
RegionalCharacterLayout  →  layout, latent
                                ↓           ↓
        CLIP ──────────── RegionalMultiCharConditioning
                                ↓  positive / negative / latent
                              KSampler
                                ↓
                             VAEDecode → image
```

### Flux / Chroma / SD3 — wording-based conditioning

```
RegionalCharacterLayout  →  layout
                                ↓
        CLIP ─────────── MultiCharPromptCompose
                                ↓  positive / negative / positive_text
                              KSampler (Flux)
                                ↓
                             VAEDecode → image
```

For a black-and-white manga result with a traceable seed, connect `VAEDecode → Grayscale Filter → Seed Label → Save Image`. Feed the same `PrimitiveInt` output to KSampler and Seed Label. If you use FaceDetailer, put the filter and label after its final image output.

### With optional LLM enrichment

```
RegionalCharacterLayout → layout
                             ↓
                  MultiCharLayoutEnhancer (transformers LLM)
                             ↓ enriched layout
                  MultiCharPromptCompose
```

### Hires + FaceDetailer

```
KSampler → VAEDecode → base_image
                              ↓
                FaceDetailer → fd_image
                              ↓
           RegionalFaceDetailerSwitch (enable_facedetailer boolean)
                              ↓ final image

KSampler → base_latent
                ↓
      UpscaleLatent + KSampler2 → hires_latent
                ↓
     RegionalHiresSwitch (enable_hires boolean)
                ↓ final latent
```

---

## Tips & Gotchas

- **Cell 0 is top-left**, not bottom-left. Row index increases downward.
- **Empty cells list = full canvas**: a character with no cells selected is conditioned everywhere — useful for a global character style without spatial restriction.
- **`region_strength > 1.0`** can cause over-saturation of character features. Start at 1.0–1.5 and reduce if colours or anatomy break down.
- **`feather`** is in *pixel* space, not latent space. A value of 48 translates to 6 latent pixels (÷8). Increase to soften region edges; set to 0 for hard boundaries.
- **`composition_frac`** of 0.5–0.7 is a good balance: regions control structure in the first half of denoising, a merged prompt polishes detail in the second half. Set to 1.0 to keep regions all the way through.
- **`merge_linked`** reduces the number of encoding passes: two linked characters become one region and are encoded once instead of three times (A + B + link). Useful for closely interacting characters.
- **`auto_split`** only splits horizontally. If you need a vertical split (characters stacked), assign them to different rows.
- **Pasting a full-layout JSON** into the "Layout JSON" textarea and clicking "Apply" sets all widgets (aspect, grid, characters, interactions) in one shot — useful for programmatic workflows (e.g. Stansa.ai output).
- **Prompt cleanup is an experiment**: a shorter prompt can help, but exact clause removal cannot fix all contact geometry, and a term cap can remove a useful guard. Compare the same seeds and inspect the report before keeping a setting.
- **`layout` is optional** on `RegionalMultiCharConditioning`: if not connected, the node produces global conditioning from the global prompts alone (no regions).
- **LLM enhancer**: the model must be a HuggingFace directory with `config.json` and model weights under `models/LLM/`. Quantized GGUF files are not supported — use bfloat16 or float16 HF checkpoints.
