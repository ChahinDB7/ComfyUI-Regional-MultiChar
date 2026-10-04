# SKILL: ComfyUI Regional MultiChar — How It Works

> **Purpose:** This document explains the core concepts, data flow, and technical mechanics of the *ComfyUI Regional MultiChar* custom node pack. It is written for use **outside** the repository — for AI assistants, integrators, and advanced users who interact with the nodes from within a ComfyUI environment but do not have direct access to the source files.

---

## What the Pack Does

Regional MultiChar solves the core multi-character diffusion problem: getting each character to stay in their own region of the canvas without the conditioning for one character bleeding into another.

It works at two levels:

| Approach | Best model family | Core nodes |
|---|---|---|
| **Masked regional conditioning** | SDXL, SD1.5, Pony, Illustrious, NoobAI | `RegionalCharacterLayout` → `RegionalMultiCharConditioning` |
| **Wording-based composition** | Flux, Chroma, SD3 (strong text encoders, no cross-attention masks) | `RegionalCharacterLayout` → `MultiCharPromptCompose` |

Both approaches share the same visual grid editor on the `RegionalCharacterLayout` node.

---

## The Seven Nodes

### 1. `RegionalCharacterLayout` ("Regional Characters (grid layout)")

**What it is:** The editor node. Everything — canvas size, grid dimensions, character prompts, spatial placement, and character interactions — is configured here.

**What it produces:**
- A `REGIONAL_LAYOUT` bundle: a plain Python dict containing `{grid_cols, grid_rows, width, height, characters, links}`.
- An empty `LATENT` sized to the chosen aspect and batch size.

**How configuration is stored internally:**
All data lives in a hidden widget called `layout_json`. The visual editor (character cards, clickable grid cells, interaction cards) serialises its state as JSON into that widget on every change. The Python side reads only `layout_json` — the visual editor is purely for editing comfort. If the JavaScript editor fails to load, `layout_json` remains visible as a plain editable textarea, and the node still works.

**Layout JSON schema:**
```json
{
  "characters": [
    {
      "name": "label (optional)",
      "cells": [0, 3, 6],
      "positive": "character appearance and actions",
      "negative": "things to avoid for this character"
    }
  ],
  "links": [
    {
      "between": [1, 2],
      "positive": "interaction description",
      "negative": "what to avoid for this interaction"
    }
  ]
}
```
- `cells` are **zero-based flat indices**: `cell_index = row * grid_cols + col`. Cell 0 is top-left; indices increase left-to-right, top-to-bottom.
- `between` values are **1-based character indices** matching card order.
- A character with an empty `cells` list receives a full-canvas mask (conditioned everywhere).

A "full layout JSON" — adding `aspect`, `grid_cols`, `grid_rows`, `batch_size` to the above — can be pasted into the visible JSON textarea and applied in one shot. This is the programmatic interface for external tools (e.g. Stansa.ai).

---

### 2. `RegionalMultiCharConditioning` ("Regional Multi-Char Conditioning")

**What it is:** The mask-based conditioning node. Takes the layout bundle, a CLIP encoder, and a latent; produces masked CONDITIONING for a standard KSampler.

**Key parameters:**

| Parameter | Effect |
|---|---|
| `feather` | Gaussian blur radius in *pixel* space applied to each mask (divided by 8 for latent space). Softens region boundaries. |
| `region_strength` | `mask_strength` on each masked conditioning. Values > 1.0 amplify region signal; start at 1.0–1.5. |
| `global_positive` / `global_negative` | Unmasked baseline conditioning that covers the whole canvas. |
| `auto_split` | When two characters pick the same grid cell, split that cell into equal vertical strips (one per character) so they receive distinct masks. |
| `regional_negatives` | ON = encode each character's negative with its own mask (more precise, slower). OFF = collect all negatives into one global negative string (faster). |
| `merge_linked` | Merge characters that share a link into one region (union of cells, concatenated prompts). Reduces encoding passes for closely interacting characters. |
| `composition_frac` | Fraction of the denoising schedule that uses regional conditioning. After this point a single merged full-scene prompt takes over. 1.0 = regions all the way through. |

**Outputs:** `positive` (CONDITIONING), `negative` (CONDITIONING), `latent` (pass-through), `mask_preview` (IMAGE showing coloured region overlays, one hue per region).

---

### 3. `MultiCharPromptCompose` ("Multi-Char Prompt Compose (wording)")

**What it is:** A text-only alternative to masked conditioning. Assembles a single natural-language positive prompt from the layout, with no masks and no per-region encoding. Best for strong text encoders (Flux, Chroma, SD3).

**Key parameters:**

| Parameter | Effect |
|---|---|
| `subject_count_lock` | Adds "There are exactly N people and no one else" to prevent dropped or extra characters. |
| `use_names` | `handle` = stable referent like "the blonde woman" (best coherence); `label` = "Woman: ..." prefix; `off` = no prefix. |
| `bind_interactions` | Rewrites interaction text to name its characters: "The woman and the man are kissing..." (you type only "kissing..."). |
| `cast_roster` | Prepends "Cast: ..." so the model registers distinct people before detailed sentences. |
| `order_and_group` | Sorts characters left-to-right and merges same-cell characters into one clause. |
| `auto_scale_hints` | Top row = "small and distant", bottom row = "large and prominent". Needs a grid with more than one row. |
| `spatial_detail` | `fine` = "in the upper left area"; `coarse` = "on the left side"; `grid_coords` = fine wording + spreadsheet tags (A1, B2). |
| `output_format` | `prose` = flowing sentences; `labeled` = "Name (loc): ..."; `numbered` = "1) ...". |
| `auto_framing` | Derives shot type from character spread (wide/medium shot). |
| `negative_mode` | `global_dedup` = merge + deduplicate all negatives; `to_positive_assertion` = convert "old" → "young" directly in the positive. |

**Outputs:** `positive` (CONDITIONING), `negative` (CONDITIONING), `positive_text` (STRING), `negative_text` (STRING), `prompt_report` (STRING, structured markdown of every decision made).

**Live preview:** A preview panel inside the node POSTs the current layout and settings to `/multichar/preview` (a backend HTTP route provided by the pack) every 3 seconds after any edit, showing the assembled prompt without running the graph.

---

### 4. `MultiCharPromptPreview` ("Multi-Char Prompt Preview (read-only)")

A passthrough STRING node that renders its input as formatted markdown inside the node panel. Wire the `prompt_report` output from `MultiCharPromptCompose` into it to inspect the full assembler decision log.

---

### 5. `MultiCharLayoutEnhancer` ("Multi-Char Layout Enhancer (LLM, optional)")

Rewrites each character's `positive` and each interaction's `positive` in the layout using a local HuggingFace instruct LLM (placed under `models/LLM/`). The enriched layout is passed downstream to `MultiCharPromptCompose`, which handles structure (count-lock, placement, interaction binding); the enhancer only makes descriptions richer.

- `enable = OFF` is a passthrough — safe to leave in any graph.
- Frees ComfyUI's GPU models before loading the LLM; frees the LLM before returning.
- Any failure returns the original layout unchanged (never breaks a run).

---

### 6. `RegionalFaceDetailerSwitch` ("Regional FaceDetailer Toggle")

Selects between a base image and a FaceDetailer-processed image using a single boolean. Both inputs are **lazy**: when OFF, the FaceDetailer branch never executes.

---

### 7. `RegionalHiresSwitch` ("Regional Hires Toggle")

Same pattern for hires second-pass latents. When OFF, the upscale + second KSampler branch is completely skipped.

---

## Core Technical Concepts

### How Masks Are Built

Each region (character or link) produces a single float32 mask tensor. The process:

1. Each character's `cells` list is mapped to pixel-space bounding boxes using the formula `cell_index = row * cols + col`, with box edges computed as `round(row * height / rows)` etc.
2. If `auto_split` is on and multiple characters claim the same cell, that cell is divided into equal-width **vertical strips** — one per claiming character, in region order. This gives overlapping selections distinct mask regions instead of identical ones.
3. A character with no valid cells gets a full-canvas mask (all ones).
4. A separable Gaussian blur (built from `torch.exp`) is applied with `replication` padding to avoid edge darkening. The blur radius is in latent space (`feather // 8`).

### How Regional Conditioning Works

Each masked region feeds into ComfyUI's standard conditioning system via `ConditioningSetMask` semantics:
- The `mask` tensor and a `mask_strength` scalar are attached to the conditioning entry.
- The global (unmasked) positive serves as the baseline so uncovered pixels always receive coherent conditioning.
- Regional conditioning entries are *concatenated* into the `positive` and `negative` lists — the KSampler receives all of them together and applies each region's mask at attention time.

### How the Wording Assembler Works

`assemble_multichar()` is a pure Python string-assembly function. Given a layout and options, it:

1. **Derives a stable handle** for each character: a natural-language referent ("the blonde woman") extracted from their name or the first recognisable role/descriptor word in their positive prompt. Duplicate handles get ordinal disambiguation.
2. **Computes a grid centroid** for each character from their cells, then converts it to a spatial phrase ("in the upper left area") and optionally a scale phrase ("large and prominent, close to the viewer").
3. **Converts negatives** (optional): known antonyms are moved into the character's positive as counter-traits ("old" → "young").
4. **Groups and orders** co-located characters and sorts groups left-to-right.
5. **Serialises** each group using the chosen output format (prose / labeled / numbered).
6. **Binds interactions**: action fragments ("kissing each other") are prepended with named subjects ("The woman and the man are kissing each other.").
7. **Deduplicates negatives** across all characters and the global negative.

The same function is called during graph execution and by the live HTTP preview route — guaranteeing the preview never disagrees with what the sampler receives.

### Composition Fraction

When `composition_frac < 1.0` on `RegionalMultiCharConditioning`:
- Regional conditionings are assigned `start_percent=0.0, end_percent=composition_frac`.
- A second, unmasked encoding of all prompts merged into one string is assigned `start_percent=composition_frac, end_percent=1.0`.
- Both lists are concatenated. The KSampler's scheduler selects the correct conditioning entry at each step.
- Effect: regional guidance controls *structure and placement* in the early steps; the merged prompt polishes *detail and coherence* in the later steps.

### Merge Linked

When `merge_linked=True` on `RegionalMultiCharConditioning`:
- A Union-Find algorithm (path-compressed) groups characters that share any link.
- All characters in the same group become a single region: cells are unioned, prompts are comma-concatenated.
- Links belonging to the group are also folded in.
- Result: instead of 3 encoding passes (character A, character B, link A-B), there is 1 pass for the merged region.

---

## Workflow Patterns

### Pattern 1: SDXL / Pony masked regional conditioning

```
RegionalCharacterLayout → layout + latent
         ↓                      ↓
CLIP → RegionalMultiCharConditioning
         ↓ positive  ↓ negative  ↓ latent
                   KSampler
                      ↓
                  VAEDecode → image
```

### Pattern 2: Flux / Chroma wording-based conditioning

```
RegionalCharacterLayout → layout
         ↓
CLIP → MultiCharPromptCompose
         ↓ positive  ↓ negative
               KSampler (Flux)
                  ↓
              VAEDecode → image
```

### Pattern 3: With LLM enrichment

```
RegionalCharacterLayout → layout
         ↓
MultiCharLayoutEnhancer → enriched layout
         ↓
MultiCharPromptCompose
```

### Pattern 4: FaceDetailer and hires as optional passes

```
KSampler → VAEDecode → base_image ─────────────────────────┐
                           ↓                                 │
              FaceDetailer → fd_image                        │
                           ↓                                 │
              RegionalFaceDetailerSwitch ← enable_facedetailer
                           ↓ final image

KSampler → base_latent ─────────────────────────────────────┐
                ↓                                            │
    UpscaleLatent + KSampler2 → hires_latent                │
                ↓                                            │
    RegionalHiresSwitch ← enable_hires                      │
                ↓ final latent
```

---

## Parameter Cheat-Sheet

### `RegionalMultiCharConditioning`

| Goal | Recommended setting |
|---|---|
| Sharp, distinct regions | `feather=0`, `auto_split=ON` |
| Soft region blending | `feather=48–96` |
| Strong character identity | `region_strength=1.5–2.0` |
| Reduce per-region encoding cost | `merge_linked=ON`, `regional_negatives=OFF` |
| Fast: structure first, detail later | `composition_frac=0.5–0.7` |

### `MultiCharPromptCompose`

| Goal | Recommended setting |
|---|---|
| Prevent dropped characters | `subject_count_lock=ON`, `cast_roster=ON` |
| Best coherence | `use_names=handle`, `bind_interactions=ON`, `order_and_group=ON` |
| Readability | `use_names=label`, `output_format=labeled` |
| Depth perception | `auto_scale_hints=ON` (requires rows > 1) |
| Avoid color words in negatives | `negative_mode=to_positive_assertion` |

---

## Common Mistakes

- **Cell indices are zero-based and row-major.** Cell 0 = top-left, not bottom-left. Row index increases *downward*.
- **Auto-split is horizontal only.** Two characters sharing a cell are split into left/right strips, not top/bottom. For vertical separation, assign them to different rows.
- **Feather is in pixel space.** A value of 48 becomes 6 latent pixels (48 ÷ 8). For a tight mask on a 1024 × 1024 canvas, 24–48 is typical.
- **`layout` is optional** on both conditioning nodes. If not connected, `RegionalMultiCharConditioning` produces global conditioning from the global prompts alone (no regions), and `MultiCharPromptCompose` produces an empty scene.
- **`region_strength` > 3.0** often causes colour or anatomy artefacts. Prefer 1.0–2.0.
- **Wording-based composition does not use masks.** The `MultiCharPromptCompose` node produces a single flat text prompt. Spatial wording (left, upper, far background) is all that guides placement — it relies entirely on the text encoder understanding it. This works well on Flux/Chroma/SD3 but not on SDXL.
- **LLM enhancer needs an HF directory** (with `config.json` and model weights), not a GGUF file.
- **`composition_frac=1.0`** (the default) keeps regions all the way through denoising. Lower values (0.5–0.7) reduce artefacts when characters strongly overlap in some regions.
- **`merge_linked=ON`** makes sense only when characters genuinely interact and their masks should overlap. For separate characters it reduces the number of distinct regions (bad).
- **Empty `between` list** on a link means the link region will have no cells (full-canvas mask). Always set `between` chips before writing an interaction positive.
