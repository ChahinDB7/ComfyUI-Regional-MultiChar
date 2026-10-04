# SKILL: ComfyUI Regional MultiChar — How It Works

> **Purpose:** This document explains the core concepts, data flow, and technical mechanics of the *ComfyUI Regional MultiChar* custom node pack. It is written for use **outside** the repository — for AI assistants, integrators, and advanced users who interact with the nodes from within a ComfyUI environment but do not have direct access to the source files.

When using this as `comfyui-multi-char.md` in another assistant's skill, load it before writing or revising a Regional MultiChar layout or prompt. It describes the JSON schema, the optional Flux 2 profile, and how to inspect the exact encoder text. Keep character appearance rules and scene-specific lessons in their own support files.

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

## The Eight Nodes

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
  "structured_characters": true,
  "characters": [
    {
      "name": "label (optional)",
      "cells": [0, 3, 6],
      "generic_positive": "general character prompt",
      "looks_positive": "stable appearance and outfit",
      "pose_action_positive": "current pose and action",
      "generic_negative": "general things to avoid",
      "looks_negative": "appearance mistakes to avoid",
      "pose_action_negative": "pose mistakes to avoid"
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

The editor's **Structured character prompts** checkbox is on by default. It applies only to character cards; interactions always use `positive` and `negative`. Character prompt fields combine in this order: Generic, Looks, Pose / action. Put stable identity details in Looks and scene-specific body wording in Pose / action. An old layout with character `positive` and `negative` and no checkbox setting is loaded into `generic_positive` and `generic_negative`; its assembled prompt stays the same. Set `structured_characters` to `false` for the legacy two-field character format. Switching off merges the three categories into `positive` and `negative`; switching on again puts the merged text into Generic, so save a JSON copy if you need the categories later. The runtime `REGIONAL_LAYOUT` bundle provides combined `positive` and `negative` aliases for structured characters so existing consumers can still read them. In pasted JSON, use one character format at a time. If both are present, a nonempty legacy `positive` or `negative` wins for that side.

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
| `global_positive` / `global_negative` | Scene-wide positive text and terms to avoid everywhere. Both appear in the composer's settings JSON. |
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
| `prompt_profile` | `default` preserves the old positive and negative strings. `flux2` removes exact repeated interaction clauses, skips redundant count/location cues, and places an interaction after its last member group. |
| `negative_term_cap` | `0` keeps all deduplicated negatives. A positive value keeps only the first N after priority ordering; inspect the report for dropped terms. |
| `negative_cap_priority` | `global_first` keeps global, character, then interaction terms. `interaction_first` gives interaction guards priority when a cap is active in `flux2`. |
| `spatial_cues` | In `flux2`: `auto` skips cues for a 1x1 grid or a fully shared cell selection; `always` forces them; `off` suppresses them. |
| `scale_cues` | In `flux2`: `off` avoids treating grid row as depth; `row_based` restores the old top=far/bottom=near wording. |
| `layout_json_override` | Legacy layout override retained for saved workflows. The composer hides it in new workflows and shows a Clear button if an old override is active. |

**Outputs:** `positive` (CONDITIONING), `negative` (CONDITIONING), `positive_text` (STRING), `negative_text` (STRING), `prompt_report` (STRING, structured markdown of every decision made).

**Live preview:** A preview panel inside the node POSTs the current layout and settings to `/multichar/preview` (a backend HTTP route provided by the pack) every 3 seconds after any edit, showing the assembled prompt without running the graph. The report lists duplicate clauses, profile adjustments, dropped negative terms, and rough length estimates. Its token estimate is only a character-based approximation.

**Prompt settings copy/paste:** The compose node's **Prompt settings JSON (edit / paste -> Apply)** box contains `global_positive`, `global_negative`, and all prompt assembly settings listed above. Copy it to revise the node elsewhere, then paste and click **Apply settings JSON**. A partial object changes only its listed keys. A workflow's `widgets_values_named` object can also be pasted; an empty legacy `layout_json_override` value is accepted to clear it. The box does not contain `grid_cols`, `grid_rows`, `characters`, `links`, `aspect`, or `batch_size`. Those belong in the **Layout JSON** box on `RegionalCharacterLayout`. Older saved composer layout overrides still work, and an active one is flagged with a **Clear old override** button.

---

### 4. `MultiCharPromptPreview` ("Multi-Char Prompt Preview (read-only)")

Connect `MultiCharPromptCompose.prompt_report` to the preview's required `text` input. On execution, it extracts the final positive and negative from the report and shows the exact raw strings in separate blocks. Each block has Copy and a Raw/Markdown view toggle. Blank encoder strings stay blank; the displayed values do not include `(empty)` placeholders.

For prompts containing Markdown code fences, also connect `MultiCharPromptCompose.positive_text` and `.negative_text` to the preview's optional inputs of the same names. Those direct connections take priority over parsing the report and are the most reliable way to inspect exactly what `clip.tokenize()` received. The required `text` input remains at the same index, and the `text` output still passes its input through. Existing workflows that only connect `prompt_report` continue to work. Unrelated Markdown sent to `text` still renders as Markdown.

---

### 5. `MultiCharLayoutEnhancer` ("Multi-Char Layout Enhancer (LLM, optional)")

Rewrites each character's combined positive text and each interaction's `positive` in the layout using a local HuggingFace instruct LLM (placed under `models/LLM/`). For a structured character, the enhancer writes the rewritten text to a runtime `positive` override; the upstream editor's six fields stay unchanged. The enriched layout is passed downstream to `MultiCharPromptCompose`, which handles structure (count-lock, placement, interaction binding).

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

### 8. `RegionalGrayscaleFilter` ("Grayscale Filter (optional)")

Takes a decoded `IMAGE` and outputs a grayscale `IMAGE` when `enabled` is on (the default). It uses weighted RGB luminance, returns three equal RGB channels, and keeps an existing alpha channel. Inputs with fewer than three channels pass through unchanged. With `enabled` off, it passes the input through unchanged. Connect it after VAEDecode and after FaceDetailer or any other image pass that could add color, then send its output to Save Image. This is a post-processing filter; it does not change prompts, conditioning, or sampling.

Optional controls appear below `enabled`: `grayscale_strength` (`0` to `1`, default `1`) mixes original color back in when below `1`; `brightness` (`-0.5` to `0.5`, default `0`) shifts the gray level; `contrast` (`0` to `2`, default `1`) changes tonal separation; and `black_lift` (`0` to `1`, default `0`) raises black toward gray while keeping white white. Tone adjustments run before the strength mix. For strict monochrome, keep strength at `1`. For a lighter manga print, start with `black_lift=0.05` to `0.15`. Saved workflows without these optional fields keep the previous output.

---

### 9. `RegionalSeedLabel` ("Seed Label (optional)")

Writes the numeric `seed` input on each decoded `IMAGE` in the batch. Put it after `RegionalGrayscaleFilter` and any image enhancement pass, before Save Image. `enabled=false` passes the image through unchanged. The default label is `Seed: <number>` at the bottom right, in white with a black shadow. `position` supports each corner and `custom`; the latter uses `x_percent` and `y_percent`. `font` offers sans, mono, serif, and Pillow default; `font_file` accepts a local `.ttf` or `.otf` path. `font_size`, `margin`, `text_color`, `shadow`, `shadow_color`, `shadow_offset`, and `prefix` are adjustable.

`optimize=none` uses the chosen text and shadow colors. `optimize=grayscale` forces white text with a black outline for light and dark manga areas. `optimize=photo_realistic` uses white text on a translucent dark plate. The preset styles the text only; it does not convert the image to grayscale.

For exact traceability, add ComfyUI's built-in `PrimitiveInt`, set its control after generate to `fixed`, and connect its single `INT` output to both `KSampler.seed` and `RegionalSeedLabel.seed`. Do not keep independent seed values in two widgets. KSampler does not output its actual seed. This post-processing node has no effect on sampling or prompt assembly, and saved graphs without it are unchanged.

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
7. **Deduplicates negatives** across the global, character, and interaction blocks.

In `flux2`, it also skips location cues when the grid gives no relative placement, avoids repeating a matching headcount, removes exact duplicate comma clauses from interactions, and puts each interaction after the last group containing a member. Row-based scale wording is off unless `scale_cues=row_based` because vertical position does not always imply depth. The output report contains the exact final positive and negative strings in separate fenced sections.

The same assembler runs during graph execution and in the live HTTP preview route. The live preview reads the connected editor layout; if another node dynamically changes that layout during execution, use the graph-run preview with direct text connections to inspect the encoder strings.

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

For monochrome manga output with a visible seed, append `RegionalGrayscaleFilter → RegionalSeedLabel → Save Image` after the final decoded or FaceDetailer image. Keep the filter enabled to remove stray color. Feed both the sampler and the label from one fixed `PrimitiveInt` seed source.

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
| Row-based depth hint | In `default`, `auto_scale_hints=ON`; in `flux2`, also set `scale_cues=row_based` when higher rows really are farther away |
| Avoid color words in negatives | `negative_mode=to_positive_assertion` |
| Try concise Flux 2 wording | `prompt_profile=flux2`, `negative_term_cap=0` first; compare the same seeds |
| Prioritize interaction guards under a cap | `prompt_profile=flux2`, `negative_cap_priority=interaction_first` |
| Keep explicit grid wording in Flux 2 | `spatial_cues=always` |

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
- **A negative cap is not a guaranteed fix.** Check the sampler: standard ComfyUI sampling at CFG 1 skips the unconditional pass, so the negative text may have no image effect. Inspect the report before capping terms, and test the same seeds before keeping `flux2` settings.
