"""Regional multi-character conditioning for ANY SDXL/SD1.5 checkpoint.

Model-agnostic fork of the Krea-2 regional pack: same grid editor + per-character
+ interaction prompts, but it outputs plain masked CONDITIONING that feeds a
standard KSampler (Pony, Illustrious, NoobAI, SD1.5, ...). No Krea-2 / RES4LYF /
ClownsharKSampler dependency.

What it adds over plain regional masking:
- merge_linked: characters joined by a link become one region (one pass, not N).
- composition_frac: run the regions only for the first fraction of steps, then a
  single merged prompt for the rest (fast, avoids per-region cost every step).
- auto_split: when two characters pick the SAME grid cell, that cell is split into
  vertical strips (one per character) so they still get distinct masks instead of
  identical ones that fight. Grid cells are a suggestion, not a hard box.
"""
import json

import torch
import torch.nn.functional as F

import node_helpers


# A default scene so a freshly-dropped node already shows how the editor works.
DEFAULT_LAYOUT = {
    "characters": [
        {"name": "A", "cells": [0, 3, 6], "positive": "1girl, solo focus", "negative": ""},
        {"name": "B", "cells": [2, 5, 8], "positive": "1boy, solo focus", "negative": ""},
    ],
    "links": [],
}

# Distinct overlay colors for the mask preview (RGB, 0-255).
_PALETTE = [
    (231, 76, 60), (46, 204, 113), (52, 152, 219), (241, 196, 15),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (149, 165, 166),
]


def _parse_layout(raw):
    """Accept the JSON string from the widget, fall back to the default."""
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(raw) if raw and raw.strip() else DEFAULT_LAYOUT
        except (ValueError, TypeError):
            data = DEFAULT_LAYOUT
    chars = data.get("characters") or []
    links = data.get("links") or []
    return chars, links


def _gaussian_blur(mask, radius):
    """Separable gaussian on a [1,1,H,W] tensor. radius is in pixels."""
    radius = int(radius)
    if radius <= 0:
        return mask
    radius = max(1, min(radius, max(1, min(mask.shape[-1], mask.shape[-2]) // 2)))
    k = radius * 2 + 1
    sigma = max(radius / 2.0, 1e-3)
    coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kx = g.view(1, 1, 1, k)
    ky = g.view(1, 1, k, 1)
    m = F.pad(mask, (radius, radius, 0, 0), mode="replicate")
    m = F.conv2d(m, kx)
    m = F.pad(m, (0, 0, radius, radius), mode="replicate")
    m = F.conv2d(m, ky)
    return m.clamp(0.0, 1.0)


def _cell_box(idx, cols, rows, height, width):
    r, c = idx // cols, idx % cols
    y0 = int(round(r * height / rows))
    y1 = int(round((r + 1) * height / rows))
    x0 = int(round(c * width / cols))
    x1 = int(round((c + 1) * width / cols))
    return y0, y1, x0, x1


def _build_region_masks(region_cells, cols, rows, height, width, feather, auto_split=True):
    """One feathered [1,H,W] mask per region, with shared cells auto-split.

    region_cells is a list (one entry per region) of grid-cell index lists. If a
    cell is claimed by more than one region and auto_split is on, that cell is
    divided into equal vertical strips - one per claiming region, in region order -
    so overlapping selections still produce distinct masks. A region that selected
    nothing gets a full-canvas mask (conditioned everywhere) so it never vanishes.
    """
    n = len(region_cells)
    total = cols * rows
    accum = [torch.zeros((1, 1, height, width), dtype=torch.float32) for _ in range(n)]
    valid_cells = []
    has_any = []
    for cells in region_cells:
        v = [c for c in cells if isinstance(c, int) and 0 <= c < total]
        valid_cells.append(v)
        has_any.append(bool(v))

    # cell -> list of region indices that selected it, in region order
    claims = {}
    for ri, cells in enumerate(valid_cells):
        for c in cells:
            claims.setdefault(c, []).append(ri)

    for cell, ris in claims.items():
        y0, y1, x0, x1 = _cell_box(cell, cols, rows, height, width)
        if not auto_split or len(ris) == 1:
            for ri in ris:
                accum[ri][0, 0, y0:y1, x0:x1] = 1.0
        else:
            k = len(ris)
            for j, ri in enumerate(ris):
                sx0 = int(round(x0 + (x1 - x0) * j / k))
                sx1 = int(round(x0 + (x1 - x0) * (j + 1) / k))
                accum[ri][0, 0, y0:y1, sx0:sx1] = 1.0

    out = []
    for ri in range(n):
        if not has_any[ri]:
            out.append(torch.ones((1, height, width), dtype=torch.float32))
        else:
            out.append(_gaussian_blur(accum[ri], feather)[0])
    return out


def _apply_mask(cond, mask, strength):
    """Same effect as core ConditioningSetMask (set_cond_area='default')."""
    if len(mask.shape) < 3:
        mask = mask.unsqueeze(0)
    return node_helpers.conditioning_set_values(
        cond,
        {"mask": mask, "set_area_to_bounds": False, "mask_strength": float(strength)},
    )


def _render_preview_from_masks(masks, labels, cols, rows, height, width):
    """Overview image built straight from the real masks, so the preview shows
    exactly where each region lands (auto-split strips included)."""
    scale = min(1.0, 768.0 / max(height, width))
    hp = max(64, int(height * scale))
    wp = max(64, int(width * scale))
    try:
        from PIL import Image, ImageDraw
        import numpy as np

        base = Image.new("RGBA", (wp, hp), (26, 27, 32, 255))
        for i, m in enumerate(masks):
            color = _PALETTE[i % len(_PALETTE)]
            md = F.interpolate(m.unsqueeze(0), size=(hp, wp), mode="bilinear", align_corners=False)[0, 0]
            a = (md.clamp(0.0, 1.0).numpy() * 110.0).astype("uint8")
            overlay = np.zeros((hp, wp, 4), dtype="uint8")
            overlay[..., 0] = color[0]
            overlay[..., 1] = color[1]
            overlay[..., 2] = color[2]
            overlay[..., 3] = a
            base = Image.alpha_composite(base, Image.fromarray(overlay, "RGBA"))

        draw = ImageDraw.Draw(base, "RGBA")
        for c in range(1, cols):
            x = c * wp / cols
            draw.line([(x, 0), (x, hp)], fill=(255, 255, 255, 40), width=1)
        for r in range(1, rows):
            y = r * hp / rows
            draw.line([(0, y), (wp, y)], fill=(255, 255, 255, 40), width=1)
        for i, m in enumerate(masks):
            md = F.interpolate(m.unsqueeze(0), size=(hp, wp), mode="bilinear", align_corners=False)[0, 0]
            ys, xs = np.where(md.numpy() > 0.4)
            if len(xs):
                draw.text((float(xs.mean()) - 3, float(ys.mean()) - 6), str(labels[i]), fill=(255, 255, 255, 255))

        arr = np.asarray(base.convert("RGB")).astype("float32") / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W,3]
    except Exception:
        return torch.full((1, hp, wp, 3), 0.1, dtype=torch.float32)


class RegionalCharacterLayout:
    """Grid + character/interaction editor. Pick an aspect, place characters on a
    grid (each with its own positive/negative), link two for an interaction.
    Outputs the layout bundle plus a matching empty SDXL latent."""

    ASPECTS = {
        "wide 1536x832": (1536, 832),
        "rectangle 1216x832": (1216, 832),
        "square 1024x1024": (1024, 1024),
        "portrait 832x1216": (832, 1216),
        "tall 832x1536": (832, 1536),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect": (list(cls.ASPECTS.keys()), {"default": "rectangle 1216x832"}),
                "grid_cols": ("INT", {"default": 3, "min": 1, "max": 8}),
                "grid_rows": ("INT", {"default": 3, "min": 1, "max": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "layout_json": ("STRING", {"multiline": True, "default": json.dumps(DEFAULT_LAYOUT)}),
            }
        }

    RETURN_TYPES = ("REGIONAL_LAYOUT", "LATENT")
    RETURN_NAMES = ("layout", "latent")
    FUNCTION = "make"
    CATEGORY = "Regional/conditioning"
    DESCRIPTION = (
        "Pick an aspect ratio and place characters on a grid, each with its own "
        "positive/negative prompt; link two for interaction. Outputs the layout "
        "plus a matching empty latent. Feed both into Regional Conditioning."
    )

    def make(self, aspect, grid_cols, grid_rows, batch_size, layout_json):
        import comfy.model_management as mm
        w, h = self.ASPECTS.get(aspect, (1216, 832))
        chars, links = _parse_layout(layout_json)
        latent = {
            "samples": torch.zeros(
                [batch_size, 4, h // 8, w // 8],
                device=mm.intermediate_device(), dtype=mm.intermediate_dtype()),
        }
        bundle = {"grid_cols": grid_cols, "grid_rows": grid_rows,
                  "width": w, "height": h, "characters": chars, "links": links}
        return (bundle, latent)


class RegionalMultiCharConditioning:
    """Turn a RegionalCharacterLayout into masked regional CONDITIONING for a
    standard SDXL/SD1.5 KSampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "latent": ("LATENT",),
                "feather": ("INT", {"default": 48, "min": 0, "max": 512}),
                "region_strength": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.05}),
                "global_positive": ("STRING", {"multiline": True, "default": ""}),
                "global_negative": ("STRING", {"multiline": True, "default": ""}),
                "auto_split": ("BOOLEAN", {"default": True, "label_on": "auto-split shared cells", "label_off": "raw masks"}),
                "regional_negatives": ("BOOLEAN", {"default": False, "label_on": "per-region (slower)", "label_off": "one global (faster)"}),
                "merge_linked": ("BOOLEAN", {"default": False, "label_on": "merge linked chars", "label_off": "separate regions"}),
                "composition_frac": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "layout": ("REGIONAL_LAYOUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "IMAGE")
    RETURN_NAMES = ("positive", "negative", "latent", "mask_preview")
    FUNCTION = "build"
    CATEGORY = "Regional/conditioning"
    DESCRIPTION = (
        "Encodes the global prompt plus each character/link region from a "
        "connected layout, masks them to their grid regions, and outputs combined "
        "positive/negative CONDITIONING for a standard KSampler. auto_split keeps "
        "two characters that pick the same cell apart; merge_linked and "
        "composition_frac cut the per-region cost."
    )

    def _encode(self, clip, text):
        return clip.encode_from_tokens_scheduled(clip.tokenize(text or ""))

    def _regions(self, chars, links, merge_linked):
        """Flatten the layout into a list of {cells, positive, negative} regions."""
        chars = chars or []
        links = links or []
        if not merge_linked:
            regions = [{"cells": c.get("cells", []),
                        "positive": (c.get("positive") or "").strip(),
                        "negative": (c.get("negative") or "").strip()} for c in chars]
            for ln in links:
                union = set()
                for b in ln.get("between", []):
                    if 0 <= b - 1 < len(chars):
                        union.update(chars[b - 1].get("cells", []))
                regions.append({"cells": sorted(union),
                                "positive": (ln.get("positive") or "").strip(),
                                "negative": (ln.get("negative") or "").strip()})
            return regions

        parent = list(range(len(chars)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for ln in links:
            members = [b - 1 for b in ln.get("between", []) if 0 <= b - 1 < len(chars)]
            for m in members[1:]:
                parent[find(m)] = find(members[0])
        comp_links = {}
        for ln in links:
            members = [b - 1 for b in ln.get("between", []) if 0 <= b - 1 < len(chars)]
            if members:
                comp_links.setdefault(find(members[0]), []).append(ln)
        comps = {}
        for i in range(len(chars)):
            comps.setdefault(find(i), []).append(i)

        regions = []
        for root, idxs in comps.items():
            cells = sorted(set().union(*[set(chars[i].get("cells", [])) for i in idxs])) if idxs else []
            tp = [chars[i].get("positive") or "" for i in idxs] + [l.get("positive") or "" for l in comp_links.get(root, [])]
            tn = [chars[i].get("negative") or "" for i in idxs] + [l.get("negative") or "" for l in comp_links.get(root, [])]
            regions.append({"cells": cells,
                            "positive": ", ".join(t.strip() for t in tp if t.strip()),
                            "negative": ", ".join(t.strip() for t in tn if t.strip())})
        return regions

    def build(self, clip, latent, feather, region_strength, global_positive, global_negative,
              auto_split=True, regional_negatives=False, merge_linked=False,
              composition_frac=1.0, layout=None):
        if layout:
            grid_cols = int(layout.get("grid_cols", 3) or 3)
            grid_rows = int(layout.get("grid_rows", 3) or 3)
            chars = layout.get("characters") or []
            links = layout.get("links") or []
        else:
            grid_cols, grid_rows, chars, links = 3, 3, [], []

        samples = latent["samples"]
        lat_h, lat_w = int(samples.shape[-2]), int(samples.shape[-1])
        height, width = lat_h, lat_w
        mask_feather = max(0, feather // 8)
        img_h, img_w = lat_h * 8, lat_w * 8

        regions = self._regions(chars, links, merge_linked)
        masks = _build_region_masks([r["cells"] for r in regions],
                                    grid_cols, grid_rows, height, width, mask_feather, auto_split)

        pos, neg = [], []
        folded_neg = [global_negative or ""]

        # full-canvas positive base so any uncovered cell still gets coherent
        # conditioning; the negative base is added only when negatives are regional.
        pos += self._encode(clip, global_positive or "")
        if regional_negatives:
            neg += self._encode(clip, global_negative or "")

        for i, r in enumerate(regions):
            if r["positive"]:
                pos += _apply_mask(self._encode(clip, r["positive"]), masks[i], region_strength)
            if r["negative"]:
                if regional_negatives:
                    neg.extend(_apply_mask(self._encode(clip, r["negative"]), masks[i], region_strength))
                else:
                    folded_neg.append(r["negative"])

        if not regional_negatives:
            neg = self._encode(clip, ", ".join(t for t in folded_neg if t and t.strip()))

        # Composition phase: regions for the first composition_frac of the schedule,
        # then one merged full prompt for the detail steps. 1.0 = regions all the way.
        if composition_frac < 1.0:
            pos = node_helpers.conditioning_set_values(
                pos, {"start_percent": 0.0, "end_percent": composition_frac})
            merged_text = ", ".join(
                t for t in ([global_positive or ""] + [r["positive"] for r in regions])
                if t and t.strip())
            late = node_helpers.conditioning_set_values(
                self._encode(clip, merged_text),
                {"start_percent": composition_frac, "end_percent": 1.0})
            pos = pos + late

        labels = list(range(1, len(regions) + 1))
        preview = _render_preview_from_masks(masks, labels, grid_cols, grid_rows, img_h, img_w)
        return (pos, neg, latent, preview)


class RegionalFaceDetailerSwitch:
    """One boolean to turn the FaceDetailer pass on/off. Both image inputs are
    lazy, so when OFF the FaceDetailer branch never runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_facedetailer": ("BOOLEAN", {"default": False, "label_on": "FaceDetailer ON", "label_off": "FaceDetailer OFF"}),
                "base": ("IMAGE", {"lazy": True}),
                "with_facedetailer": ("IMAGE", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pick"
    CATEGORY = "Regional/conditioning"
    DESCRIPTION = "Toggle the FaceDetailer pass with one boolean; the branch is skipped when off."

    def check_lazy_status(self, enable_facedetailer, base=None, with_facedetailer=None):
        if enable_facedetailer and with_facedetailer is None:
            return ["with_facedetailer"]
        if (not enable_facedetailer) and base is None:
            return ["base"]
        return []

    def pick(self, enable_facedetailer, base=None, with_facedetailer=None):
        return (with_facedetailer if enable_facedetailer else base,)


class RegionalHiresSwitch:
    """One boolean to turn the hires second pass on/off (latent domain). Both
    inputs are lazy, so when OFF the upscale + second sampler never run."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_hires": ("BOOLEAN", {"default": False, "label_on": "hires ON", "label_off": "hires OFF"}),
                "base": ("LATENT", {"lazy": True}),
                "with_hires": ("LATENT", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "pick"
    CATEGORY = "Regional/conditioning"
    DESCRIPTION = "Toggle the hires second pass with one boolean; the upscale + 2nd sampler are skipped when off."

    def check_lazy_status(self, enable_hires, base=None, with_hires=None):
        if enable_hires and with_hires is None:
            return ["with_hires"]
        if (not enable_hires) and base is None:
            return ["base"]
        return []

    def pick(self, enable_hires, base=None, with_hires=None):
        return (with_hires if enable_hires else base,)


class MultiCharPromptCompose:
    """Structured multi-character PROMPT composer - the breakdown editor WITHOUT masks.

    Reuses the same RegionalCharacterLayout editor (per-character cards + interaction
    links + a grid for rough placement), but instead of masking it converts each
    character's grid position into LOCATION WORDING ("on the left side", "on the right
    side", ...) and assembles ONE natural-language positive + negative. Feed it to a
    strong text-encoder model (Chroma / Flux / SD3) which follows positional wording.

    Outputs CONDITIONING (positive, negative) ready for a KSampler, plus the assembled
    text (so you can see/copy exactly what was built). No latent, no masks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "global_positive": ("STRING", {"multiline": True, "default": ""}),
                "global_negative": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {"layout": ("REGIONAL_LAYOUT",)},
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "positive_text", "negative_text")
    FUNCTION = "compose"
    CATEGORY = "Regional/conditioning"
    DESCRIPTION = (
        "Multi-character breakdown by WORDING (not masks). Takes the character/interaction "
        "layout from Regional Characters, turns each character's grid spot into a location "
        "phrase, and assembles one positive + negative prompt encoded for a KSampler. Best "
        "on text-encoder-strong models (Chroma/Flux/SD3)."
    )

    def _location(self, cells, cols, rows):
        cells = [c for c in cells if isinstance(c, int) and 0 <= c < cols * rows]
        if not cells:
            return ""
        cf = (sum(c % cols for c in cells) / len(cells)) / (cols - 1) if cols > 1 else 0.5
        rf = (sum(c // cols for c in cells) / len(cells)) / (rows - 1) if rows > 1 else 0.5
        col = ("on the left side of the scene" if cf < 0.34
               else "in the center of the scene" if cf <= 0.66
               else "on the right side of the scene")
        row = (" in the foreground" if rf > 0.66 else " in the far background" if rf < 0.34 else "")
        return col + row

    def compose(self, clip, global_positive, global_negative, layout=None):
        cols = int(layout.get("grid_cols", 3) or 3) if layout else 3
        rows = int(layout.get("grid_rows", 1) or 1) if layout else 1
        chars = (layout.get("characters") if layout else []) or []
        links = (layout.get("links") if layout else []) or []

        pos_parts = [global_positive.strip()] if global_positive.strip() else []
        neg_parts = [global_negative.strip()] if global_negative.strip() else []
        for c in chars:
            p = (c.get("positive") or "").strip()
            if p:
                loc = self._location(c.get("cells", []), cols, rows)
                pos_parts.append(f"{loc.capitalize()}, {p}." if loc else p + ".")
            n = (c.get("negative") or "").strip()
            if n:
                neg_parts.append(n)
        for ln in links:
            p = (ln.get("positive") or "").strip()
            if p:
                pos_parts.append(p + ".")
            n = (ln.get("negative") or "").strip()
            if n:
                neg_parts.append(n)

        pos_text = " ".join(pos_parts)
        neg_text = ", ".join(neg_parts)
        pos = clip.encode_from_tokens_scheduled(clip.tokenize(pos_text))
        neg = clip.encode_from_tokens_scheduled(clip.tokenize(neg_text))
        return (pos, neg, pos_text, neg_text)


NODE_CLASS_MAPPINGS = {
    "RegionalCharacterLayout": RegionalCharacterLayout,
    "RegionalMultiCharConditioning": RegionalMultiCharConditioning,
    "MultiCharPromptCompose": MultiCharPromptCompose,
    "RegionalFaceDetailerSwitch": RegionalFaceDetailerSwitch,
    "RegionalHiresSwitch": RegionalHiresSwitch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionalCharacterLayout": "Regional Characters (grid layout)",
    "RegionalMultiCharConditioning": "Regional Multi-Char Conditioning",
    "MultiCharPromptCompose": "Multi-Char Prompt Compose (wording)",
    "RegionalFaceDetailerSwitch": "Regional FaceDetailer Toggle",
    "RegionalHiresSwitch": "Regional Hires Toggle",
}
