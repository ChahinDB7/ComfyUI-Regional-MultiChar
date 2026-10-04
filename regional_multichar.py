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
import re

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


# ---------------------------------------------------------------------------
# Wording-based multi-character composer (no masks) + a read-only prompt viewer.
# The composer mirrors the RegionalCharacterLayout editor fields in plain text
# with several optional structuring aids that help a strong text encoder
# (Chroma / Flux / SD3) understand the scene. See
# docs/MULTICHAR_PROMPT_COMPOSE_IMPROVEMENTS.md. With every aid off (count off,
# names=off, bind off, roster off, order_group off, scale off, spatial=coarse,
# format=prose, framing off, negatives=global_dedup) it reproduces the old flat
# concatenation.
# ---------------------------------------------------------------------------

# vocab used to derive a stable "handle" from a character's free-text positive
# when no explicit name is given (longer entries first so "old man" beats "man").
_ROLE_WORDS = [
    "old man", "old woman", "young man", "young woman", "boyfriend", "girlfriend",
    "teenager", "gentleman", "princess", "prince", "warrior", "soldier", "student",
    "mother", "father", "sister", "brother", "knight", "witch", "queen", "king",
    "nurse", "maid", "woman", "man", "girl", "boy", "lady", "guy", "child", "kid",
    "teen", "person", "figure", "mom", "dad", "elf",
]
_DESCRIPTOR_WORDS = [
    "blonde", "blond", "brunette", "redhead", "red-haired", "dark-haired",
    "black-haired", "silver-haired", "white-haired", "grey-haired", "gray-haired",
    "pink-haired", "blue-haired", "long-haired", "short-haired", "young", "old",
    "elderly", "mature", "adult", "middle-aged", "tall", "short", "small", "large", "muscular", "athletic", "slim",
    "elegant", "beautiful", "handsome", "cute", "sleeping", "smiling",
]
_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
             "eighth", "ninth", "tenth"]
_NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
              "eight", "nine", "ten", "eleven", "twelve"]
# per-character negative -> positive counter-trait (negative_mode=to_positive_assertion)
_ANTONYMS = {
    "old": "young", "old man": "young man", "old woman": "young woman",
    "old face": "youthful face", "elderly": "youthful", "child": "adult",
    "children": "adults", "ugly": "beautiful", "masculine woman": "feminine woman",
    "feminine man": "masculine man", "fat": "slim", "awake": "asleep",
    "standing": "seated", "large in frame": "small in the frame",
    "close-up": "seen at a distance", "far apart": "close together",
    "not touching": "touching", "facing away": "facing each other",
}


def _num_word(n):
    return _NUM_WORDS[n] if 0 <= n < len(_NUM_WORDS) else str(n)


def _cap(s):
    return (s[0].upper() + s[1:]) if s else s


def _join_and(items):
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return items[0] + " and " + items[1]
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _match_vocab(text, vocab):
    t = " " + " ".join((text or "").lower().replace(",", " ").replace(".", " ").split()) + " "
    for w in vocab:
        if " " + w + " " in t:
            return w
    return ""


def _derive_handle(name, positive, index, *, role_descriptions=False):
    """A short, stable referent reused in the roster and interactions. Uses the
    explicit name when set, else a distinctive noun phrase from the positive."""
    name = (name or "").strip()
    if name:
        if role_descriptions:
            words = name.lower().split()
            roles = set(_ROLE_WORDS)
            descriptors = set(_DESCRIPTOR_WORDS)
            if (len(words) <= 4 and any(name.lower().endswith(role) for role in roles)
                    and (all(w in descriptors or w in roles or w in {"the", "a", "an"}
                             for w in words))):
                phrase = name.lower()
                for article in ("the ", "a ", "an "):
                    if phrase.startswith(article):
                        phrase = phrase[len(article):]
                        break
                return "the " + phrase
        return ("the " + name.lower()) if name.lower() in _ROLE_WORDS else name
    role = _match_vocab(positive, _ROLE_WORDS) or "person"
    desc = _match_vocab(positive, _DESCRIPTOR_WORDS)
    if role == "person" and not desc:
        return "person " + str(index + 1)
    return ("the " + desc + " " + role) if desc else ("the " + role)


def _dedup_handles(handles):
    """Make identical handles distinct ('the first blonde woman', ...)."""
    counts = {}
    for h in handles:
        counts[h] = counts.get(h, 0) + 1
    seen, out = {}, []
    for h in handles:
        if counts[h] > 1:
            i = seen.get(h, 0)
            seen[h] = i + 1
            ordw = _ORDINALS[i] if i < len(_ORDINALS) else str(i + 1)
            out.append(("the " + ordw + " " + h[4:]) if h.startswith("the ") else (h + " (" + ordw + ")"))
        else:
            out.append(h)
    return out


def _cell_fractions(cells, cols, rows):
    cells = [c for c in cells if isinstance(c, int) and 0 <= c < cols * rows]
    if not cells:
        return None, None
    cf = (sum(c % cols for c in cells) / len(cells)) / (cols - 1) if cols > 1 else 0.5
    rf = (sum(c // cols for c in cells) / len(cells)) / (rows - 1) if rows > 1 else 0.5
    return cf, rf


def _loc_phrase(cf, rf, mode):
    """Grid position -> location wording. coarse = left/center/right (+ fore/
    background); fine/grid_coords = 2D area (upper-left, bottom center, ...)."""
    if cf is None:
        return ""
    if mode == "coarse":
        col = ("on the left side of the scene" if cf < 0.34
               else "in the center of the scene" if cf <= 0.66
               else "on the right side of the scene")
        row = (" in the foreground" if rf > 0.66 else " in the far background" if rf < 0.34 else "")
        return col + row
    horiz = ("far left" if cf < 0.2 else "left" if cf < 0.4
             else "center" if cf <= 0.6 else "right" if cf <= 0.8 else "far right")
    vert = ("top" if rf < 0.2 else "upper" if rf < 0.4
            else "" if rf <= 0.6 else "lower" if rf <= 0.8 else "bottom")
    if not vert:
        if horiz == "center":
            return "in the center of the scene"
        if horiz in ("left", "right"):
            return "on the " + horiz + " side of the scene"
        return "on the " + horiz + " of the scene"
    return "in the " + (vert + " " + horiz).strip() + " area of the scene"


def _scale_phrase(rf):
    """Depth (row) -> size wording: top = small/distant, bottom = large/close."""
    if rf is None:
        return ""
    if rf < 0.34:
        return "small and distant, a small figure far away in the background"
    if rf > 0.66:
        return "large and prominent, close to the viewer in the foreground"
    return ""


def _excel_cells(cells, cols):
    """Spreadsheet-style cell tags: column letters A,B,C..., row numbers 1,2,3..."""
    out = []
    for c in sorted(x for x in cells if isinstance(x, int) and x >= 0):
        out.append(chr(65 + (c % cols)) + str(c // cols + 1))
    return ", ".join(out)


def _convert_negatives(neg_text):
    """Split a per-character negative into (positive counter-traits, leftovers)."""
    additions, leftovers = [], []
    for term in [t.strip() for t in (neg_text or "").split(",") if t.strip()]:
        if term.lower() in _ANTONYMS:
            additions.append(_ANTONYMS[term.lower()])
        else:
            leftovers.append(term)
    return additions, leftovers


def _dedup_terms(chunks):
    terms, seen = [], set()
    for chunk in chunks:
        for t in (chunk or "").split(","):
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                terms.append(t)
    return terms


def _action_needs_subject(text):
    """True if the interaction text is a bare action fragment ('kissing ...') that
    should get '{who} are ...' prepended; False if it already reads as a full
    clause with its own subject ('the woman and the man are kissing ...')."""
    words = (text or "").strip().lower().split()
    first = words[0] if words else ""
    return first not in ("the", "a", "an", "they", "he", "she", "both", "two", "three")


def _report_md(s, gpos, chars, cols, interactions, neg_terms, pos_text, neg_text,
               *, audit=None, adjustments=None, dropped_negatives=None, warnings=None):
    """Human-readable markdown of everything the composer decided."""
    L = ["# Multi-Char Prompt — Assembled Breakdown", ""]
    L.append("## Settings")
    L.append("- format: **%s** · spatial: **%s** · grid: **%dx%d**"
             % (s["format"], s["spatial"], s["grid"][0], s["grid"][1]))
    L.append("- count lock: **%s** · names: **%s** · roster: **%s** · order+group: **%s**"
             % (s["count"], s["names"], s["roster"], s["order_group"]))
    L.append("- scale hints: **%s** · bind interactions: **%s** · auto framing: **%s** · negatives: **%s**"
             % (s["scale"], s["bind"], s["framing"], s["neg_mode"]))
    L.append("- profile: **%s** · negative cap: **%s** · cap priority: **%s** · spatial cues: **%s** · scale cues: **%s**"
             % (s["profile"], s["cap"] or "unlimited", s["cap_priority"],
                s["spatial_cues"], s["scale_cues"]))
    L += ["", "## Global positive", "> " + (gpos if gpos else "*(empty)*"), ""]
    L.append("## Characters (%d)" % len(chars))
    for c in chars:
        cells = ", ".join(str(x) for x in sorted(c["cells"])) or "none"
        grid = _excel_cells(c["cells"], cols) or "—"
        loc = c["loc"] or ("*(not emitted)*" if c.get("loc_skipped") and c["cells"] else "*(whole image)*")
        bits = ["cells [%s]" % cells, "grid %s" % grid, "loc: %s" % loc]
        if c["scale"]:
            bits.append("scale: " + c["scale"])
        if c["pos_add"]:
            bits.append("neg→pos: " + ", ".join(c["pos_add"]))
        L.append("- **%s** (label: %s) — %s" % (c["handle"], c["label"], " · ".join(bits)))
    L += ["", "## Interactions (%d)" % len(interactions)]
    if interactions:
        for refs, sent in interactions:
            L.append("- between #%s → %s" % (",".join(str(r) for r in refs) or "?", sent))
    else:
        L.append("- *(none)*")
    L += ["", "## Negative terms (%d)" % len(neg_terms),
          "> " + (", ".join(neg_terms) if neg_terms else "*(empty)*"), ""]
    L += ["## Duplication audit"]
    if audit:
        for clause, blocks in audit:
            L.append("- `%s` in %s" % (clause, ", ".join(blocks)))
    else:
        L.append("- *(no exact repeated clauses across blocks)*")
    L += ["", "## Profile adjustments"]
    L.extend("- " + item for item in (adjustments or ["None."]))
    if dropped_negatives:
        L.append("- Dropped negative terms: " + ", ".join(dropped_negatives))
    L += ["", "## Length stats"]
    for name, value in (("Positive", pos_text), ("Negative", neg_text)):
        words = len(value.split())
        estimate = max(0, (len(value) + 3) // 4)
        L.append("- %s: %d words, about %d tokens (character estimate)." % (name, words, estimate))
    L.append("- Token counts are rough; the active text encoder may split text differently.")
    L.extend("- Warning: " + item for item in (warnings or []))
    L.append("")
    L += ["## FINAL POSITIVE", "```", pos_text, "```", ""]
    L += ["## FINAL NEGATIVE", "```", neg_text, "```"]
    return "\n".join(L)


def _clause_key(text):
    return " ".join(text.strip().rstrip(".!? ").lower().split())


def _clauses(text):
    return [part.strip() for part in re.split(r"[,;\n]+", text or "") if part.strip()]


def _duplication_audit(global_positive, global_negative, chars, links):
    blocks = [("positive", "global positive", global_positive),
              ("negative", "global negative", global_negative)]
    blocks += [("positive", "character #%d positive" % c["ref"], c["positive"])
               for c in chars]
    blocks += [("negative", "character #%d negative" % c["ref"], c["negative"])
               for c in chars]
    blocks += [("positive", "interaction #%d positive" % i, (ln.get("positive") or ""))
               for i, ln in enumerate(links, 1)]
    blocks += [("negative", "interaction #%d negative" % i, (ln.get("negative") or ""))
               for i, ln in enumerate(links, 1)]
    seen = {}
    for polarity, label, value in blocks:
        for clause in _clauses(value):
            key = _clause_key(clause)
            if key:
                entry = seen.setdefault((polarity, key), (clause, []))
                if label not in entry[1]:
                    entry[1].append(label)
    return [(clause, labels) for clause, labels in seen.values() if len(labels) > 1]


def _explicit_count(text):
    values = {word: i for i, word in enumerate(_NUM_WORDS)}
    word = r"(\d+|" + "|".join(_NUM_WORDS) + r")"
    patterns = [r"\b(?:exactly|only)\s+" + word + r"\s+(?:people|persons|person|characters|subjects)\b",
                r"\btotal\s+headcount\s*(?:is|:|of)?\s*" + word + r"\b"]
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text or "", re.IGNORECASE):
            count = match.group(1).lower()
            matches.append(int(count) if count.isdigit() else values[count])
    return matches


def _flux_sentence(text):
    text = (text or "").strip()
    if not text:
        return text
    text = re.sub(r"([.!?])\.+$", r"\1", text)
    text = re.sub(r"(?<=[.!?])\s+([a-z])", lambda m: " " + m.group(1).upper(), text)
    text = _cap(text)
    return text if text.endswith((".", "!", "?")) else text + "."


def assemble_multichar(cols, rows, raw_chars, links, opts, *, prompt_profile="default",
                       negative_term_cap=0, spatial_cues="auto", scale_cues="off",
                       negative_cap_priority="global_first"):
    """Pure text assembly shared by the compose node and the /multichar/preview
    route (single source of truth — no encoding). Returns {positive, negative, report}."""
    global_positive = opts.get("global_positive", "") or ""
    global_negative = opts.get("global_negative", "") or ""
    subject_count_lock = bool(opts.get("subject_count_lock", True))
    use_names = opts.get("use_names", "handle")
    bind_interactions = bool(opts.get("bind_interactions", True))
    cast_roster = bool(opts.get("cast_roster", True))
    order_and_group = bool(opts.get("order_and_group", True))
    auto_scale_hints = bool(opts.get("auto_scale_hints", True))
    spatial_detail = opts.get("spatial_detail", "fine")
    output_format = opts.get("output_format", "prose")
    auto_framing = bool(opts.get("auto_framing", False))
    negative_mode = opts.get("negative_mode", "global_dedup")
    cols = int(cols or 3)
    rows = int(rows or 1)
    raw_chars = raw_chars or []
    links = links or []
    flux2 = prompt_profile == "flux2"
    adjustments, warnings = [], []

    # keep only characters that actually describe someone; remember their 1-based
    # position so interaction links (which reference it) still resolve.
    chars = []
    for i, c in enumerate(raw_chars):
        p = (c.get("positive") or "").strip()
        if not p:
            continue
        chars.append({
            "ref": i + 1,
            "name": (c.get("name") or "").strip(),
            "positive": p,
            "negative": (c.get("negative") or "").strip(),
            "cells": [x for x in (c.get("cells") or []) if isinstance(x, int)],
            "pos_add": [],
        })

    # stable, de-duplicated handles (used in roster + interactions)
    handles = _dedup_handles([_derive_handle(c["name"], c["positive"], k,
                                               role_descriptions=flux2)
                              for k, c in enumerate(chars)])
    for c, h in zip(chars, handles):
        c["handle"] = h
        c["label"] = c["name"] if c["name"] else (_cap(h[4:]) if h.startswith("the ") else _cap(h))

    # placement (location phrase, scale hint, excel coords)
    same_cells = len(chars) > 1 and len({tuple(sorted(c["cells"])) for c in chars}) == 1
    skip_spatial = flux2 and (spatial_cues == "off" or
                              (spatial_cues == "auto" and (cols * rows == 1 or same_cells)))
    skip_scale = flux2 and scale_cues != "row_based"
    if skip_spatial:
        adjustments.append("Skipped grid location cues: no useful relative position in this layout."
                           if spatial_cues == "auto" else "Skipped grid location cues by setting.")
    if skip_scale and auto_scale_hints:
        adjustments.append("Skipped row-based scale cues; grid height alone does not establish depth.")
    for c in chars:
        cf, rf = _cell_fractions(c["cells"], cols, rows)
        c["cf"], c["rf"] = cf, rf
        c["loc_skipped"] = skip_spatial
        c["loc"] = "" if skip_spatial else _loc_phrase(cf, rf, spatial_detail)
        c["scale"] = _scale_phrase(rf) if auto_scale_hints and not skip_scale else ""
        c["coords"] = (_excel_cells(c["cells"], cols)
                       if spatial_detail == "grid_coords" and not skip_spatial else "")

    byref = {c["ref"]: c for c in chars}

    # ---- negatives (pre-pass: to_positive_assertion feeds the positive) ----
    global_neg = [global_negative.strip()] if global_negative.strip() else []
    char_neg = []
    link_neg = []
    for c in chars:
        if not c["negative"]:
            continue
        if negative_mode == "to_positive_assertion":
            adds, leftover = _convert_negatives(c["negative"])
            c["pos_add"].extend(adds)
            if leftover:
                char_neg.append(", ".join(leftover))
        else:
            char_neg.append(c["negative"])
    for ln in links:
        n = (ln.get("negative") or "").strip()
        if n:
            link_neg.append(n)
    cap = max(0, int(negative_term_cap or 0))
    neg_chunks = (link_neg + global_neg + char_neg if flux2 and cap and negative_cap_priority == "interaction_first"
                  else global_neg + char_neg + link_neg)
    neg_terms = _dedup_terms(neg_chunks)
    dropped_negatives = []
    if cap and len(neg_terms) > cap:
        dropped_negatives = neg_terms[cap:]
        neg_terms = neg_terms[:cap]
        adjustments.append("Capped negatives at %d terms; dropped %d." % (cap, len(dropped_negatives)))
    neg_text = ", ".join(neg_terms)

    # ---- positive ----
    def desc_of(c):
        extra = list(c["pos_add"])
        if c["scale"]:
            extra.append(c["scale"])
        if c["coords"]:
            extra.append("located around grid " + c["coords"])
        return (c["positive"] + ", " + ", ".join(extra)) if extra else c["positive"]

    pos_parts = []
    if global_positive.strip():
        pos_parts.append(global_positive.strip())

    if auto_framing and chars:
        cfs = [c["cf"] for c in chars if c["cf"] is not None]
        if len(cfs) >= 2 and (max(cfs) - min(cfs)) > 0.5:
            pos_parts.append("This is a wide establishing shot that shows the whole scene.")
        elif len(chars) == 1:
            pos_parts.append("This is a medium close-up shot.")
        else:
            pos_parts.append("This is a medium shot.")

    counts = _explicit_count(global_positive) if flux2 else []
    skip_count = flux2 and bool(chars) and len(chars) in counts
    if skip_count and subject_count_lock:
        adjustments.append("Skipped count lock: global positive already states %d people." % len(chars))
    if counts and any(count != len(chars) for count in counts):
        warnings.append("Global positive has a count that conflicts with the %d described characters." % len(chars))
    if subject_count_lock and chars and not skip_count:
        n = len(chars)
        pos_parts.append("There is exactly one person in the scene and no one else."
                         if n == 1 else
                         "There are exactly " + _num_word(n) + " people in the scene and no one else.")

    if cast_roster and chars:
        pos_parts.append("Cast: " + "; ".join(c["handle"] for c in chars) + ".")

    # order + group characters by shared cells (co-located = one clause)
    if order_and_group:
        groups, order = {}, []
        for c in chars:
            sig = tuple(sorted(c["cells"]))
            if sig not in groups:
                groups[sig] = []
                order.append(sig)
            groups[sig].append(c)

        def gkey(sig):
            g = groups[sig]
            cfs = [x["cf"] for x in g if x["cf"] is not None]
            rfs = [x["rf"] for x in g if x["rf"] is not None]
            return (min(cfs) if cfs else 0.5, min(rfs) if rfs else 0.5)

        order.sort(key=gkey)
        grouped = [groups[sig] for sig in order]
    else:
        grouped = [[c] for c in chars]

    num = 0
    group_end_positions = []
    group_by_ref = {}
    for group_index, group in enumerate(grouped):
        for c in group:
            group_by_ref[c["ref"]] = group_index
        loc = group[0]["loc"]
        loc_cap = _cap(loc)
        rendered = []
        for c in group:
            num += 1
            subj = "" if use_names == "off" else (c["label"] if use_names == "label" else c["handle"])
            if flux2 and output_format in ("labeled", "numbered") and c["name"] and use_names == "handle":
                subj = c["label"]
            rendered.append((num, subj, desc_of(c)))

        if output_format == "numbered":
            for n_, subj, d in rendered:
                if subj:
                    body = (subj + " (" + loc + "): " + d + ".") if loc else (subj + ": " + d + ".")
                else:
                    body = ("(" + loc + ") " + d + ".") if loc else (d + ".")
                pos_parts.append(str(n_) + ") " + body)
        elif output_format == "labeled":
            if loc_cap and len(rendered) > 1:
                inner = "; ".join((subj + ": " + d) if subj else d for _, subj, d in rendered)
                pos_parts.append(loc_cap + ": " + inner + ".")
            else:
                for _, subj, d in rendered:
                    if subj:
                        pos_parts.append((subj + " (" + loc + "): " + d + ".") if loc else (subj + ": " + d + "."))
                    else:
                        pos_parts.append((loc_cap + ", " + d + ".") if loc_cap else (d + "."))
        else:  # prose
            if len(rendered) > 1 and loc_cap:
                inner = "; and ".join((subj + " — " + d) if subj else d for _, subj, d in rendered)
                pos_parts.append(loc_cap + ", " + inner + ".")
            else:
                for _, subj, d in rendered:
                    if subj:
                        pos_parts.append((loc_cap + ", " + subj + " — " + d + ".") if loc_cap
                                         else (_cap(subj) + " — " + d + "."))
                    else:
                        pos_parts.append((loc_cap + ", " + d + ".") if loc_cap else (d + "."))
        group_end_positions.append(len(pos_parts))

    # interactions (bound to the named characters, or verbatim)
    interactions_report = []
    insert_after = {}
    deduped_clauses = 0
    for ln in links:
        p = (ln.get("positive") or "").strip()
        if not p:
            continue
        refs = [b for b in (ln.get("between") or []) if isinstance(b, int)]
        members = [byref[b] for b in refs if b in byref]
        if flux2 and members:
            member_clauses = {_clause_key(clause) for m in members for clause in _clauses(m["positive"])}
            clauses = _clauses(p)
            kept = [clause for clause in clauses if _clause_key(clause) not in member_clauses]
            if not kept:
                deduped_clauses += len(clauses)
                interactions_report.append((refs, "(omitted: all clauses repeat member descriptions)"))
                continue
            if kept[0] != clauses[0] and not (_action_needs_subject(kept[0]) and
                                              kept[0].split()[0].lower().endswith("ing")):
                kept.insert(0, clauses[0])
            deduped_clauses += len(clauses) - len(kept)
            p = ", ".join(kept)
        if bind_interactions and members and _action_needs_subject(p):
            subj = _join_and([m["handle"] for m in members])
            verb = "is" if len(members) == 1 else "are"
            sent = _cap(subj) + " " + verb + " " + p + "."
        else:
            sent = p + "."
        if flux2 and members:
            last_group = max(group_by_ref[m["ref"]] for m in members)
            insert_after.setdefault(last_group, []).append(sent)
        else:
            pos_parts.append(sent)
        interactions_report.append((refs, sent))

    if flux2 and insert_after:
        reordered, previous = [], 0
        for group_index, end in enumerate(group_end_positions):
            reordered.extend(pos_parts[previous:end])
            reordered.extend(insert_after.get(group_index, []))
            previous = end
        reordered.extend(pos_parts[previous:])
        pos_parts = reordered
    if flux2:
        pos_parts = [_flux_sentence(part) for part in pos_parts]
        if deduped_clauses:
            adjustments.append("Removed %d exact interaction clauses already in member descriptions." % deduped_clauses)
    if cap and dropped_negatives and not flux2:
        adjustments.append("Negative cap applies to the default profile because it was set explicitly.")
    pos_text = " ".join(pos_parts)
    audit = _duplication_audit(global_positive, global_negative, chars, links)
    if max((len(pos_text) + 3) // 4, (len(neg_text) + 3) // 4) > 450:
        warnings.append("Approximate length exceeds 450 tokens; check the active encoder context limit.")
    report = _report_md(
        {"format": output_format, "spatial": spatial_detail, "scale": auto_scale_hints,
         "count": subject_count_lock, "names": use_names, "roster": cast_roster,
         "order_group": order_and_group, "bind": bind_interactions,
         "framing": auto_framing, "neg_mode": negative_mode, "grid": (cols, rows),
         "profile": prompt_profile, "cap": cap, "cap_priority": negative_cap_priority,
         "spatial_cues": spatial_cues, "scale_cues": scale_cues},
        global_positive.strip(), chars, cols, interactions_report, neg_terms, pos_text, neg_text,
        audit=audit, adjustments=adjustments, dropped_negatives=dropped_negatives, warnings=warnings)
    return {"positive": pos_text, "negative": neg_text, "report": report}


def _compose_layout(layout, layout_json_override):
    if not layout_json_override or not layout_json_override.strip():
        return layout or {}
    try:
        parsed = json.loads(layout_json_override)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid Layout JSON override: %s" % exc) from exc
    if not isinstance(parsed, dict):
        raise ValueError("Layout JSON override must be an object")
    chars = parsed.get("characters", (layout or {}).get("characters", []))
    links = parsed.get("links", parsed.get("interactions", (layout or {}).get("links", [])))
    if not isinstance(chars, list) or not isinstance(links, list):
        raise ValueError("Layout JSON characters and links must be arrays")
    return {
        "grid_cols": max(1, min(8, int(parsed.get("grid_cols", (layout or {}).get("grid_cols", 3))))),
        "grid_rows": max(1, min(8, int(parsed.get("grid_rows", (layout or {}).get("grid_rows", 1))))),
        "characters": chars,
        "links": links,
    }


class MultiCharPromptCompose:
    """Structured multi-character PROMPT composer - the breakdown editor WITHOUT masks.

    Reuses the RegionalCharacterLayout editor (per-character cards + interaction
    links + a grid) and assembles ONE natural-language positive + negative that a
    strong text encoder (Chroma / Flux / SD3) can follow. Optional structuring
    aids help the model understand who is present, where, and doing what:
    count lock, stable handles, bound interactions, cast roster, ordered/grouped
    placement, scale hints, finer spatial wording, output format, auto framing,
    and negative handling. With all aids off it reproduces the old flat output.

    Outputs CONDITIONING (positive, negative), the assembled positive/negative
    text, and a markdown prompt_report for the Multi-Char Prompt Preview node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "global_positive": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Scene-wide style + environment applied to the whole image (art style, setting, lighting)."}),
                "global_negative": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Things to avoid everywhere. On Chroma/Flux the negative is global (no per-region binding)."}),
                "subject_count_lock": ("BOOLEAN", {"default": True, "label_on": "state the count", "label_off": "no count",
                    "tooltip": "ON: adds 'There are exactly N people and no one else' to stop dropped/extra characters. OFF: model may add background people."}),
                "use_names": (["handle", "label", "off"], {"default": "handle",
                    "tooltip": "How each character is named. handle: natural referent ('the blonde woman') reused everywhere (best coherence). label: 'Woman: ...' prefix (most readable). off: no subject prefix (old behavior; fine for one character)."}),
                "bind_interactions": ("BOOLEAN", {"default": True, "label_on": "bind to characters", "label_off": "verbatim",
                    "tooltip": "ON: rewrites an interaction to name its two characters ('The woman and the man are kissing...') from the between chips — type only the action. OFF: uses the interaction text verbatim."}),
                "cast_roster": ("BOOLEAN", {"default": True, "label_on": "list the cast", "label_off": "no roster",
                    "tooltip": "ON: adds a 'Cast: ...' line up front so the model registers distinct people before the detailed sentences (reduces merging). Turn off for 1-2 simple characters."}),
                "order_and_group": ("BOOLEAN", {"default": True, "label_on": "order + group", "label_off": "card order",
                    "tooltip": "ON: sorts characters left->right and merges same-cell characters into one clause. OFF: card order, each separate. Keep ON for multi-character scenes."}),
                "auto_scale_hints": ("BOOLEAN", {"default": True, "label_on": "add scale words", "label_off": "no scale",
                    "tooltip": "ON: adds size words from grid depth (top row = small/distant, bottom row = large/close). Needs a grid with rows>1. Fixes a background character rendering too big."}),
                "spatial_detail": (["fine", "coarse", "grid_coords"], {"default": "fine",
                    "tooltip": "Grid cell -> words. fine: left/center/right + upper/lower areas. coarse: just left/center/right (+ fore/background). grid_coords: fine wording PLUS spreadsheet tags (A1,B2) — experimental extra cue."}),
                "output_format": (["prose", "labeled", "numbered"], {"default": "prose",
                    "tooltip": "Sentence style. prose: flowing sentences (best for the model). labeled: 'Name (location): ...' (readable). numbered: '1) ...' (most explicit)."}),
                "auto_framing": ("BOOLEAN", {"default": False, "label_on": "derive shot", "label_off": "off",
                    "tooltip": "ON: derives a shot from character spread (wide establishing shot when spread out, else medium shot). Turn OFF if you set the shot yourself in global_positive."}),
                "negative_mode": (["global_dedup", "to_positive_assertion"], {"default": "global_dedup",
                    "tooltip": "Per-character negatives. global_dedup: merge + de-duplicate all negatives. to_positive_assertion: convert a character negative into a positive counter-trait ('old'->'young') on that character (often obeyed better)."}),
            },
            "optional": {
                "layout": ("REGIONAL_LAYOUT",),
                "prompt_profile": (["default", "flux2"], {"default": "default",
                    "tooltip": "default keeps the existing prompt assembly. flux2 trims redundant cues and places interactions near their characters."}),
                "negative_term_cap": ("INT", {"default": 0, "min": 0, "max": 1000,
                    "tooltip": "0 keeps all deduplicated negative terms. A positive value caps them in the selected priority order."}),
                "negative_cap_priority": (["global_first", "interaction_first"], {"default": "global_first",
                    "tooltip": "When a cap is set, keep global negatives first or keep interaction guards first. interaction_first works in flux2 mode."}),
                "spatial_cues": (["auto", "always", "off"], {"default": "auto",
                    "tooltip": "flux2 only: auto skips grid wording when it carries no relative position; always forces it; off suppresses it."}),
                "scale_cues": (["off", "row_based"], {"default": "off",
                    "tooltip": "flux2 only: row_based infers apparent depth from row position; off avoids that assumption."}),
                "layout_json_override": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Optional pasted layout. When set, it replaces the connected layout for this composer only."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "positive_text", "negative_text", "prompt_report")
    FUNCTION = "compose"
    CATEGORY = "Regional/conditioning"
    DESCRIPTION = (
        "Multi-character breakdown by WORDING (not masks). Mirrors the character/"
        "interaction layout in one natural-language prompt with optional structuring "
        "aids (count lock, stable handles, bound interactions, cast roster, ordered/"
        "grouped placement, scale hints, finer spatial wording, output format, auto "
        "framing, negative handling). Best on Chroma/Flux/SD3. Also emits a markdown "
        "prompt_report for the Preview node."
    )

    def compose(self, clip, global_positive, global_negative, subject_count_lock,
                use_names, bind_interactions, cast_roster, order_and_group,
                auto_scale_hints, spatial_detail, output_format, auto_framing,
                negative_mode, layout=None, *, prompt_profile="default", negative_term_cap=0,
                negative_cap_priority="global_first", spatial_cues="auto", scale_cues="off",
                layout_json_override=""):
        layout = _compose_layout(layout, layout_json_override)
        cols = int(layout.get("grid_cols", 3) or 3) if layout else 3
        rows = int(layout.get("grid_rows", 1) or 1) if layout else 1
        raw_chars = (layout.get("characters") if layout else []) or []
        links = (layout.get("links") if layout else []) or []
        r = assemble_multichar(cols, rows, raw_chars, links, {
            "global_positive": global_positive, "global_negative": global_negative,
            "subject_count_lock": subject_count_lock, "use_names": use_names,
            "bind_interactions": bind_interactions, "cast_roster": cast_roster,
            "order_and_group": order_and_group, "auto_scale_hints": auto_scale_hints,
            "spatial_detail": spatial_detail, "output_format": output_format,
            "auto_framing": auto_framing, "negative_mode": negative_mode},
            prompt_profile=prompt_profile, negative_term_cap=negative_term_cap,
            negative_cap_priority=negative_cap_priority, spatial_cues=spatial_cues,
            scale_cues=scale_cues)
        pos = clip.encode_from_tokens_scheduled(clip.tokenize(r["positive"]))
        neg = clip.encode_from_tokens_scheduled(clip.tokenize(r["negative"]))
        return (pos, neg, r["positive"], r["negative"], r["report"])


def _prompts_from_report(text):
    marker = "## FINAL POSITIVE\n```\n"
    divider = "\n```\n\n## FINAL NEGATIVE\n```\n"
    start = text.rfind(marker)
    if start < 0:
        return None
    start += len(marker)
    middle = text.find(divider, start)
    if middle < 0:
        return None
    neg_start = middle + len(divider)
    end = text.rfind("\n```")
    if end < neg_start:
        return None
    return text[start:middle], text[neg_start:end]


class MultiCharPromptPreview:
    """Show the exact positive and negative prompt text from a compose report.

    Direct text inputs can override report parsing. The text output passes through
    the original input so existing graph links keep working.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"text": ("STRING", {"forceInput": True})},
            "optional": {
                "positive_text": ("STRING", {"forceInput": True}),
                "negative_text": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show"
    OUTPUT_NODE = True
    CATEGORY = "Regional/conditioning"
    DESCRIPTION = (
        "Shows the raw positive and negative text in separate copyable blocks. "
        "Wire prompt_report to text; optional positive_text and negative_text inputs "
        "take the exact composer outputs directly."
    )

    def show(self, text, positive_text=None, negative_text=None):
        text = text if isinstance(text, str) else ("" if text is None else str(text))
        parsed = _prompts_from_report(text)
        if positive_text is None and parsed is not None:
            positive_text = parsed[0]
        if negative_text is None and parsed is not None:
            negative_text = parsed[1]
        has_prompts = positive_text is not None or negative_text is not None
        positive_text = positive_text if isinstance(positive_text, str) else (
            "" if positive_text is None else str(positive_text))
        negative_text = negative_text if isinstance(negative_text, str) else (
            "" if negative_text is None else str(negative_text))
        return {
            "ui": {"text": [text], "positive_text": [positive_text],
                   "negative_text": [negative_text], "has_prompts": [has_prompts]},
            "result": (text,),
        }


def _list_llm_dirs():
    """Discover local HF instruct-LLM directories (config.json + weights) under the
    ComfyUI models dir, so the enhancer can offer them in a dropdown."""
    import os
    roots = []
    try:
        import folder_paths
        m = folder_paths.models_dir
        roots = [os.path.join(m, "LLM"), os.path.join(m, "LLM", "Qwen-VL"),
                 os.path.join(m, "llm"), os.path.join(m, "prompt_generator")]
    except Exception:
        pass
    found = {}
    for r in roots:
        if not os.path.isdir(r):
            continue
        for name in sorted(os.listdir(r)):
            p = os.path.join(r, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
                found.setdefault(name, p)
    return found


class MultiCharLayoutEnhancer:
    """OPTIONAL: enrich each character's (and interaction's) description in a layout
    with a local uncensored instruct LLM, then pass the enriched layout to Multi-Char
    Prompt Compose. Compose still does the STRUCTURE (count-lock, placement, interaction
    binding); this only makes the per-character DESCRIPTIONS richer, so the structure is
    preserved. `enable` off = passthrough, so it's safe to leave in a graph.

    Loads the chosen model with transformers, frees ComfyUI's other models first so it
    fits on the GPU, and frees itself before returning so the sampler has VRAM. On any
    failure it passes the layout through unchanged (never breaks a run).
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = _list_llm_dirs()
        cls._MODELS = models
        names = list(models.keys()) or ["(put an HF model in models/LLM/Qwen-VL)"]
        return {
            "required": {
                "layout": ("REGIONAL_LAYOUT",),
                "enable": ("BOOLEAN", {"default": False, "label_on": "enrich", "label_off": "bypass"}),
                "model": (names,),
                "enrich_characters": ("BOOLEAN", {"default": True}),
                "enrich_interactions": ("BOOLEAN", {"default": True}),
                "max_words": ("INT", {"default": 50, "min": 12, "max": 160}),
                "smart_wording": ("BOOLEAN", {"default": True, "label_on": "smart (adaptive length)", "label_off": "fixed (always short)"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("REGIONAL_LAYOUT", "STRING")
    RETURN_NAMES = ("layout", "report")
    FUNCTION = "enhance"
    CATEGORY = "Regional/conditioning"
    DESCRIPTION = (
        "Enrich each character description in a layout with a local uncensored LLM, then "
        "feed Multi-Char Prompt Compose (which keeps the structure). Off = passthrough. "
        "Best on a strong text-encoder model (Chroma/Flux/SD3)."
    )

    def enhance(self, layout, enable, model, enrich_characters, enrich_interactions,
                max_words, smart_wording, temperature, top_p, seed):
        import copy, os
        lay = copy.deepcopy(layout) if isinstance(layout, dict) else {"characters": [], "links": []}
        if not enable:
            return (lay, "layout enhancer: bypassed")
        model_path = (getattr(self, "_MODELS", None) or _list_llm_dirs()).get(model)
        if not model_path or not os.path.isdir(model_path):
            return (lay, "layout enhancer: model not found (%s) - passthrough" % model)
        try:
            import torch, gc
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            return (lay, "layout enhancer: transformers unavailable (%s) - passthrough" % e)

        # free ComfyUI's cached models so the LLM fits on the GPU
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception:
            pass

        if smart_wording:
            # adaptive: max_words is a soft UPPER bound - use only as many as needed
            SYS_C = ("You rewrite ONE character's description for a black-and-white MANGA image (grayscale, no color). "
                     "Rules: use only as many words as this character genuinely needs - be concise (a simple character "
                     "may need just 10-15 words, an elaborate one more), and NEVER exceed %d words; make this character "
                     "clearly DISTINCT from anyone else (unique hair, build, age, outfit); give face/expression, outfit, "
                     "and a clear full-body pose stating where the arms, hands, and legs are; NEVER use color words (it is "
                     "grayscale - use tone words like pale, dark, light, or omit color); do NOT mention other people, the "
                     "room, or the background. Output only the description as comma-separated phrases, no preamble, no quotes." % int(max_words))
            SYS_L = ("Rewrite this interaction between characters for a manga image. Describe the physical contact and "
                     "both poses concretely (grips, hands, bodies - e.g. holding a bottle by its neck tilted directly "
                     "above a glass with a stream pouring in). Use only as many words as needed, never more than 28. "
                     "No color words. Output only the description.")
        else:
            # fixed: always terse, aim for the cap (the original behaviour)
            SYS_C = ("You rewrite ONE character's description for a black-and-white MANGA image (grayscale, no color). "
                     "Rules: keep it SHORT (max %d words); make this character clearly DISTINCT from anyone else "
                     "(unique hair, build, age, outfit); give face/expression, outfit, and a clear full-body pose "
                     "stating where the arms, hands, and legs are; NEVER use color words (it is grayscale - use tone "
                     "words like pale, dark, light, or omit color); do NOT mention other people, the room, or the "
                     "background. Output only the description as comma-separated phrases, no preamble, no quotes." % int(max_words))
            SYS_L = ("Rewrite this interaction between characters for a manga image. Describe the physical contact and "
                     "both poses concretely (grips, hands, bodies - e.g. holding a bottle by its neck tilted directly "
                     "above a glass with a stream pouring in). Max 24 words. No color words. Output only the description.")
        try:
            tok = AutoTokenizer.from_pretrained(model_path)
            mdl = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda")
        except Exception as e:
            return (lay, "layout enhancer: model load failed (%s) - passthrough" % e)
        if seed:
            try:
                torch.manual_seed(int(seed))
            except Exception:
                pass

        def gen(sysp, usr, maxn):
            msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": usr}]
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            i = tok(t, return_tensors="pt").to(mdl.device)
            o = mdl.generate(**i, max_new_tokens=maxn, do_sample=temperature > 0,
                             temperature=max(float(temperature), 0.01),
                             top_p=min(max(float(top_p), 0.01), 1.0),
                             repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
            return tok.decode(o[0][i.input_ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")

        rep = []
        try:
            if enrich_characters:
                for c in lay.get("characters", []):
                    if (c.get("positive") or "").strip():
                        c["positive"] = gen(SYS_C, c["positive"], int(max_words) * 3)
                        rep.append("- %s: %s" % (c.get("name", "?"), c["positive"]))
            if enrich_interactions:
                for l in lay.get("links", []):
                    if (l.get("positive") or "").strip():
                        l["positive"] = gen(SYS_L, l["positive"], 90)
                        rep.append("- interaction: %s" % l["positive"])
        finally:
            try:
                del mdl, tok
            except Exception:
                pass
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return (lay, "\n".join(rep) if rep else "layout enhancer: nothing to enrich")


NODE_CLASS_MAPPINGS = {
    "RegionalCharacterLayout": RegionalCharacterLayout,
    "RegionalMultiCharConditioning": RegionalMultiCharConditioning,
    "MultiCharPromptCompose": MultiCharPromptCompose,
    "MultiCharPromptPreview": MultiCharPromptPreview,
    "MultiCharLayoutEnhancer": MultiCharLayoutEnhancer,
    "RegionalFaceDetailerSwitch": RegionalFaceDetailerSwitch,
    "RegionalHiresSwitch": RegionalHiresSwitch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionalCharacterLayout": "Regional Characters (grid layout)",
    "RegionalMultiCharConditioning": "Regional Multi-Char Conditioning",
    "MultiCharPromptCompose": "Multi-Char Prompt Compose (wording)",
    "MultiCharPromptPreview": "Multi-Char Prompt Preview (read-only)",
    "MultiCharLayoutEnhancer": "Multi-Char Layout Enhancer (LLM, optional)",
    "RegionalFaceDetailerSwitch": "Regional FaceDetailer Toggle",
    "RegionalHiresSwitch": "Regional Hires Toggle",
}


# ---------------------------------------------------------------------------
# Live-preview route: the editor POSTs the layout + settings and gets the
# assembled positive/negative/report back from the SAME assembler used for
# generation (single source of truth -> the preview never disagrees with what
# is rendered). Pure string work, no model, so it is instant. Registered only
# if the ComfyUI server is importable (it always is at custom-node load time).
# ---------------------------------------------------------------------------
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/multichar/preview")
    async def _multichar_preview(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        layout = data.get("layout") or {}
        opts = data.get("opts") or {}
        try:
            layout = _compose_layout(layout, opts.get("layout_json_override", ""))
            result = assemble_multichar(
                layout.get("grid_cols", 3), layout.get("grid_rows", 1),
                layout.get("characters") or [], layout.get("links") or [], opts,
                prompt_profile=opts.get("prompt_profile", "default"),
                negative_term_cap=opts.get("negative_term_cap", 0),
                negative_cap_priority=opts.get("negative_cap_priority", "global_first"),
                spatial_cues=opts.get("spatial_cues", "auto"),
                scale_cues=opts.get("scale_cues", "off"))
            return web.json_response(result)
        except Exception as e:  # never 500 the editor; show the error inline
            return web.json_response(
                {"positive": "", "negative": "", "report": "**preview error:** " + str(e)})
except Exception:
    pass
