import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


torch = types.ModuleType("torch")
torch.nn = types.ModuleType("torch.nn")
torch.nn.functional = types.ModuleType("torch.nn.functional")
sys.modules.setdefault("torch", torch)
sys.modules.setdefault("torch.nn", torch.nn)
sys.modules.setdefault("torch.nn.functional", torch.nn.functional)
sys.modules.setdefault("node_helpers", types.ModuleType("node_helpers"))

spec = importlib.util.spec_from_file_location(
    "regional_multichar", Path(__file__).resolve().parents[1] / "regional_multichar.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeClip:
    def tokenize(self, value):
        return value

    def encode_from_tokens_scheduled(self, value):
        return value


class ComposeTests(unittest.TestCase):
    def setUp(self):
        self.characters = [
            {"name": "Mature Woman", "cells": [0], "positive": "woman sitting, looking left",
             "negative": "looking at camera, extra arm"},
            {"name": "Man", "cells": [0], "positive": "man standing, hand on shoulder",
             "negative": "bad hands"},
        ]
        self.links = [
            {"between": [1, 2], "positive": "looking left, holding hands",
             "negative": "apart, looking at camera"},
        ]
        self.options = {
            "global_positive": "Exactly two people here.",
            "global_negative": "blurry, low quality",
        }

    def test_default_text_matches_existing_assembly(self):
        result = module.assemble_multichar(1, 1, self.characters, self.links, self.options)
        self.assertEqual(
            result["positive"],
            "Exactly two people here. There are exactly two people in the scene and no one else. "
            "Cast: Mature Woman; the man. In the center of the scene, Mature Woman — "
            "woman sitting, looking left; and the man — man standing, hand on shoulder. "
            "Mature Woman and the man are looking left, holding hands."
        )
        self.assertEqual(
            result["negative"],
            "blurry, low quality, looking at camera, extra arm, bad hands, apart"
        )
        self.assertIn("## Duplication audit", result["report"])
        self.assertIn("## Length stats", result["report"])

    def test_migrating_old_character_fields_keeps_prompt_text(self):
        structured = []
        for character in self.characters:
            migrated = {key: value for key, value in character.items()
                        if key not in ("positive", "negative")}
            migrated["generic_positive"] = character["positive"]
            migrated["generic_negative"] = character["negative"]
            structured.append(migrated)
        for profile in ("default", "flux2"):
            old = module.assemble_multichar(
                1, 1, self.characters, self.links, self.options, prompt_profile=profile
            )
            new = module.assemble_multichar(
                1, 1, structured, self.links, self.options, prompt_profile=profile
            )
            self.assertEqual((new["positive"], new["negative"]),
                             (old["positive"], old["negative"]))

    def test_structured_character_fields_feed_both_composers(self):
        character = {
            "name": "Mature Woman", "cells": [0],
            "generic_positive": "a woman",
            "looks_positive": "dark hair, tailored coat",
            "pose_action_positive": "leaning forward",
            "generic_negative": "extra person",
            "looks_negative": "red coat",
            "pose_action_negative": "arms crossed",
        }
        self.assertEqual(module._character_prompt(character, "positive"),
                         "a woman, dark hair, tailored coat, leaning forward")
        self.assertEqual(module._character_prompt(character, "negative"),
                         "extra person, red coat, arms crossed")
        result = module.assemble_multichar(
            1, 1, [character], [], {"subject_count_lock": False, "cast_roster": False},
            prompt_profile="flux2"
        )
        self.assertIn("dark hair, tailored coat, leaning forward", result["positive"])
        self.assertIn("extra person, red coat, arms crossed", result["negative"])
        regions = module.RegionalMultiCharConditioning()._regions([character], [], False)
        self.assertEqual(regions[0]["positive"], module._character_prompt(character, "positive"))
        self.assertEqual(regions[0]["negative"], module._character_prompt(character, "negative"))
        merged = module.RegionalMultiCharConditioning()._regions([character], [], True)
        self.assertEqual(merged[0]["positive"], regions[0]["positive"])
        self.assertEqual(merged[0]["negative"], regions[0]["negative"])

    def test_legacy_positive_overrides_mixed_structured_fields(self):
        character = {"positive": "legacy text", "negative": "legacy guard",
                     "generic_positive": "different text", "looks_positive": "red hair",
                     "generic_negative": "different guard"}
        self.assertEqual(module._character_prompt(character, "positive"), "legacy text")
        self.assertEqual(module._character_prompt(character, "negative"), "legacy guard")

    def test_layout_bundle_keeps_legacy_prompt_aliases(self):
        default = json.loads(module.RegionalCharacterLayout.INPUT_TYPES()
                             ["required"]["layout_json"][1]["default"])
        self.assertTrue(default["structured_characters"])
        self.assertEqual(module._character_prompt(default["characters"][0], "positive"),
                         "1girl, solo focus")
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        management = types.ModuleType("comfy.model_management")
        management.intermediate_device = lambda: "cpu"
        management.intermediate_dtype = lambda: "float32"
        comfy.model_management = management
        character = {"name": "A", "cells": [0], "generic_positive": "a woman",
                     "looks_positive": "dark hair", "pose_action_positive": "walking",
                     "generic_negative": "extra limb"}
        raw = json.dumps({"structured_characters": True, "characters": [character], "links": []})
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.model_management": management}), \
                patch.object(module.torch, "zeros", lambda *args, **kwargs: "latent", create=True):
            layout, latent = module.RegionalCharacterLayout().make(
                "rectangle 1216x832", 1, 1, 1, raw
            )
            legacy_layout, _ = module.RegionalCharacterLayout().make(
                "rectangle 1216x832", 1, 1, 1,
                json.dumps({"characters": self.characters[:1], "links": []})
            )
        self.assertEqual(latent["samples"], "latent")
        self.assertEqual(layout["characters"][0]["positive"], "a woman, dark hair, walking")
        self.assertEqual(layout["characters"][0]["negative"], "extra limb")
        self.assertEqual(layout["characters"][0]["looks_positive"], "dark hair")
        self.assertEqual(legacy_layout["characters"], self.characters[:1])

    def test_flux2_trims_redundancy_and_keeps_link_geometry(self):
        result = module.assemble_multichar(
            1, 1, self.characters, self.links, self.options, prompt_profile="flux2"
        )
        positive = result["positive"]
        self.assertNotIn("in the center of the scene", positive.lower())
        self.assertNotIn("There are exactly two people", positive)
        self.assertIn("Cast: the mature woman; the man.", positive)
        self.assertEqual(positive.count("looking left"), 1)
        self.assertIn("The mature woman and the man are holding hands.", positive)
        self.assertIn("Skipped count lock", result["report"])
        self.assertIn("Removed 1 exact interaction clauses", result["report"])

    def test_negative_cap_priority_reports_dropped_terms(self):
        global_first = module.assemble_multichar(
            1, 1, self.characters, self.links, self.options,
            prompt_profile="flux2", negative_term_cap=2
        )
        self.assertEqual(global_first["negative"], "blurry, low quality")
        self.assertIn("Dropped negative terms: looking at camera", global_first["report"])
        link_first = module.assemble_multichar(
            1, 1, self.characters, self.links, self.options,
            prompt_profile="flux2", negative_term_cap=2,
            negative_cap_priority="interaction_first"
        )
        self.assertEqual(link_first["negative"], "apart, looking at camera")
        unlimited = module.assemble_multichar(
            1, 1, self.characters, self.links, self.options,
            prompt_profile="flux2", negative_cap_priority="interaction_first"
        )
        self.assertTrue(unlimited["negative"].startswith("blurry, low quality"))

    def test_interaction_follows_last_member_group(self):
        chars = [
            {"name": "A", "cells": [0], "positive": "a woman"},
            {"name": "B", "cells": [1], "positive": "a man"},
            {"name": "C", "cells": [2], "positive": "a child"},
        ]
        link = [{"between": [1, 2], "positive": "holding hands"}]
        result = module.assemble_multichar(
            3, 1, chars, link, {"subject_count_lock": False, "cast_roster": False},
            prompt_profile="flux2"
        )
        positive = result["positive"]
        self.assertLess(positive.index("B —"), positive.index("A and B are holding hands"))
        self.assertLess(positive.index("A and B are holding hands"), positive.index("C —"))

    def test_role_name_is_exact_in_labeled_format(self):
        result = module.assemble_multichar(
            1, 1, self.characters[:1], [], {"output_format": "labeled", "cast_roster": False},
            prompt_profile="flux2"
        )
        self.assertIn("Mature Woman: woman sitting", result["positive"])

    def test_flux2_cues_can_be_forced_and_sentence_ends_are_clean(self):
        chars = [{"name": "Mature Woman", "cells": [0],
                  "positive": "woman sitting.", "negative": ""}]
        result = module.assemble_multichar(
            1, 1, chars, [{"between": [1], "positive": "smiling."}],
            {"subject_count_lock": False, "cast_roster": False},
            prompt_profile="flux2", spatial_cues="always"
        )
        self.assertIn("In the center of the scene", result["positive"])
        self.assertNotIn("..", result["positive"])
        result = module.assemble_multichar(
            1, 2, [{**chars[0], "cells": [0]}], [],
            {"subject_count_lock": False, "cast_roster": False},
            prompt_profile="flux2", scale_cues="row_based"
        )
        self.assertIn("small and distant", result["positive"])

    def test_layout_override_is_opt_in_and_does_not_change_ports(self):
        original = {"grid_cols": 3, "grid_rows": 1, "characters": self.characters, "links": self.links}
        self.assertIs(module._compose_layout(original, ""), original)
        override = json.dumps({"grid_cols": 1, "grid_rows": 1,
                               "characters": self.characters[:1], "interactions": []})
        parsed = module._compose_layout(original, override)
        self.assertEqual(len(parsed["characters"]), 1)
        self.assertEqual(parsed["links"], [])
        self.assertEqual(module.MultiCharPromptCompose.RETURN_NAMES,
                         ("positive", "negative", "positive_text", "negative_text", "prompt_report"))
        self.assertEqual(list(module.MultiCharPromptCompose.INPUT_TYPES()["optional"])[0], "layout")
        with self.assertRaises(ValueError):
            module._compose_layout(original, "not JSON")

    def test_compose_accepts_old_call_and_pasted_layout(self):
        node = module.MultiCharPromptCompose()
        args = [FakeClip(), "scene", "blur", True, "handle", True, True, True,
                True, "fine", "prose", False, "global_dedup"]
        old_layout = {"grid_cols": 2, "grid_rows": 1,
                      "characters": self.characters, "links": self.links}
        old_result = node.compose(*args, layout=old_layout)
        self.assertIn("There are exactly two people", old_result[2])
        new_result = node.compose(
            *args, layout=old_layout, prompt_profile="flux2",
            layout_json_override=json.dumps({"grid_cols": 1, "grid_rows": 1,
                                             "characters": self.characters[:1], "links": []})
        )
        self.assertIn("the mature woman", new_result[2].lower())
        self.assertNotIn("the man", new_result[2].lower())
        self.assertEqual(new_result[0], new_result[2])
        self.assertEqual(new_result[1], new_result[3])

    def test_preview_extracts_exact_encoder_text_from_existing_report_link(self):
        result = module.assemble_multichar(
            1, 1, self.characters, self.links, self.options, prompt_profile="flux2"
        )
        preview = module.MultiCharPromptPreview().show(result["report"])
        self.assertEqual(preview["ui"]["positive_text"], [result["positive"]])
        self.assertEqual(preview["ui"]["negative_text"], [result["negative"]])
        self.assertEqual(preview["ui"]["has_prompts"], [True])
        self.assertEqual(preview["result"], (result["report"],))
        inputs = module.MultiCharPromptPreview.INPUT_TYPES()
        self.assertEqual(list(inputs["required"]), ["text"])
        self.assertEqual(list(inputs["optional"]), ["positive_text", "negative_text"])

    def test_preview_preserves_empty_prompts_and_direct_inputs(self):
        empty = module.assemble_multichar(1, 1, [], [], {})
        preview = module.MultiCharPromptPreview().show(empty["report"])
        self.assertEqual(preview["ui"]["positive_text"], [""])
        self.assertEqual(preview["ui"]["negative_text"], [""])
        self.assertNotIn("(empty)", empty["report"].split("## FINAL POSITIVE", 1)[1])
        direct = module.MultiCharPromptPreview().show(
            empty["report"], positive_text="**literal** ``` value", negative_text="<tag>"
        )
        self.assertEqual(direct["ui"]["positive_text"], ["**literal** ``` value"])
        self.assertEqual(direct["ui"]["negative_text"], ["<tag>"])
        arbitrary = module.MultiCharPromptPreview().show("# A note")
        self.assertEqual(arbitrary["ui"]["has_prompts"], [False])

    def test_report_link_preserves_markdown_fences_in_prompt_text(self):
        result = module.assemble_multichar(
            1, 1, [], [], {"global_positive": "A scene\n```\nframe\n```",
                           "global_negative": "avoid ``` in image"},
            prompt_profile="flux2"
        )
        preview = module.MultiCharPromptPreview().show(result["report"])
        self.assertEqual(preview["ui"]["positive_text"], [result["positive"]])
        self.assertEqual(preview["ui"]["negative_text"], [result["negative"]])


if __name__ == "__main__":
    unittest.main()
