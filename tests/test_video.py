import importlib.util
import inspect
import json
import sys
import types
import unittest
from pathlib import Path


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


ORIGINAL_NODES = {
    "RegionalCharacterLayout", "RegionalMultiCharConditioning", "MultiCharPromptCompose",
    "MultiCharPromptPreview", "MultiCharLayoutEnhancer", "RegionalFaceDetailerSwitch",
    "RegionalHiresSwitch", "RegionalGrayscaleFilter", "RegionalSeedLabel",
}


class FrameCountTests(unittest.TestCase):
    def test_snaps_up_to_17k_plus_5(self):
        self.assertEqual(module.video_frame_count(5), 124)
        self.assertEqual(module.video_frame_count(124 / 24), 124)  # already on the grid
        self.assertEqual(module.video_frame_count(1), 39)

    def test_minimum_is_five_frames(self):
        self.assertEqual(module.video_frame_count(0), 5)
        self.assertEqual(module.video_frame_count(0.1), 5)

    def test_every_result_is_on_the_grid(self):
        for tenths in range(3, 400):
            self.assertEqual(module.video_frame_count(tenths / 10) % 17, 5)


class TimelineTests(unittest.TestCase):
    def test_even_split(self):
        text, frames, duration = module.build_video_timeline("a\nb\nc", 5)
        self.assertEqual(frames, 124)
        self.assertAlmostEqual(duration, 124 / 24)
        self.assertEqual(text.splitlines(),
                         ["[0s-1.7s] a", "[1.7s-3.4s] b", "[3.4s-5.2s] c"])

    def test_blank_lines_and_padding_are_ignored(self):
        text, _, _ = module.build_video_timeline("\n  a  \n\n b\n", 5)
        self.assertEqual(text.splitlines(), ["[0s-2.6s] a", "[2.6s-5.2s] b"])

    def test_weights(self):
        text, _, duration = module.build_video_timeline("a\nb\nc", 4, "1,2,1")
        edges = [0.25 * duration, 0.75 * duration, duration]
        want = ["[0s-%.1fs] a" % edges[0], "[%.1fs-%.1fs] b" % (edges[0], edges[1]),
                "[%.1fs-%.1fs] c" % (edges[1], edges[2])]
        self.assertEqual([ln.replace(".0s", "s") for ln in text.splitlines()],
                         [ln.replace(".0s", "s") for ln in want])

    def test_bad_weights_raise(self):
        for bad in ("1,2", "1,x,1", "1,0,1", "1,-1,1"):
            with self.assertRaises(ValueError, msg=bad):
                module.build_video_timeline("a\nb\nc", 5, bad)

    def test_no_shots_gives_empty_text_but_still_a_length(self):
        self.assertEqual(module.build_video_timeline("  \n", 5), ("", 124, 124 / 24))

    def test_many_shots_never_make_an_empty_range(self):
        text, _, _ = module.build_video_timeline("\n".join("s%d" % i for i in range(12)), 1)
        for line in text.splitlines():
            rng = line[1:line.index("]")]
            start, end = rng.split("-")
            self.assertNotEqual(start, end, line)

    def test_last_range_ends_at_the_real_duration(self):
        # 10 s is not on the 17k+5 grid, it becomes 243 frames = 10.125 s
        text, frames, duration = module.build_video_timeline("a\nb", 10)
        self.assertEqual(frames, 243)
        self.assertAlmostEqual(duration, 10.125)
        self.assertEqual(text.splitlines(), ["[0s-5.1s] a", "[5.1s-10.1s] b"])

    def test_node_returns_the_three_outputs(self):
        out = module.MultiCharVideoTimeline().build(5.0, "a\nb")
        self.assertEqual(len(out), len(module.MultiCharVideoTimeline.RETURN_TYPES))
        self.assertEqual(out[1], 124)


class VideoComposeTests(unittest.TestCase):
    def setUp(self):
        self.layout = {
            "grid_cols": 2, "grid_rows": 1,
            "characters": [
                {"name": "barista", "cells": [0], "positive": "a cheerful barista girl at the counter",
                 "negative": "old, ugly"},
                {"name": "boy", "cells": [1], "positive": "a lanky teenage boy typing on a laptop",
                 "negative": ""},
            ],
            "links": [{"between": [1, 2], "positive": "handing a coffee cup across the counter",
                       "negative": ""}],
        }
        self.node = module.MultiCharVideoPromptCompose()

    def build(self, **over):
        args = dict(global_positive="Anime cafe.", timeline="[0s-5.2s] The camera pushes in.",
                    audio="cafe ambience", subject_count_lock=True, use_names="handle",
                    bind_interactions=True, cast_roster=True, order_and_group=True,
                    spatial_detail="fine", negative_handling="to_positive_assertion",
                    layout=self.layout, global_negative="blurry")
        args.update(over)
        return self.node.build(**args)

    def test_has_no_clip_input(self):
        sig = inspect.signature(module.MultiCharVideoPromptCompose.build)
        self.assertNotIn("clip", sig.parameters)
        types_ = module.MultiCharVideoPromptCompose.INPUT_TYPES()
        self.assertNotIn("clip", types_["required"])
        self.assertNotIn("clip", types_.get("optional", {}))
        self.assertEqual(module.MultiCharVideoPromptCompose.RETURN_TYPES, ("STRING",) * 3)

    def test_section_order(self):
        prompt, _, _ = self.build(rules="Hard cuts only.", end_notes="No subtitles.")
        order = [prompt.index(s) for s in
                 ("Anime cafe.", "exactly two people", "Timeline:\n[0s-5.2s]",
                  "Hard cuts only.", "Audio: cafe ambience", "No subtitles.")]
        self.assertEqual(order, sorted(order))
        self.assertEqual(prompt.count("Timeline:"), 1)
        self.assertEqual(prompt.count("Audio:"), 1)

    def test_cast_block_names_the_characters_and_the_interaction(self):
        prompt, _, _ = self.build()
        self.assertIn("barista", prompt)
        self.assertIn("handing a coffee cup", prompt)

    def test_headers_are_not_doubled(self):
        prompt, _, _ = self.build(timeline="Timeline:\n[0s-5s] x", audio="Audio: y")
        self.assertEqual(prompt.count("Timeline:"), 1)
        self.assertEqual(prompt.count("Audio:"), 1)

    def test_empty_timeline_and_audio_add_no_headers(self):
        prompt, _, _ = self.build(timeline="", audio="")
        self.assertNotIn("Timeline:", prompt)
        self.assertNotIn("Audio:", prompt)

    def test_works_without_a_layout(self):
        prompt, _, _ = self.build(layout=None)
        self.assertTrue(prompt.startswith("Anime cafe."))
        self.assertIn("Timeline:", prompt)

    def test_negative_to_positive_assertion(self):
        prompt, neg, _ = self.build()
        self.assertIn("young", prompt)  # 'old' becomes a positive counter-trait
        self.assertNotIn("Avoid:", prompt)
        self.assertEqual(neg, "")

    def test_negative_drop(self):
        prompt, neg, report = self.build(negative_handling="drop")
        self.assertNotIn("Avoid:", prompt)
        self.assertEqual(neg, "")
        self.assertIn("not used", report)

    def test_negative_avoid_sentence(self):
        prompt, neg, _ = self.build(negative_handling="avoid_sentence")
        self.assertIn("Avoid: ", prompt)
        self.assertIn("blurry", neg)
        self.assertTrue(prompt.rstrip().endswith("."))

    def test_report_round_trips_through_the_preview_parser(self):
        prompt, neg, report = self.build(negative_handling="avoid_sentence")
        self.assertEqual(module._prompts_from_report(report), (prompt, neg))

    def test_survives_json_layout_from_the_editor(self):
        layout = json.loads(json.dumps(self.layout))
        prompt, _, _ = self.build(layout=layout, prompt_profile="default")
        self.assertIn("Timeline:", prompt)


class BackwardCompatTests(unittest.TestCase):
    def test_original_nodes_are_still_registered(self):
        self.assertTrue(ORIGINAL_NODES <= set(module.NODE_CLASS_MAPPINGS))

    def test_new_nodes_are_registered_with_display_names(self):
        for name in ("MultiCharVideoTimeline", "MultiCharVideoPromptCompose"):
            self.assertIn(name, module.NODE_CLASS_MAPPINGS)
            self.assertIn(name, module.NODE_DISPLAY_NAME_MAPPINGS)

    def test_every_node_has_a_display_name(self):
        self.assertEqual(set(module.NODE_CLASS_MAPPINGS), set(module.NODE_DISPLAY_NAME_MAPPINGS))

    def test_compose_signature_is_unchanged(self):
        sig = list(inspect.signature(module.MultiCharPromptCompose.compose).parameters)
        self.assertEqual(sig[:3], ["self", "clip", "global_positive"])
        self.assertEqual(list(module.MultiCharPromptCompose.RETURN_NAMES),
                         ["positive", "negative", "positive_text", "negative_text", "prompt_report"])


if __name__ == "__main__":
    unittest.main()
