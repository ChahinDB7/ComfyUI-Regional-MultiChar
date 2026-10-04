import unittest
from unittest.mock import patch

from PIL import Image, ImageChops

from test_compose import module

try:
    import numpy as np
except ImportError:
    np = None


class SeedLabelTests(unittest.TestCase):
    def test_registration_and_defaults(self):
        self.assertIs(module.NODE_CLASS_MAPPINGS["RegionalSeedLabel"], module.RegionalSeedLabel)
        inputs = module.RegionalSeedLabel.INPUT_TYPES()
        self.assertEqual(inputs["required"]["seed"][0], "INT")
        self.assertEqual(inputs["required"]["optimize"][1]["default"], "none")
        self.assertEqual(inputs["required"]["position"][1]["default"], "bottom_right")

    def test_disabled_returns_original_object(self):
        image = object()
        self.assertIs(module.RegionalSeedLabel().apply(image, 42, enabled=False)[0], image)

    def test_grayscale_style_changes_only_label_area_and_keeps_rgb_equal(self):
        source = Image.new("RGB", (320, 180), (127, 127, 127))
        output = module._render_seed_label(source, 42, "bottom_right", "grayscale",
                                           "sans", "", 28, "#FF0000", True,
                                           "#0000FF", 2, 24, "Seed: ", 50, 50)
        changed = ImageChops.difference(source, output)
        self.assertIsNotNone(changed.getbbox())
        self.assertGreater(changed.getbbox()[0], 100)
        self.assertGreater(changed.getbbox()[1], 90)
        self.assertEqual(output.getchannel("R").tobytes(), output.getchannel("G").tobytes())
        self.assertEqual(output.getchannel("G").tobytes(), output.getchannel("B").tobytes())
        self.assertEqual(source.getpixel((200, 130)), (127, 127, 127))

    def test_seed_digits_change_rendered_image(self):
        source = Image.new("RGB", (320, 180), (127, 127, 127))
        common = ("bottom_right", "grayscale", "sans", "", 28,
                  "#FFFFFF", True, "#000000", 2, 24, "Seed: ", 50, 50)
        first = module._render_seed_label(source, 42, *common)
        second = module._render_seed_label(source, 43, *common)
        self.assertIsNotNone(ImageChops.difference(first, second).getbbox())

    def test_custom_position_and_photo_plate(self):
        source = Image.new("RGB", (320, 180), (255, 255, 255))
        output = module._render_seed_label(source, 123, "custom", "photo_realistic",
                                           "mono", "", 24, "#00FF00", False,
                                           "#000000", 2, 10, "Seed: ", 0, 0)
        changed = ImageChops.difference(source, output)
        self.assertIsNotNone(changed.getbbox())
        self.assertLess(changed.getbbox()[0], 30)
        self.assertLess(changed.getbbox()[1], 30)
        self.assertEqual(source.getpixel((300, 170)), output.getpixel((300, 170)))

    @unittest.skipIf(np is None, "NumPy is unavailable")
    def test_batch_keeps_shape_and_grayscale(self):
        class FakeTensor:
            def __init__(self, data):
                self.data = np.asarray(data)
                self.shape = self.data.shape
                self.ndim = self.data.ndim
                self.device = "cpu"
                self.dtype = self.data.dtype

            def __iter__(self):
                return (FakeTensor(item) for item in self.data)

            def detach(self):
                return self

            def clamp(self, low, high):
                return FakeTensor(np.clip(self.data, low, high))

            def __mul__(self, value):
                return FakeTensor(self.data * value)

            def __truediv__(self, value):
                return FakeTensor(self.data / value)

            def round(self):
                return FakeTensor(np.round(self.data))

            def byte(self):
                return FakeTensor(self.data.astype(np.uint8))

            def cpu(self):
                return self

            def numpy(self):
                return self.data

            def to(self, device, dtype):
                return FakeTensor(self.data.astype(dtype))

        source = FakeTensor(np.full((2, 180, 320, 3), 0.5, dtype=np.float32))
        rgba_pixels = np.full((1, 180, 320, 4), 0.5, dtype=np.float32)
        rgba_pixels[..., 3] = 0.25
        rgba_source = FakeTensor(rgba_pixels)
        with patch.object(module.torch, "from_numpy", lambda array: FakeTensor(array), create=True), \
             patch.object(module.torch, "stack", lambda items, dim: FakeTensor(
                 np.stack([item.data for item in items], axis=dim)), create=True):
            output = module.RegionalSeedLabel().apply(source, 42, optimize="grayscale")[0]
            rgba_output = module.RegionalSeedLabel().apply(rgba_source, 42)[0]
        self.assertEqual(output.shape, source.shape)
        self.assertTrue(np.allclose(output.data[..., 0], output.data[..., 1]))
        self.assertTrue(np.allclose(output.data[..., 1], output.data[..., 2]))
        self.assertTrue(np.any(output.data != source.data))
        self.assertEqual(rgba_output.shape, rgba_source.shape)
        self.assertAlmostEqual(float(rgba_output.data[0, 0, 0, 3]), 0.25, places=2)
        self.assertGreater(float(rgba_output.data[0, 130, 240, 3]), 0.25)


if __name__ == "__main__":
    unittest.main()
