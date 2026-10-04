import unittest
from unittest.mock import patch

from test_compose import module


class ImageTensor:
    def __init__(self, pixels):
        self.pixels = pixels
        self.shape = (len(pixels), 1, 1, len(pixels[0]))

    def __getitem__(self, key):
        self_slice = key[-1]
        return ImageTensor([pixel[self_slice] for pixel in self.pixels])

    def __mul__(self, factor):
        return ImageTensor([[value * factor for value in pixel] for pixel in self.pixels])

    def __add__(self, other):
        if not isinstance(other, ImageTensor):
            return ImageTensor([[value + other for value in pixel] for pixel in self.pixels])
        return ImageTensor([
            [left + right for left, right in zip(left_pixel, right_pixel)]
            for left_pixel, right_pixel in zip(self.pixels, other.pixels)
        ])

    def clamp(self, low, high):
        return ImageTensor([
            [max(low, min(high, value)) for value in pixel] for pixel in self.pixels
        ])


def concatenate(tensors, dim):
    if dim != -1:
        raise AssertionError("Expected channel concatenation")
    return ImageTensor([
        [value for tensor in tensors for value in tensor.pixels[index]]
        for index in range(len(tensors[0].pixels))
    ])


class GrayscaleTests(unittest.TestCase):
    def test_rgb_batch_and_passthrough(self):
        node = module.RegionalGrayscaleFilter()
        image = ImageTensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertIs(node.apply(image, False)[0], image)
        self.assertIs(node.apply(image, grayscale_strength=0.0)[0], image)
        with patch.object(module.torch, "cat", concatenate, create=True):
            output = node.apply(image)[0]
        self.assertEqual(output.shape, image.shape)
        for pixel, expected in zip(output.pixels, (0.2126, 0.7152)):
            for channel in pixel:
                self.assertAlmostEqual(channel, expected)
        self.assertEqual(image.pixels[0], [1.0, 0.0, 0.0])

    def test_alpha_is_preserved(self):
        image = ImageTensor([[0.0, 0.0, 1.0, 0.3]])
        with patch.object(module.torch, "cat", concatenate, create=True):
            output = module.RegionalGrayscaleFilter().apply(image)[0]
        self.assertEqual(output.shape, image.shape)
        for channel in output.pixels[0][:3]:
            self.assertAlmostEqual(channel, 0.0722)
        self.assertEqual(output.pixels[0][3], 0.3)

    def test_strength_and_tone_controls(self):
        node = module.RegionalGrayscaleFilter()
        with patch.object(module.torch, "cat", concatenate, create=True):
            blended = node.apply(ImageTensor([[1.0, 0.0, 0.0]]),
                                 grayscale_strength=0.5)[0]
            lifted = node.apply(ImageTensor([[0.0, 0.0, 0.0],
                                             [1.0, 1.0, 1.0]]), black_lift=0.2)[0]
            softer = node.apply(ImageTensor([[0.0, 0.0, 0.0],
                                             [1.0, 1.0, 1.0]]), contrast=0.5)[0]
            brighter = node.apply(ImageTensor([[0.4, 0.4, 0.4]]), brightness=0.1)[0]
        for actual, expected in zip(blended.pixels[0], (0.6063, 0.1063, 0.1063)):
            self.assertAlmostEqual(actual, expected)
        for channel in lifted.pixels[0]:
            self.assertAlmostEqual(channel, 0.2)
        for channel in lifted.pixels[1]:
            self.assertAlmostEqual(channel, 1.0)
        for channel in softer.pixels[0]:
            self.assertAlmostEqual(channel, 0.25)
        for channel in softer.pixels[1]:
            self.assertAlmostEqual(channel, 0.75)
        for channel in brighter.pixels[0]:
            self.assertAlmostEqual(channel, 0.5)

    def test_registration_and_input_types(self):
        self.assertIs(module.NODE_CLASS_MAPPINGS["RegionalGrayscaleFilter"],
                      module.RegionalGrayscaleFilter)
        inputs = module.RegionalGrayscaleFilter.INPUT_TYPES()["required"]
        self.assertEqual(inputs["image"], ("IMAGE",))
        self.assertTrue(inputs["enabled"][1]["default"])
        optional = module.RegionalGrayscaleFilter.INPUT_TYPES()["optional"]
        self.assertEqual([optional[name][1]["default"] for name in optional],
                         [1.0, 0.0, 1.0, 0.0])
        self.assertTrue(all(config[1]["display"] == "slider" for config in optional.values()))


if __name__ == "__main__":
    unittest.main()
