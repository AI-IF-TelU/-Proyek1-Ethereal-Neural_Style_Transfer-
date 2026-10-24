"""Regresi preprocessing S02-03, kompatibel dengan unittest dan pytest."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.preprocessing import deprocess_image, load_image, preprocess_image


class TestLoadImage(unittest.TestCase):
    """Validasi input, konversi warna, dan resize loader."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_rgb_jpeg_and_png_keep_size(self) -> None:
        for extension in ("jpg", "png"):
            with self.subTest(extension=extension):
                path = self.root / f"gambar.{extension}"
                Image.new("RGB", (9, 7), (20, 40, 60)).save(path)
                result = load_image(str(path))
                self.assertEqual(result.mode, "RGB")
                self.assertEqual(result.size, (9, 7))
                path.unlink()
                self.assertEqual(len(result.tobytes()), 9 * 7 * 3)

    def test_grayscale_to_rgb(self) -> None:
        path = self.root / "abu.png"
        Image.new("L", (3, 2), 80).save(path)
        result = load_image(path)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (80, 80, 80))

    def test_rgba_to_rgb(self) -> None:
        path = self.root / "alpha.png"
        Image.new("RGBA", (3, 2), (20, 40, 60, 0)).save(path)
        result = load_image(path)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (20, 40, 60))

    def test_missing_file(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "tidak ditemukan"):
            load_image(self.root / "hilang.jpg")

    def test_invalid_file(self) -> None:
        path = self.root / "rusak.png"
        path.write_bytes(b"bukan data gambar")
        with self.assertRaisesRegex(ValueError, "bukan gambar"):
            load_image(path)

    def test_unsupported_format_despite_extension(self) -> None:
        path = self.root / "palsu.jpg"
        Image.new("RGB", (3, 2)).save(path, format="BMP")
        with self.assertRaisesRegex(ValueError, "JPEG atau PNG"):
            load_image(path)

    def test_directory_path(self) -> None:
        with self.assertRaises(OSError):
            load_image(self.root)

    def test_resize_not_silently_ignored(self) -> None:
        path = self.root / "gambar.png"
        pixels = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
        image = Image.fromarray(pixels)
        image.save(path)
        result = load_image(path, size=(256, 512))
        self.assertEqual(result.size, (512, 256))
        expected = image.resize((512, 256), Image.Resampling.LANCZOS)
        self.assertEqual(result.tobytes(), expected.tobytes())

    def test_invalid_sizes(self) -> None:
        for size in (0, True, "3,4", [3, 4], (), (3,), (3, 4, 5),
                     (0, 4), (3, -1), (True, 4), (3, False), (3.0, 4)):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    load_image(self.root / "absent.png", size=size)
                with self.assertRaises(ValueError):
                    preprocess_image(Image.new("RGB", (2, 2)), size=size)
                with self.assertRaises(ValueError):
                    preprocess_image(self.root / "absent.png", size=size)

    def test_path_and_pil_preprocessing_match_and_resize_once(self) -> None:
        path = self.root / "gambar.png"
        image = Image.fromarray(
            np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
        )
        image.save(path)
        original_resize = Image.Image.resize
        calls = []

        def resize(instance, *args, **kwargs):
            calls.append(args)
            return original_resize(instance, *args, **kwargs)

        with patch.object(Image.Image, "resize", resize):
            actual = preprocess_image(path, size=(4, 6))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ((6, 4), Image.Resampling.LANCZOS))
        expected = preprocess_image(image, size=(4, 6))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            preprocess_image(str(path), size=None),
            preprocess_image(image, size=None), rtol=0, atol=0,
        )


class TestPreprocessImage(unittest.TestCase):
    """Kontrak shape, dtype, warna, device dan normalisasi numerik."""

    def test_defaults_and_non_square_size(self) -> None:
        image = Image.new("RGB", (9, 7))
        default = preprocess_image(image)
        self.assertEqual(default.shape, (1, 3, 512, 512))
        self.assertEqual(default.dtype, torch.float32)
        self.assertEqual(default.device, torch.device("cpu"))
        self.assertEqual(preprocess_image(image, (256, 512)).shape,
                         (1, 3, 256, 512))
        self.assertEqual(preprocess_image(image, None).shape, (1, 3, 7, 9))
        self.assertEqual(preprocess_image(image, (1, 1)).shape, (1, 3, 1, 1))

    def test_normalization_numeric_and_channel_order(self) -> None:
        pixels = np.array([[[0, 128, 255], [255, 0, 64]]], dtype=np.uint8)
        actual = preprocess_image(Image.fromarray(pixels), size=None)
        expected = ((pixels.astype(np.float32) / 255.0
                     - np.array([0.485, 0.456, 0.406], dtype=np.float32))
                    / np.array([0.229, 0.224, 0.225], dtype=np.float32))
        np.testing.assert_allclose(actual[0].permute(1, 2, 0).numpy(),
                                   expected, rtol=0, atol=1e-6)
        self.assertTrue(torch.isfinite(actual).all().item())

    def test_pil_modes_no_mutation_and_no_redundant_resize(self) -> None:
        for mode, color, rgb in (("L", 80, (80, 80, 80)),
                                 ("RGBA", (20, 40, 60, 0), (20, 40, 60)),
                                 ("RGB", (20, 40, 60), (20, 40, 60))):
            with self.subTest(mode=mode):
                image = Image.new(mode, (9, 7), color)
                before = image.tobytes()
                with patch.object(Image.Image, "resize",
                                  side_effect=AssertionError("double resize")):
                    result = preprocess_image(image, size=(7, 9))
                self.assertEqual(deprocess_image(result).getpixel((0, 0)), rgb)
                preprocess_image(image, size=(4, 6))
                self.assertEqual(image.mode, mode)
                self.assertEqual(image.size, (9, 7))
                self.assertEqual(image.tobytes(), before)

    def test_invalid_image_types_and_empty_dimensions(self) -> None:
        for image in (None, 1, True, [], np.zeros((2, 2, 3)),
                      torch.zeros(3, 2, 2)):
            with self.subTest(image_type=type(image)):
                with self.assertRaises(TypeError):
                    preprocess_image(image)
                with self.assertRaises(TypeError):
                    load_image(image)
        for dimensions in ((0, 2), (2, 0)):
            with self.assertRaises(ValueError):
                preprocess_image(Image.new("RGB", dimensions))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA tidak tersedia")
    def test_cuda_device_and_deprocess(self) -> None:
        result = preprocess_image(Image.new("RGB", (3, 2), (20, 40, 60)),
                                  size=None, device=torch.device("cuda"))
        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(deprocess_image(result).getpixel((0, 0)), (20, 40, 60))


class TestDeprocessImage(unittest.TestCase):
    """Inverse, clamp, validasi dan keamanan tensor input."""

    def test_round_trip_all_pixel_levels_chw_and_bchw(self) -> None:
        pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
        pixels = np.stack((pixels, np.flipud(pixels), np.fliplr(pixels)), axis=2)
        tensor = preprocess_image(Image.fromarray(pixels), size=None)
        for value in (tensor, tensor[0]):
            with self.subTest(shape=value.shape):
                result = deprocess_image(value)
                self.assertEqual(result.mode, "RGB")
                np.testing.assert_array_equal(np.array(result), pixels)

    def test_inverse_numeric_non_square_clamp_and_no_mutation(self) -> None:
        pixels = torch.tensor([-0.5, 0.0, 0.25, 0.5, 1.0, 1.5])
        pixels = pixels.reshape(1, 2, 3).expand(3, 2, 3)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = ((pixels - mean) / std).unsqueeze(0).requires_grad_()
        before = tensor.detach().clone()
        result = deprocess_image(tensor)
        self.assertEqual(result.size, (3, 2))
        expected = np.array([[0, 0, 64], [128, 255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(np.array(result),
                                      np.repeat(expected[:, :, None], 3, axis=2))
        torch.testing.assert_close(tensor, before, rtol=0, atol=0)
        self.assertTrue(tensor.requires_grad)
        self.assertIsNone(tensor.grad)

    def test_noncontiguous_tensor(self) -> None:
        tensor = torch.zeros(3, 4, 7).transpose(1, 2)
        self.assertFalse(tensor.is_contiguous())
        self.assertEqual(deprocess_image(tensor).size, (4, 7))

    def test_invalid_shapes(self) -> None:
        for shape in ((), (3,), (3, 2), (2, 3, 2, 4), (0, 3, 2, 4),
                      (1, 1, 2, 4), (4, 2, 4), (3, 0, 4), (1, 3, 2, 0),
                      (1, 1, 3, 2, 4)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    deprocess_image(torch.zeros(shape))

    def test_invalid_type_dtype_and_nonfinite(self) -> None:
        for value in (None, [], np.zeros((3, 2, 4))):
            with self.assertRaises(TypeError):
                deprocess_image(value)
        for dtype in (torch.int64, torch.bool, torch.complex64):
            with self.assertRaises(ValueError):
                deprocess_image(torch.zeros(3, 2, 4, dtype=dtype))
        for value in (float("nan"), float("inf"), -float("inf")):
            tensor = torch.zeros(1, 3, 2, 4)
            tensor[0, 0, 0, 0] = value
            with self.assertRaises(ValueError):
                deprocess_image(tensor)
