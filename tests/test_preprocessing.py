"""Tes loader S02-02, dapat dijalankan dengan unittest maupun pytest."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.preprocessing import load_image


class TestLoadImage(unittest.TestCase):
    """Validasi input, konversi warna, dan batas tahap loader."""

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
        with self.assertRaises(NotImplementedError):
            load_image(self.root / "gambar.png", size=(256, 256))
