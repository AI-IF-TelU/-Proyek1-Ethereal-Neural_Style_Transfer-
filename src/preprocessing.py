"""Load, resize (H, W), dan normalisasi gambar RGB untuk VGG-19."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _validate_size(size: tuple[int, int] | None) -> None:
    if size is not None and (
        not isinstance(size, tuple)
        or len(size) != 2
        or any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0
            for v in size
        )
    ):
        raise ValueError("size harus None atau tuple (H, W) integer positif.")


def load_image(
    image_path: str | Path, size: tuple[int, int] | None = None
) -> Image.Image:
    """Baca JPEG/PNG menjadi RGB mandiri; size=None menjaga ukuran asli.

    Resize langsung memakai Lanczos tanpa crop, dengan size=(H, W).
    RGBA dikonversi dengan membuang kanal alpha.
    File sumber ditutup sebelum hasil dikembalikan.

    Raises:
        TypeError: Path bukan str atau Path.
        FileNotFoundError: File tidak ditemukan.
        ValueError: Format bukan JPEG/PNG, file tidak dikenali, size invalid.
        OSError: File tidak dapat diakses atau decoding gagal.
    """
    _validate_size(size)
    if not isinstance(image_path, (str, Path)):
        raise TypeError("image_path harus str atau Path.")

    path = Path(image_path)
    try:
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Format gambar harus JPEG atau PNG.")
            image.load()
            result = image.convert("RGB")
            if size is not None and (result.height, result.width) != size:
                result = result.resize(
                    (size[1], size[0]), Image.Resampling.LANCZOS
                )
            return result
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path}") from error
    except UnidentifiedImageError as error:
        raise ValueError(f"File bukan gambar yang valid: {path}") from error


def preprocess_image(
    image: Image.Image | str | Path,
    size: tuple[int, int] | None = (512, 512),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Konversi PIL/path ke float32 BCHW ImageNet tanpa mengubah input."""
    _validate_size(size)
    if isinstance(image, (str, Path)):
        image = load_image(image, size=size)
    elif isinstance(image, Image.Image):
        if image.width <= 0 or image.height <= 0:
            raise ValueError("Dimensi gambar harus positif.")
        image = image.convert("RGB")
        if size is not None and (image.height, image.width) != size:
            image = image.resize((size[1], size[0]), Image.Resampling.LANCZOS)
    else:
        raise TypeError("image harus PIL.Image, str, atau Path.")

    pixels = np.array(image, dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0).to(device)


def deprocess_image(tensor: torch.Tensor) -> Image.Image:
    """Inverse ImageNet ke RGB uint8; detach dan salin tanpa mutasi input."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor harus torch.Tensor.")
    if (
        tensor.ndim not in (3, 4)
        or (tensor.ndim == 4 and tensor.shape[0] != 1)
        or tensor.shape[-3] != 3
        or tensor.shape[-2] <= 0
        or tensor.shape[-1] <= 0
    ):
        raise ValueError("Shape harus (3, H, W) atau (1, 3, H, W), H/W positif.")
    if not tensor.is_floating_point():
        raise ValueError("Tensor harus bertipe floating point.")
    image = tensor.detach().cpu().clone()
    if not torch.isfinite(image).all().item():
        raise ValueError("Tensor harus berisi nilai finite.")
    if image.ndim == 4:
        image = image.squeeze(0)
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    image = (image * std + mean).clamp(0.0, 1.0)
    pixels = (image * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(pixels)
