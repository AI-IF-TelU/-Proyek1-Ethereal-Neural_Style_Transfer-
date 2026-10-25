"""Modul ekstraksi fitur berbasis VGG-19 beku (frozen pretrained backbone).

Mengimplementasikan kontrak Keputusan Arsitektur D-006 (Stage 3: S03-01 & S03-02):
- Memuat blok konvolusi ``vgg19(weights=...).features[:29]`` hingga ``conv5_1`` (indeks ``'28'``).
- Mengganti seluruh ``nn.ReLU(inplace=True)`` menjadi ``nn.ReLU(inplace=False)`` agar aktivasi
  keluaran ``Conv2d`` tidak termutasi secara in-place dan aman bagi grafik autograd PyTorch.
- Membekukan seluruh parameter model (``requires_grad = False``) dan mengunci mode ``.eval()``.
- Mengekstrak representasi fitur konten (``conv4_2``) dan gaya (``conv1_1`` s.d. ``conv5_1``)
  tanpa ``torch.no_grad()`` sehingga gradien tetap mengalir menuju tensor gambar target.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19

DEFAULT_LAYERS: dict[str, str] = {
    "0": "conv1_1",
    "5": "conv2_1",
    "10": "conv3_1",
    "19": "conv4_1",
    "21": "conv4_2",
    "28": "conv5_1",
}

CONTENT_LAYERS: tuple[str, ...] = ("conv4_2",)
STYLE_LAYERS: tuple[str, ...] = (
    "conv1_1",
    "conv2_1",
    "conv3_1",
    "conv4_1",
    "conv5_1",
)
MAX_FEATURE_LAYER_INDEX: int = 28
MIN_SPATIAL_SIZE: int = 16

__all__ = [
    "CONTENT_LAYERS",
    "DEFAULT_LAYERS",
    "MAX_FEATURE_LAYER_INDEX",
    "MIN_SPATIAL_SIZE",
    "STYLE_LAYERS",
    "get_features",
    "get_vgg19_model",
]


def _validate_device(device: torch.device | str) -> torch.device:
    """Memvalidasi dan mengonversi parameter device menjadi ``torch.device``."""
    if isinstance(device, torch.device):
        return device
    if not isinstance(device, str):
        raise TypeError("device harus berupa torch.device atau str")
    if not device.strip():
        raise ValueError("device string tidak boleh kosong")
    try:
        return torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"device tidak valid: {device!r}") from exc


def _validate_image_tensor(image_tensor: torch.Tensor) -> None:
    """Memvalidasi kontrak tensor gambar masukan 4D ``(N, 3, H, W)`` bertipe floating."""
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor harus berupa torch.Tensor")
    if image_tensor.ndim != 4:
        raise ValueError(
            f"image_tensor harus berdimensi 4D (N, 3, H, W), diterima {image_tensor.ndim}D"
        )
    batch_size, channels, height, width = image_tensor.shape
    if batch_size < 1:
        raise ValueError("Dimensi batch image_tensor harus >= 1")
    if channels != 3:
        raise ValueError(
            f"image_tensor harus memiliki 3 channel RGB, diterima {channels} channel"
        )
    if height < MIN_SPATIAL_SIZE or width < MIN_SPATIAL_SIZE:
        raise ValueError(
            f"Ukuran spasial image_tensor minimal ({MIN_SPATIAL_SIZE}, {MIN_SPATIAL_SIZE}) "
            f"untuk melewati 4 tahap MaxPool2d VGG-19, diterima ({height}, {width})"
        )
    if not image_tensor.is_floating_point():
        raise ValueError("image_tensor harus bertipe floating-point (misal torch.float32)")
    if not torch.isfinite(image_tensor).all():
        raise ValueError("image_tensor tidak boleh mengandung nilai NaN atau Inf")


def _validate_layers(
    layers: dict[str, str] | None,
    model: nn.Module,
) -> dict[str, str]:
    """Memvalidasi dictionary pemetaan indeks layer ke nama kanonik fitur."""
    if layers is None:
        target_layers = dict(DEFAULT_LAYERS)
    elif isinstance(layers, dict):
        if not layers:
            raise ValueError("layers tidak boleh berupa dictionary kosong")
        target_layers = {}
        for key, val in layers.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Kunci pada layers harus berupa string non-kosong")
            if not isinstance(val, str) or not val.strip():
                raise ValueError("Nama fitur pada layers harus berupa string non-kosong")
            target_layers[key] = val
        if len(set(target_layers.values())) != len(target_layers):
            raise ValueError("Nama fitur tujuan pada layers harus unik")
    else:
        raise TypeError("layers harus berupa dict[str, str] atau None")

    available_keys = set(model._modules.keys())
    missing_keys = [key for key in target_layers if key not in available_keys]
    if missing_keys:
        raise ValueError(
            f"Indeks layer {missing_keys} tidak ditemukan di dalam model feature extractor"
        )

    return target_layers


def get_vgg19_model(
    device: torch.device | str = "cpu",
    weights: VGG19_Weights | None = VGG19_Weights.DEFAULT,
) -> nn.Sequential:
    """Memuat blok konvolusi VGG-19 hingga ``conv5_1`` dengan bobot beku dan ReLU aman.

    Args:
        device: Perangkat komputasi tujuan (``"cpu"``, ``"cuda"``, atau ``torch.device``).
        weights: Bobot pretrained Torchvision (default ``VGG19_Weights.DEFAULT``).
            Dapat diisi ``None`` untuk pengujian unit test struktur/autograd tanpa unduhan.

    Returns:
        ``nn.Sequential`` berisi 29 layer pertama ``vgg19.features`` (indeks ``'0'`` s.d. ``'28'``)
        dalam mode ``.eval()``, parameter ``requires_grad=False``, dan ``nn.ReLU(inplace=False)``.
    """
    target_device = _validate_device(device)
    if weights is not None and not isinstance(weights, VGG19_Weights):
        raise TypeError("weights harus berupa VGG19_Weights atau None")

    raw_features = vgg19(weights=weights).features
    truncated_layers: list[nn.Module] = []
    for idx in range(MAX_FEATURE_LAYER_INDEX + 1):
        layer = raw_features[idx]
        if isinstance(layer, nn.ReLU):
            layer = nn.ReLU(inplace=False)
        truncated_layers.append(layer)

    model = nn.Sequential(*truncated_layers)
    for param in model.parameters():
        param.requires_grad_(False)

    model.eval()
    return model.to(target_device)


def get_features(
    image_tensor: torch.Tensor,
    model: nn.Module,
    layers: dict[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Mengekstrak peta fitur (feature maps) dari layer VGG-19 yang ditentukan.

    Fungsi ini sengaja tidak membungkus forward pass dengan ``torch.no_grad()`` agar
    ketika ``image_tensor.requires_grad == True`` (gambar hasil yang sedang dioptimasi),
    grafik komputasi autograd tetap terhubung dari loss kembali ke ``image_tensor``.

    Args:
        image_tensor: Tensor gambar 4D ``(N, 3, H, W)`` bertipe floating-point.
        model: Modul ekstraktor fitur VGG-19 (keluaran ``get_vgg19_model``).
        layers: Pemetaan indeks sub-modul pada ``model`` ke nama fitur keluaran.
            Jika ``None``, menggunakan ``DEFAULT_LAYERS``.

    Returns:
        Dictionary yang memetakan nama fitur (misal ``'conv4_2'``) ke tensor 4D.
    """
    _validate_image_tensor(image_tensor)
    if not isinstance(model, nn.Module):
        raise TypeError("model harus berupa instance torch.nn.Module")

    target_layers = _validate_layers(layers, model)

    model_param = next(model.parameters(), None)
    if model_param is not None and model_param.device != image_tensor.device:
        raise ValueError(
            f"Device image_tensor ({image_tensor.device}) tidak sama dengan "
            f"device model ({model_param.device})"
        )

    features: dict[str, torch.Tensor] = {}
    x = image_tensor
    for name, layer in model._modules.items():
        if layer is None:
            continue
        x = layer(x)
        if name in target_layers:
            features[target_layers[name]] = x
            if len(features) == len(target_layers):
                break

    return features
