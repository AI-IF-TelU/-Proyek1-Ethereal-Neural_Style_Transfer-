"""Modul loop optimasi piksel gambar (Optimization Loop) untuk Neural Style Transfer.

Mengimplementasikan kontrak Keputusan Arsitektur D-008 (Stage 4: S04-04 & S04-05):
- clamp_imagenet_tensor_: Menjaga nilai piksel tensor tetap berada di dalam rentang RGB [0, 1]
  pada ruang ternormalisasi ImageNet secara in-place tanpa memutus grafik autograd.
- run_style_transfer: Menjalankan perulangan optimasi piksel gambar (L-BFGS / Adam) dengan
  prekomputasi target fitur konten dan Gram Matrix gaya sekali di awal untuk efisiensi komputasi.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
from src.feature_extractor import DEFAULT_LAYERS, get_features
from src.losses import (
    DEFAULT_STYLE_WEIGHTS,
    compute_content_loss,
    compute_style_loss,
    compute_total_loss,
    gram_matrix,
)
from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD

__all__ = [
    "clamp_imagenet_tensor_",
    "run_style_transfer",
]

# Batas per-channel di ruang ternormalisasi ImageNet: (val - mean) / std
# untuk nilai RGB asli [0.0, 1.0]
_IMAGENET_MIN: tuple[float, float, float] = tuple(
    (0.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)
)
_IMAGENET_MAX: tuple[float, float, float] = tuple(
    (1.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)
)


def clamp_imagenet_tensor_(tensor: torch.Tensor) -> torch.Tensor:
    """Melakukan in-place clamping per-channel pada tensor gambar ternormalisasi ImageNet.

    Menjaga nilai piksel tetap ekuivalen dengan rentang RGB [0.0, 1.0] tanpa memotong
    nilai negatif yang sah di ruang ImageNet (D-008).

    Args:
        tensor: Tensor gambar 4D (B, 3, H, W) atau 3D (3, H, W) bertipe floating-point.

    Returns:
        Tensor yang sama setelah dilakukan clamping secara in-place.

    Raises:
        TypeError: Jika input bukan merupakan torch.Tensor.
        ValueError: Jika tipe bukan floating-point, memuat NaN/Inf, atau dimensi kanal != 3.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor harus berupa torch.Tensor, diterima {type(tensor).__name__}")

    if not tensor.is_floating_point():
        raise ValueError("tensor harus bertipe floating-point")

    if not torch.isfinite(tensor).all():
        raise ValueError("tensor tidak boleh memuat nilai NaN atau Inf")

    if tensor.ndim == 4:
        if tensor.shape[1] != 3:
            raise ValueError(f"Dimensi kanal tensor 4D harus 3, diterima shape {tuple(tensor.shape)}")
    elif tensor.ndim == 3:
        if tensor.shape[0] != 3:
            raise ValueError(f"Dimensi kanal tensor 3D harus 3, diterima shape {tuple(tensor.shape)}")
    else:
        raise ValueError(f"tensor harus berdimensi 3D atau 4D, diterima {tensor.ndim}D")

    with torch.no_grad():
        if tensor.ndim == 4:
            for c in range(3):
                tensor[:, c, :, :].clamp_(min=_IMAGENET_MIN[c], max=_IMAGENET_MAX[c])
        else:
            for c in range(3):
                tensor[c, :, :].clamp_(min=_IMAGENET_MIN[c], max=_IMAGENET_MAX[c])

    return tensor


def run_style_transfer(
    content_tensor: torch.Tensor,
    style_tensor: torch.Tensor,
    model: nn.Module,
    num_steps: int = 300,
    alpha: float = 1.0,
    beta: float = 1e4,
    optimizer_type: str = "lbfgs",
    learning_rate: float = 1.0,
    init_mode: str = "content",
    style_weights: Mapping[str, float] | None = None,
    content_layer: str = "conv4_2",
    seed: int = 42,
) -> tuple[torch.Tensor, list[float]]:
    """Menjalankan loop optimasi Neural Style Transfer pada piksel gambar (D-008).

    Args:
        content_tensor: Tensor gambar konten 4D (B, 3, H, W) ternormalisasi ImageNet.
        style_tensor: Tensor gambar gaya 4D (B, 3, H, W) ternormalisasi ImageNet.
        model: VGG-19 feature extractor beku (eval mode, frozen parameters).
        num_steps: Jumlah iterasi optimasi (>= 1, default 300).
        alpha: Bobot kerugian konten (>= 0, default 1.0).
        beta: Bobot kerugian gaya (>= 0, default 10000.0).
        optimizer_type: Algoritma optimasi, 'lbfgs' atau 'adam' (case-insensitive).
        learning_rate: Laju pembelajaran optimizer (> 0, default 1.0).
        init_mode: Mode inisialisasi piksel gambar, 'content', 'style', atau 'random'.
        style_weights: Kamus bobot gaya per layer (default DEFAULT_STYLE_WEIGHTS).
        content_layer: Nama layer untuk content loss (default 'conv4_2').
        seed: Bilangan acak untuk reprodusibilitas (khusus init_mode='random').

    Returns:
        tuple (generated_tensor, loss_history):
            generated_tensor: Tensor gambar hasil 4D (B, 3, H, W) ter-detach.
            loss_history: Daftar nilai total loss float per iterasi.

    Raises:
        TypeError: Jika tipe parameter tidak sesuai.
        ValueError: Jika argumen melanggar batas numerik, dimensi, device, atau pilihan mode.
    """
    # 1. Validasi Tensor Masukan
    if not isinstance(content_tensor, torch.Tensor):
        raise TypeError(f"content_tensor harus berupa torch.Tensor, diterima {type(content_tensor).__name__}")
    if not isinstance(style_tensor, torch.Tensor):
        raise TypeError(f"style_tensor harus berupa torch.Tensor, diterima {type(style_tensor).__name__}")
    if not isinstance(model, nn.Module):
        raise TypeError(f"model harus berupa torch.nn.Module, diterima {type(model).__name__}")

    if content_tensor.ndim != 4 or content_tensor.shape[1] != 3:
        raise ValueError(f"content_tensor harus berdimensi 4D (B, 3, H, W), diterima {tuple(content_tensor.shape)}")
    if style_tensor.ndim != 4 or style_tensor.shape[1] != 3:
        raise ValueError(f"style_tensor harus berdimensi 4D (B, 3, H, W), diterima {tuple(style_tensor.shape)}")

    if not content_tensor.is_floating_point() or not style_tensor.is_floating_point():
        raise ValueError("content_tensor dan style_tensor harus bertipe floating-point")

    if not torch.isfinite(content_tensor).all():
        raise ValueError("content_tensor tidak boleh mengandung NaN atau Inf")
    if not torch.isfinite(style_tensor).all():
        raise ValueError("style_tensor tidak boleh mengandung NaN atau Inf")

    if content_tensor.shape[0] != style_tensor.shape[0]:
        raise ValueError(
            f"Batch size content_tensor ({content_tensor.shape[0]}) harus sama dengan "
            f"style_tensor ({style_tensor.shape[0]})"
        )

    if content_tensor.shape[2] < 16 or content_tensor.shape[3] < 16:
        raise ValueError(
            f"Dimensi spasial content_tensor minimal 16x16, diterima {content_tensor.shape[2]}x{content_tensor.shape[3]}"
        )
    if style_tensor.shape[2] < 16 or style_tensor.shape[3] < 16:
        raise ValueError(
            f"Dimensi spasial style_tensor minimal 16x16, diterima {style_tensor.shape[2]}x{style_tensor.shape[3]}"
        )

    if content_tensor.device != style_tensor.device:
        raise ValueError(
            f"Device content_tensor ({content_tensor.device}) harus sama dengan style_tensor ({style_tensor.device})"
        )

    # Validasi kesesuaian device model dengan content_tensor
    try:
        model_device = next(model.parameters()).device
        if model_device != content_tensor.device:
            raise ValueError(f"Device model ({model_device}) harus sama dengan tensor ({content_tensor.device})")
    except StopIteration:
        pass

    # 2. Validasi Parameter Numerik & Mode
    if isinstance(num_steps, bool) or not isinstance(num_steps, int):
        raise TypeError(f"num_steps harus bertipe int, diterima {type(num_steps).__name__}")
    if num_steps < 1:
        raise ValueError(f"num_steps harus >= 1, diterima {num_steps}")

    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise TypeError(f"learning_rate harus bertipe int atau float, diterima {type(learning_rate).__name__}")
    if math.isnan(learning_rate) or math.isinf(learning_rate) or learning_rate <= 0:
        raise ValueError(f"learning_rate harus berupa numerik finite > 0, diterima {learning_rate}")

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha harus bertipe int atau float, diterima {type(alpha).__name__}")
    if math.isnan(alpha) or math.isinf(alpha) or alpha < 0:
        raise ValueError(f"alpha harus berupa numerik finite >= 0, diterima {alpha}")

    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError(f"beta harus bertipe int atau float, diterima {type(beta).__name__}")
    if math.isnan(beta) or math.isinf(beta) or beta < 0:
        raise ValueError(f"beta harus berupa numerik finite >= 0, diterima {beta}")

    opt_type_lower = optimizer_type.lower()
    if opt_type_lower not in ("lbfgs", "adam"):
        raise ValueError(f"optimizer_type harus 'lbfgs' atau 'adam', diterima '{optimizer_type}'")

    mode_lower = init_mode.lower()
    if mode_lower not in ("content", "style", "random"):
        raise ValueError(f"init_mode harus 'content', 'style', atau 'random', diterima '{init_mode}'")

    if mode_lower == "style" and (content_tensor.shape[2:] != style_tensor.shape[2:]):
        raise ValueError(
            f"Ukuran spasial style_tensor {tuple(style_tensor.shape[2:])} harus sama dengan "
            f"content_tensor {tuple(content_tensor.shape[2:])} untuk init_mode='style'"
        )

    active_style_weights: Mapping[str, float] = (
        DEFAULT_STYLE_WEIGHTS if style_weights is None else style_weights
    )

    # 3. Prekomputasi Target Konten & Target Gaya Sekali di Awal (D-008.2)
    required_layers = set(active_style_weights.keys())
    required_layers.add(content_layer)

    with torch.no_grad():
        content_features_dict = get_features(content_tensor, model, layers=DEFAULT_LAYERS)
        if content_layer not in content_features_dict:
            raise ValueError(f"content_layer '{content_layer}' tidak ditemukan pada output model")
        target_content_feat = content_features_dict[content_layer].detach()

        style_features_dict = get_features(style_tensor, model, layers=DEFAULT_LAYERS)
        target_style_grams: dict[str, torch.Tensor] = {}
        for layer_name in active_style_weights.keys():
            if layer_name not in style_features_dict:
                raise ValueError(f"Layer gaya '{layer_name}' tidak ditemukan pada output model")
            target_style_grams[layer_name] = gram_matrix(style_features_dict[layer_name].detach())

    # 4. Inisialisasi Piksel Citra yang Dioptimasi
    if mode_lower == "content":
        generated = content_tensor.detach().clone()
    elif mode_lower == "style":
        generated = style_tensor.detach().clone()
    else:  # mode_lower == "random"
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rand_rgb = torch.rand(content_tensor.shape, generator=generator, dtype=content_tensor.dtype)
        mean_t = torch.tensor(IMAGENET_MEAN, dtype=content_tensor.dtype).view(1, 3, 1, 1)
        std_t = torch.tensor(IMAGENET_STD, dtype=content_tensor.dtype).view(1, 3, 1, 1)
        generated = ((rand_rgb - mean_t) / std_t).to(content_tensor.device)

    # Pastikan in-place clamp awal
    clamp_imagenet_tensor_(generated)
    generated.requires_grad_(True)

    # 5. Konfigurasi Optimizer
    optimizer: torch.optim.Optimizer
    if opt_type_lower == "lbfgs":
        optimizer = torch.optim.LBFGS([generated], lr=learning_rate, max_iter=1)
    else:
        optimizer = torch.optim.Adam([generated], lr=learning_rate)

    loss_history: list[float] = []

    # 6. Perulangan Optimasi (Optimization Loop)
    for _ in range(num_steps):
        if opt_type_lower == "lbfgs":
            current_loss_val = 0.0

            def closure() -> torch.Tensor:
                nonlocal current_loss_val
                optimizer.zero_grad()
                gen_features = get_features(generated, model, layers=DEFAULT_LAYERS)
                c_loss = compute_content_loss(gen_features[content_layer], target_content_feat)
                s_loss = compute_style_loss(gen_features, target_style_grams, layer_weights=active_style_weights)
                t_loss = compute_total_loss(c_loss, s_loss, alpha=alpha, beta=beta)
                t_loss.backward()
                current_loss_val = float(t_loss.detach().item())
                return t_loss

            optimizer.step(closure)
            clamp_imagenet_tensor_(generated)
            loss_history.append(current_loss_val)
        else:  # opt_type_lower == "adam"
            optimizer.zero_grad()
            gen_features = get_features(generated, model, layers=DEFAULT_LAYERS)
            c_loss = compute_content_loss(gen_features[content_layer], target_content_feat)
            s_loss = compute_style_loss(gen_features, target_style_grams, layer_weights=active_style_weights)
            t_loss = compute_total_loss(c_loss, s_loss, alpha=alpha, beta=beta)
            t_loss.backward()
            optimizer.step()
            clamp_imagenet_tensor_(generated)
            loss_history.append(float(t_loss.detach().item()))

    return generated.detach(), loss_history
