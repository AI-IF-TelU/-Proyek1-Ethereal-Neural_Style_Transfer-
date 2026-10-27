"""Modul fungsi kerugian (Loss Functions) untuk Neural Style Transfer.

Mengimplementasikan kontrak Keputusan Arsitektur dan Dokumen Perancangan (Stage 4: S04-01 s.d. S04-03):
- gram_matrix: Menghitung korelasi feature channel G = F @ F.T dinormalisasi dengan (C * H * W).
- compute_content_loss: Menghitung Mean Squared Error representasi konten pada layer target.
- compute_style_loss: Menghitung akumulasi loss bobot Gram Matrix multi-layer.
- compute_total_loss: Mengombinasikan content loss dan style loss dengan pembobotan alpha dan beta.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F

__all__ = [
    "DEFAULT_STYLE_WEIGHTS",
    "gram_matrix",
    "compute_content_loss",
    "compute_style_loss",
    "compute_total_loss",
]

# Bobot default multi-layer untuk ekstraksi gaya (D-007)
DEFAULT_STYLE_WEIGHTS: dict[str, float] = {
    "conv1_1": 0.2,
    "conv2_1": 0.2,
    "conv3_1": 0.2,
    "conv4_1": 0.2,
    "conv5_1": 0.2,
}


def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Menghitung Gram Matrix dari feature tensor untuk ekstraksi gaya (style).

    Mengukur korelasi antar-channel fitur:
        G = (F @ F.T) / (C * H * W)

    Args:
        tensor: Tensor fitur 4D berukuran (B, C, H, W) atau 3D (C, H, W)
            bertipe floating-point (misal torch.float32).

    Returns:
        Tensor Gram Matrix berukuran (B, C, C) bertipe floating-point.

    Raises:
        TypeError: Jika input bukan merupakan torch.Tensor.
        ValueError: Jika tensor bukan floating-point, mengandung NaN/Inf,
            dimensi spasial/batch < 1, atau bukan 3D/4D.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor harus berupa torch.Tensor, diterima {type(tensor).__name__}")

    if not tensor.is_floating_point():
        raise ValueError("tensor harus bertipe floating-point (misal torch.float32)")

    if not torch.isfinite(tensor).all():
        raise ValueError("tensor tidak boleh mengandung nilai NaN atau Inf")

    # Standardisasi ke 4D (B, C, H, W) jika input 3D (C, H, W)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 4:
        raise ValueError(
            f"tensor harus berdimensi 4D (B, C, H, W) atau 3D (C, H, W), diterima {tensor.ndim}D"
        )

    batch_size, channels, height, width = tensor.shape
    if batch_size < 1 or channels < 1 or height < 1 or width < 1:
        raise ValueError(
            f"Setiap dimensi tensor harus >= 1, diterima shape {tuple(tensor.shape)}"
        )

    # Reshape spasial: (B, C, H * W) menggunakan reshape agar tahan non-contiguous (D-007)
    features = tensor.reshape(batch_size, channels, height * width)

    # Batch Matrix Multiplication: (B, C, H*W) @ (B, H*W, C) -> (B, C, C)
    gram = torch.bmm(features, features.transpose(1, 2))

    # Normalisasi sesuai kontrak ARCHITECTURE.md (C * H * W)
    normalization_factor = channels * height * width
    return gram / normalization_factor


def compute_content_loss(
    gen_features: torch.Tensor,
    content_features: torch.Tensor,
) -> torch.Tensor:
    """Menghitung Content Loss antara representasi citra hasil dan citra konten.

    Formula (D-007):
        L_content = F.mse_loss(gen_features, content_features.detach())

    Args:
        gen_features: Tensor fitur citra yang dioptimasi berdimensi 4D (B, C, H, W).
        content_features: Tensor fitur citra konten referensi berdimensi 4D (B, C, H, W).

    Returns:
        Tensor skalar floating-point bernilai loss konten.

    Raises:
        TypeError: Jika gen_features atau content_features bukan torch.Tensor.
        ValueError: Jika tipe bukan floating-point, mengandung NaN/Inf, dimensi bukan 4D,
            dimensi < 1, atau shape dan device tidak identik.
    """
    if not isinstance(gen_features, torch.Tensor):
        raise TypeError(
            f"gen_features harus berupa torch.Tensor, diterima {type(gen_features).__name__}"
        )
    if not isinstance(content_features, torch.Tensor):
        raise TypeError(
            f"content_features harus berupa torch.Tensor, diterima {type(content_features).__name__}"
        )

    if not gen_features.is_floating_point() or not content_features.is_floating_point():
        raise ValueError("gen_features dan content_features harus bertipe floating-point")

    if not torch.isfinite(gen_features).all():
        raise ValueError("gen_features tidak boleh mengandung nilai NaN atau Inf")
    if not torch.isfinite(content_features).all():
        raise ValueError("content_features tidak boleh mengandung nilai NaN atau Inf")

    if gen_features.ndim != 4:
        raise ValueError(
            f"gen_features harus berdimensi 4D (B, C, H, W), diterima {gen_features.ndim}D"
        )
    if content_features.ndim != 4:
        raise ValueError(
            f"content_features harus berdimensi 4D (B, C, H, W), diterima {content_features.ndim}D"
        )

    if any(dim < 1 for dim in gen_features.shape) or any(dim < 1 for dim in content_features.shape):
        raise ValueError("Setiap dimensi tensor gen_features dan content_features harus >= 1")

    if gen_features.shape != content_features.shape:
        raise ValueError(
            f"Shape gen_features {tuple(gen_features.shape)} harus identik dengan "
            f"content_features {tuple(content_features.shape)}"
        )

    if gen_features.device != content_features.device:
        raise ValueError(
            f"Device gen_features ({gen_features.device}) harus identik dengan "
            f"content_features ({content_features.device})"
        )

    # Detach content_features untuk memastikan gradien hanya mengalir ke gen_features
    return F.mse_loss(gen_features, content_features.detach())


def compute_style_loss(
    gen_features_dict: Mapping[str, torch.Tensor],
    style_gram_dict: Mapping[str, torch.Tensor],
    layer_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Menghitung akumulasi Style Loss tertimbang di berbagai layer VGG.

    Formula (D-007):
        L_style = sum(w_l * F.mse_loss(G^l(gen), A^l(style).detach()))

    Args:
        gen_features_dict: Kamus pemetaan nama layer ke tensor fitur hasil (4D).
        style_gram_dict: Kamus pemetaan nama layer ke tensor target gaya.
            Dapat berupa Gram Matrix 3D (B, C, C) atau peta fitur 4D (B, C, H, W).
        layer_weights: Bobot bobot w_l per layer. Jika None, menggunakan
            DEFAULT_STYLE_WEIGHTS.

    Returns:
        Tensor skalar floating-point bernilai style loss tertimbang.

    Raises:
        TypeError: Jika argumen dictionary atau bobot bukan tipe yang diharapkan.
        ValueError: Jika dictionary kosong, ada layer wajib yang tidak ditemukan,
            bobot negatif/non-finite, atau shape target tidak sesuai.
    """
    if not isinstance(gen_features_dict, Mapping):
        raise TypeError(
            f"gen_features_dict harus bertipe Mapping/dict, diterima {type(gen_features_dict).__name__}"
        )
    if not isinstance(style_gram_dict, Mapping):
        raise TypeError(
            f"style_gram_dict harus bertipe Mapping/dict, diterima {type(style_gram_dict).__name__}"
        )

    if len(gen_features_dict) == 0:
        raise ValueError("gen_features_dict tidak boleh kosong")
    if len(style_gram_dict) == 0:
        raise ValueError("style_gram_dict tidak boleh kosong")

    if layer_weights is None:
        active_weights: Mapping[str, float] = DEFAULT_STYLE_WEIGHTS
    elif not isinstance(layer_weights, Mapping):
        raise TypeError(
            f"layer_weights harus bertipe Mapping/dict atau None, diterima {type(layer_weights).__name__}"
        )
    else:
        active_weights = layer_weights

    if len(active_weights) == 0:
        raise ValueError("layer_weights tidak boleh kosong")

    # Inisialisasi total style loss
    first_layer_name = next(iter(active_weights.keys()))
    if first_layer_name not in gen_features_dict:
        raise ValueError(f"Layer '{first_layer_name}' tidak ditemukan pada gen_features_dict")
    sample_tensor = gen_features_dict[first_layer_name]
    if not isinstance(sample_tensor, torch.Tensor):
        raise TypeError(f"Fitur pada layer '{first_layer_name}' harus berupa torch.Tensor")

    total_style_loss = torch.tensor(0.0, dtype=sample_tensor.dtype, device=sample_tensor.device)

    for layer_name, weight in active_weights.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError(
                f"Bobot layer '{layer_name}' harus berupa int atau float, diterima {type(weight).__name__}"
            )
        if math.isnan(weight) or math.isinf(weight) or weight < 0:
            raise ValueError(
                f"Bobot layer '{layer_name}' harus bernilai numerik riil finite >= 0, diterima {weight}"
            )

        if layer_name not in gen_features_dict:
            raise ValueError(f"Layer '{layer_name}' tidak ditemukan pada gen_features_dict")
        if layer_name not in style_gram_dict:
            raise ValueError(f"Layer '{layer_name}' tidak ditemukan pada style_gram_dict")

        gen_feat = gen_features_dict[layer_name]
        target_entry = style_gram_dict[layer_name]

        if not isinstance(gen_feat, torch.Tensor):
            raise TypeError(f"Fitur gen pada layer '{layer_name}' harus berupa torch.Tensor")
        if not isinstance(target_entry, torch.Tensor):
            raise TypeError(f"Target gaya pada layer '{layer_name}' harus berupa torch.Tensor")

        # Hitung Gram Matrix dari gen_features
        gen_gram = gram_matrix(gen_feat)

        # Proses target style (bisa precomputed 3D atau raw feature 4D)
        target_detached = target_entry.detach()
        if target_detached.ndim == 4:
            target_gram = gram_matrix(target_detached)
        elif target_detached.ndim == 3:
            if not target_detached.is_floating_point():
                raise ValueError(f"Target gram pada layer '{layer_name}' harus floating-point")
            if not torch.isfinite(target_detached).all():
                raise ValueError(f"Target gram pada layer '{layer_name}' tidak boleh memuat NaN/Inf")
            if target_detached.shape != gen_gram.shape:
                raise ValueError(
                    f"Shape target gram {tuple(target_detached.shape)} pada layer '{layer_name}' "
                    f"tidak cocok dengan gen_gram {tuple(gen_gram.shape)}"
                )
            if target_detached.device != gen_gram.device:
                raise ValueError(
                    f"Device target gram ({target_detached.device}) pada layer '{layer_name}' "
                    f"tidak cocok dengan gen_gram ({gen_gram.device})"
                )
            target_gram = target_detached
        else:
            raise ValueError(
                f"Target style tensor pada layer '{layer_name}' harus berdimensi 3D (B, C, C) "
                f"atau 4D (B, C, H, W), diterima {target_detached.ndim}D"
            )

        layer_loss = F.mse_loss(gen_gram, target_gram)
        total_style_loss = total_style_loss + (float(weight) * layer_loss)

    return total_style_loss


def compute_total_loss(
    content_loss: torch.Tensor,
    style_loss: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1e4,
) -> torch.Tensor:
    """Menghitung total gabungan loss: alpha * L_content + beta * L_style.

    Args:
        content_loss: Tensor skalar content loss.
        style_loss: Tensor skalar style loss.
        alpha: Bobot loss konten (>= 0, default 1.0).
        beta: Bobot loss gaya (>= 0, default 10000.0).

    Returns:
        Tensor skalar total loss.

    Raises:
        TypeError: Jika loss bukan tensor atau alpha/beta bukan int/float.
        ValueError: Jika loss bukan skalar, bukan floating-point, memuat NaN/Inf,
            berbeda device, atau bobot alpha/beta < 0 / non-finite.
    """
    if not isinstance(content_loss, torch.Tensor):
        raise TypeError(
            f"content_loss harus berupa torch.Tensor, diterima {type(content_loss).__name__}"
        )
    if not isinstance(style_loss, torch.Tensor):
        raise TypeError(
            f"style_loss harus berupa torch.Tensor, diterima {type(style_loss).__name__}"
        )

    if content_loss.numel() != 1:
        raise ValueError(
            f"content_loss harus berupa skalar (numel == 1), diterima numel {content_loss.numel()}"
        )
    if style_loss.numel() != 1:
        raise ValueError(
            f"style_loss harus berupa skalar (numel == 1), diterima numel {style_loss.numel()}"
        )

    if not content_loss.is_floating_point() or not style_loss.is_floating_point():
        raise ValueError("content_loss dan style_loss harus bertipe floating-point")

    if not torch.isfinite(content_loss).all():
        raise ValueError("content_loss tidak boleh mengandung NaN atau Inf")
    if not torch.isfinite(style_loss).all():
        raise ValueError("style_loss tidak boleh mengandung NaN atau Inf")

    if content_loss.device != style_loss.device:
        raise ValueError(
            f"Device content_loss ({content_loss.device}) harus identik dengan "
            f"style_loss ({style_loss.device})"
        )

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha harus bertipe int atau float, diterima {type(alpha).__name__}")
    if math.isnan(alpha) or math.isinf(alpha) or alpha < 0:
        raise ValueError(f"alpha harus bernilai numerik finite >= 0, diterima {alpha}")

    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError(f"beta harus bertipe int atau float, diterima {type(beta).__name__}")
    if math.isnan(beta) or math.isinf(beta) or beta < 0:
        raise ValueError(f"beta harus bernilai numerik finite >= 0, diterima {beta}")

    return (float(alpha) * content_loss) + (float(beta) * style_loss)
