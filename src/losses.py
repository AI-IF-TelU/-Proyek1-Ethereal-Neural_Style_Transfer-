"""Modul fungsi kerugian (Loss Functions) untuk Neural Style Transfer.

Mengimplementasikan kontrak Keputusan Arsitektur dan Dokumen Perancangan (Stage 4: S04-01):
- gram_matrix: Menghitung korelasi feature channel G = F @ F.T dinormalisasi dengan (C * H * W).
"""

from __future__ import annotations

import torch

__all__ = ["gram_matrix"]


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

    # Reshape spasial: (B, C, H * W)
    features = tensor.view(batch_size, channels, height * width)

    # Batch Matrix Multiplication: (B, C, H*W) @ (B, H*W, C) -> (B, C, C)
    gram = torch.bmm(features, features.transpose(1, 2))

    # Normalisasi sesuai kontrak ARCHITECTURE.md (C * H * W)
    normalization_factor = channels * height * width
    return gram / normalization_factor
