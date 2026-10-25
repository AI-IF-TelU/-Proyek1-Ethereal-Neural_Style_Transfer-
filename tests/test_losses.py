"""Unit test untuk modul losses (src/losses.py) — Stage 4: S04-01."""

import pytest
import torch
from src.losses import gram_matrix


class TestGramMatrix:
    """Pengujian fungsi matematis gram_matrix."""

    def test_output_shape_and_type(self):
        """Memverifikasi dimensi output (B, C, C) dan tipe data floating-point."""
        batch_size, channels, height, width = 1, 64, 32, 32
        x = torch.randn(batch_size, channels, height, width)
        g = gram_matrix(x)

        assert g.shape == (batch_size, channels, channels)
        assert g.dtype == torch.float32

    def test_multi_batch_shape(self):
        """Memverifikasi batch dimension dipertahankan (B > 1)."""
        batch_size, channels, height, width = 3, 16, 8, 8
        x = torch.randn(batch_size, channels, height, width)
        g = gram_matrix(x)

        assert g.shape == (batch_size, channels, channels)

    def test_symmetry(self):
        """Membuktikan Gram Matrix bersifat simetris: G == G^T."""
        x = torch.randn(2, 32, 16, 16)
        g = gram_matrix(x)
        g_transpose = g.transpose(-1, -2)

        # Toleransi floating point
        assert torch.allclose(g, g_transpose, atol=1e-6)

    def test_autograd_flow(self):
        """Memverifikasi gradien autograd mengalir sempurna ke input tensor."""
        x = torch.randn(1, 16, 8, 8, requires_grad=True)
        g = gram_matrix(x)
        loss = g.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert (x.grad != 0).any()

    def test_3d_input_support(self):
        """Mendukung input 3D (C, H, W) dan mengembalikan (1, C, C)."""
        x = torch.randn(16, 8, 8)
        g = gram_matrix(x)
        assert g.shape == (1, 16, 16)

    def test_normalization_values(self):
        """Menguji perhitungan numerik eksak normalisasi (C * H * W)."""
        # Tensor konstan 1: shape (1, 2, 2, 2)
        # C = 2, H = 2, W = 2 -> total elemen normalisasi = 2 * 2 * 2 = 8
        # Fitur per channel berukuran 4 elemen semuanya 1.0
        # Dot product channel i dan j: sum(1 * 1) = 4.0
        # Normalisasi: 4.0 / 8 = 0.5
        x = torch.ones((1, 2, 2, 2), dtype=torch.float32)
        g = gram_matrix(x)
        expected = torch.full((1, 2, 2), 0.5, dtype=torch.float32)

        assert torch.allclose(g, expected)

    def test_input_validation_errors(self):
        """Memvalidasi penanganan exception saat input tidak valid."""
        # Bukan tensor
        with pytest.raises(TypeError, match="harus berupa torch.Tensor"):
            gram_matrix([[1, 2], [3, 4]])

        # Bukan floating-point (misal int)
        with pytest.raises(ValueError, match="harus bertipe floating-point"):
            gram_matrix(torch.ones((1, 3, 16, 16), dtype=torch.long))

        # Dimensi salah (misal 2D atau 5D)
        with pytest.raises(ValueError, match="harus berdimensi 4D"):
            gram_matrix(torch.randn(10, 10))

        # Mengandung NaN / Inf
        nan_tensor = torch.randn(1, 3, 16, 16)
        nan_tensor[0, 0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="tidak boleh mengandung nilai NaN atau Inf"):
            gram_matrix(nan_tensor)

        inf_tensor = torch.randn(1, 3, 16, 16)
        inf_tensor[0, 0, 0, 0] = float("inf")
        with pytest.raises(ValueError, match="tidak boleh mengandung nilai NaN atau Inf"):
            gram_matrix(inf_tensor)

        # Dimensi spasial atau channel kosong (0)
        empty_tensor = torch.zeros((1, 0, 16, 16), dtype=torch.float32)
        with pytest.raises(ValueError, match="Setiap dimensi tensor harus >= 1"):
            gram_matrix(empty_tensor)
