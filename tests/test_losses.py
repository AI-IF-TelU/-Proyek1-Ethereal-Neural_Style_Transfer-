"""Unit test untuk modul losses (src/losses.py) — Stage 4: S04-01 s.d. S04-03."""

import pytest
import torch
from src.losses import (
    DEFAULT_STYLE_WEIGHTS,
    compute_content_loss,
    compute_style_loss,
    compute_total_loss,
    gram_matrix,
)


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
        x = torch.ones((1, 2, 2, 2), dtype=torch.float32)
        g = gram_matrix(x)
        expected = torch.full((1, 2, 2), 0.5, dtype=torch.float32)

        assert torch.allclose(g, expected)

    def test_non_contiguous_tensor_support(self):
        """Memverifikasi gram_matrix bekerja aman pada tensor non-contiguous (D-007)."""
        x = torch.randn(2, 8, 8, 16).transpose(1, 3)  # shape (2, 16, 8, 8), non-contiguous
        assert not x.is_contiguous()
        g = gram_matrix(x)
        assert g.shape == (2, 16, 16)

    def test_input_validation_errors(self):
        """Memvalidasi penanganan exception saat input tidak valid."""
        with pytest.raises(TypeError, match="harus berupa torch.Tensor"):
            gram_matrix([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="harus bertipe floating-point"):
            gram_matrix(torch.ones((1, 3, 16, 16), dtype=torch.long))

        with pytest.raises(ValueError, match="harus berdimensi 4D"):
            gram_matrix(torch.randn(10, 10))

        nan_tensor = torch.randn(1, 3, 16, 16)
        nan_tensor[0, 0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="tidak boleh mengandung nilai NaN atau Inf"):
            gram_matrix(nan_tensor)

        inf_tensor = torch.randn(1, 3, 16, 16)
        inf_tensor[0, 0, 0, 0] = float("inf")
        with pytest.raises(ValueError, match="tidak boleh mengandung nilai NaN atau Inf"):
            gram_matrix(inf_tensor)

        empty_tensor = torch.zeros((1, 0, 16, 16), dtype=torch.float32)
        with pytest.raises(ValueError, match="Setiap dimensi tensor harus >= 1"):
            gram_matrix(empty_tensor)


class TestContentLoss:
    """Pengujian fungsi compute_content_loss (S04-02)."""

    def test_zero_loss_for_identical_features(self):
        """Memverifikasi loss bernilai 0.0 jika fitur identik."""
        features = torch.randn(1, 64, 32, 32)
        loss = compute_content_loss(features, features)

        assert loss.item() == pytest.approx(0.0, abs=1e-7)
        assert loss.ndim == 0

    def test_numerical_accuracy(self):
        """Menguji keakuratan perhitungan Mean Squared Error."""
        gen = torch.zeros(1, 1, 2, 2)
        target = torch.full((1, 1, 2, 2), 2.0)
        # MSE = ((0 - 2)^2 * 4) / 4 = 4.0
        loss = compute_content_loss(gen, target)
        assert loss.item() == pytest.approx(4.0)

    def test_autograd_flow_only_to_gen_features(self):
        """Memverifikasi gradien autograd hanya mengalir ke gen_features."""
        gen = torch.randn(1, 32, 16, 16, requires_grad=True)
        target = torch.randn(1, 32, 16, 16, requires_grad=True)

        loss = compute_content_loss(gen, target)
        loss.backward()

        assert gen.grad is not None
        assert not torch.isnan(gen.grad).any()
        assert target.grad is None

    def test_input_validation_errors(self):
        """Memvalidasi penanganan exception pada parameter invalid."""
        valid_tensor = torch.randn(1, 8, 8, 8)

        # Bukan tensor
        with pytest.raises(TypeError, match="gen_features harus berupa torch.Tensor"):
            compute_content_loss([1.0], valid_tensor)
        with pytest.raises(TypeError, match="content_features harus berupa torch.Tensor"):
            compute_content_loss(valid_tensor, "not_tensor")

        # Tipe bukan floating-point
        int_tensor = torch.ones(1, 8, 8, 8, dtype=torch.long)
        with pytest.raises(ValueError, match="harus bertipe floating-point"):
            compute_content_loss(int_tensor, valid_tensor)

        # Bukan 4D
        with pytest.raises(ValueError, match="harus berdimensi 4D"):
            compute_content_loss(torch.randn(8, 8), torch.randn(8, 8))

        # NaN / Inf
        nan_tensor = valid_tensor.clone()
        nan_tensor[0, 0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="tidak boleh mengandung nilai NaN atau Inf"):
            compute_content_loss(nan_tensor, valid_tensor)

        # Shape tidak cocok
        different_shape = torch.randn(1, 16, 8, 8)
        with pytest.raises(ValueError, match="harus identik"):
            compute_content_loss(valid_tensor, different_shape)


class TestStyleLoss:
    """Pengujian fungsi compute_style_loss (S04-03)."""

    @pytest.fixture
    def dummy_feature_maps(self):
        """Fixture dictionary fitur multi-layer sintetis."""
        torch.manual_seed(42)
        layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        channels = [64, 128, 256, 512, 512]
        spatial = [32, 16, 8, 4, 2]
        return {
            layer: torch.randn(1, c, s, s)
            for layer, c, s in zip(layers, channels, spatial)
        }

    def test_zero_loss_for_identical_features(self, dummy_feature_maps):
        """Memverifikasi style loss bernilai 0.0 jika fitur gen dan target identik."""
        loss = compute_style_loss(dummy_feature_maps, dummy_feature_maps)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_precomputed_3d_gram_matrix_support(self, dummy_feature_maps):
        """Memverifikasi dukungan terhadap target style_gram_dict yang sudah berbentuk 3D (B, C, C)."""
        target_grams = {
            layer: gram_matrix(feat)
            for layer, feat in dummy_feature_maps.items()
        }
        loss = compute_style_loss(dummy_feature_maps, target_grams)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_custom_layer_weights(self, dummy_feature_maps):
        """Menguji penggunaan bobot kustom per layer."""
        custom_weights = {"conv1_1": 1.0, "conv2_1": 0.5}
        target_grams = {
            layer: gram_matrix(feat + 1.0)
            for layer, feat in dummy_feature_maps.items()
        }
        loss = compute_style_loss(dummy_feature_maps, target_grams, layer_weights=custom_weights)
        assert loss.item() > 0.0
        assert loss.ndim == 0

    def test_autograd_flow_to_all_layers(self, dummy_feature_maps):
        """Memverifikasi gradien mengalir ke seluruh layer yang memiliki bobot > 0."""
        gen_dict = {
            layer: (feat + 0.5).clone().requires_grad_(True)
            for layer, feat in dummy_feature_maps.items()
        }
        loss = compute_style_loss(gen_dict, dummy_feature_maps)
        loss.backward()

        for layer, tensor in gen_dict.items():
            assert tensor.grad is not None
            assert not torch.isnan(tensor.grad).any()
            assert (tensor.grad != 0).any()

    def test_input_validation_errors(self, dummy_feature_maps):
        """Memvalidasi penanganan exception pada parameter invalid."""
        # Bukan dict
        with pytest.raises(TypeError, match="harus bertipe Mapping/dict"):
            compute_style_loss([1, 2], dummy_feature_maps)

        # Dict kosong
        with pytest.raises(ValueError, match="tidak boleh kosong"):
            compute_style_loss({}, dummy_feature_maps)

        # Layer hilang di dict
        missing_dict = {"conv1_1": dummy_feature_maps["conv1_1"]}
        with pytest.raises(ValueError, match="tidak ditemukan pada style_gram_dict"):
            compute_style_loss(dummy_feature_maps, missing_dict)

        # Bobot layer bukan number atau bernilai negatif / bool
        with pytest.raises(TypeError, match="harus berupa int atau float"):
            compute_style_loss(
                dummy_feature_maps, dummy_feature_maps,
                layer_weights={"conv1_1": True}
            )
        with pytest.raises(ValueError, match="harus bernilai numerik riil finite >= 0"):
            compute_style_loss(
                dummy_feature_maps, dummy_feature_maps,
                layer_weights={"conv1_1": -0.5}
            )


class TestTotalLoss:
    """Pengujian fungsi compute_total_loss (S04-03)."""

    def test_numerical_combination(self):
        """Menguji kombinasi linear: alpha * L_content + beta * L_style."""
        content_loss = torch.tensor(2.0)
        style_loss = torch.tensor(0.5)
        alpha = 1.5
        beta = 100.0

        # expected = 1.5 * 2.0 + 100.0 * 0.5 = 3.0 + 50.0 = 53.0
        total = compute_total_loss(content_loss, style_loss, alpha=alpha, beta=beta)
        assert total.item() == pytest.approx(53.0)

    def test_autograd_flow_from_total_loss(self):
        """Memverifikasi autograd mengalir mundur melalui total loss."""
        x = torch.randn(1, 16, 8, 8, requires_grad=True)
        c_loss = (x ** 2).mean()
        s_loss = (x.abs()).mean()

        total = compute_total_loss(c_loss, s_loss, alpha=2.0, beta=3.0)
        total.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_input_validation_errors(self):
        """Memvalidasi penanganan exception pada parameter invalid."""
        valid_scalar = torch.tensor(1.0)

        # Bukan tensor
        with pytest.raises(TypeError, match="content_loss harus berupa torch.Tensor"):
            compute_total_loss(1.0, valid_scalar)

        # Bukan skalar
        non_scalar = torch.ones(2, 2)
        with pytest.raises(ValueError, match="harus berupa skalar"):
            compute_total_loss(non_scalar, valid_scalar)

        # Bukan floating-point
        int_scalar = torch.tensor(1, dtype=torch.long)
        with pytest.raises(ValueError, match="harus bertipe floating-point"):
            compute_total_loss(int_scalar, valid_scalar)

        # NaN / Inf
        nan_scalar = torch.tensor(float("nan"))
        with pytest.raises(ValueError, match="tidak boleh mengandung NaN atau Inf"):
            compute_total_loss(nan_scalar, valid_scalar)

        # alpha / beta negatif atau bertipe bool
        with pytest.raises(TypeError, match="alpha harus bertipe int atau float"):
            compute_total_loss(valid_scalar, valid_scalar, alpha=False)
        with pytest.raises(ValueError, match="alpha harus bernilai numerik finite >= 0"):
            compute_total_loss(valid_scalar, valid_scalar, alpha=-1.0)
        with pytest.raises(ValueError, match="beta harus bernilai numerik finite >= 0"):
            compute_total_loss(valid_scalar, valid_scalar, beta=float("inf"))
