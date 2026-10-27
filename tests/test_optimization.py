"""Unit test untuk modul optimization (src/optimization.py) — Stage 4: S04-04 & S04-05."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from src.feature_extractor import get_vgg19_model
from src.optimization import _IMAGENET_MAX, _IMAGENET_MIN, clamp_imagenet_tensor_, run_style_transfer
from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD


class TestClampImageNetTensor:
    """Pengujian fungsi clamping per-channel pada ruang ternormalisasi ImageNet (D-008.1)."""

    def test_clamp_4d_tensor_inplace_and_boundaries(self):
        """Memverifikasi batas clamping min dan max per-channel pada tensor 4D."""
        tensor = torch.tensor([
            [
                [[-10.0, 10.0], [0.0, 0.0]],  # Kanal 0 (R)
                [[-10.0, 10.0], [0.0, 0.0]],  # Kanal 1 (G)
                [[-10.0, 10.0], [0.0, 0.0]],  # Kanal 2 (B)
            ]
        ], dtype=torch.float32)

        out = clamp_imagenet_tensor_(tensor)
        assert out is tensor  # Memastikan in-place mutation

        for c in range(3):
            assert tensor[0, c, 0, 0].item() == pytest.approx(_IMAGENET_MIN[c])
            assert tensor[0, c, 0, 1].item() == pytest.approx(_IMAGENET_MAX[c])
            assert tensor[0, c, 1, 0].item() == pytest.approx(0.0)

    def test_clamp_3d_tensor(self):
        """Mendukung tensor 3D (3, H, W)."""
        tensor = torch.full((3, 4, 4), 10.0, dtype=torch.float32)
        clamp_imagenet_tensor_(tensor)
        for c in range(3):
            assert tensor[c].max().item() == pytest.approx(_IMAGENET_MAX[c])

    def test_values_within_bounds_remain_unchanged(self):
        """Nilai yang sudah di dalam rentang tidak berubah."""
        midpoint_r = (_IMAGENET_MIN[0] + _IMAGENET_MAX[0]) / 2.0
        tensor = torch.full((1, 3, 4, 4), midpoint_r, dtype=torch.float32)
        orig = tensor.clone()
        clamp_imagenet_tensor_(tensor)
        assert torch.allclose(tensor, orig)

    def test_input_validation_errors(self):
        """Memvalidasi penanganan exception parameter invalid."""
        with pytest.raises(TypeError, match="harus berupa torch.Tensor"):
            clamp_imagenet_tensor_([[1.0, 2.0]])

        with pytest.raises(ValueError, match="harus bertipe floating-point"):
            clamp_imagenet_tensor_(torch.ones(1, 3, 4, 4, dtype=torch.long))

        # Channel bukan 3
        with pytest.raises(ValueError, match="Dimensi kanal tensor 4D harus 3"):
            clamp_imagenet_tensor_(torch.randn(1, 4, 4, 4))

        # Dimensi bukan 3D atau 4D
        with pytest.raises(ValueError, match="harus berdimensi 3D atau 4D"):
            clamp_imagenet_tensor_(torch.randn(3, 4))

        # NaN / Inf
        nan_t = torch.randn(1, 3, 4, 4)
        nan_t[0, 0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="tidak boleh memuat nilai NaN atau Inf"):
            clamp_imagenet_tensor_(nan_t)


@pytest.fixture(scope="module")
def vgg_model():
    """Fixture model VGG-19 beku tanpa download bobot untuk efisiensi pengujian."""
    torch.manual_seed(42)
    return get_vgg19_model(device="cpu", weights=None)


class TestRunStyleTransfer:
    """Pengujian fungsi run_style_transfer dan loop optimasi (D-008)."""

    @pytest.fixture
    def dummy_tensors(self):
        """Fixture tensor konten dan gaya 4D sintetis (16x16)."""
        torch.manual_seed(42)
        content = torch.randn(1, 3, 16, 16)
        style = torch.randn(1, 3, 16, 16)
        clamp_imagenet_tensor_(content)
        clamp_imagenet_tensor_(style)
        return content, style

    def test_adam_optimizer_decreases_loss(self, vgg_model, dummy_tensors):
        """Memverifikasi optimizer Adam berhasil menurunkan nilai total loss."""
        content, style = dummy_tensors
        gen, loss_history = run_style_transfer(
            content_tensor=content,
            style_tensor=style,
            model=vgg_model,
            num_steps=10,
            optimizer_type="adam",
            learning_rate=0.05,
            init_mode="content",
        )

        assert gen.shape == content.shape
        assert len(loss_history) == 10
        assert not gen.requires_grad
        # Loss pada langkah akhir harus lebih kecil dari langkah awal
        assert loss_history[-1] < loss_history[0]

    def test_lbfgs_optimizer_execution_and_step_count(self, vgg_model, dummy_tensors):
        """Memverifikasi optimizer L-BFGS berjalan lancar dengan len(loss_history) == num_steps."""
        content, style = dummy_tensors
        gen, loss_history = run_style_transfer(
            content_tensor=content,
            style_tensor=style,
            model=vgg_model,
            num_steps=5,
            optimizer_type="lbfgs",
            learning_rate=1.0,
            init_mode="content",
        )

        assert gen.shape == content.shape
        assert len(loss_history) == 5
        assert all(isinstance(loss, float) and loss >= 0.0 for loss in loss_history)

    def test_init_modes(self, vgg_model, dummy_tensors):
        """Menguji ketiga mode inisialisasi: content, style, dan random."""
        content, style = dummy_tensors

        # init_mode='content'
        gen_c, _ = run_style_transfer(content, style, vgg_model, num_steps=1, init_mode="content")
        assert gen_c.shape == content.shape

        # init_mode='style'
        gen_s, _ = run_style_transfer(content, style, vgg_model, num_steps=1, init_mode="style")
        assert gen_s.shape == style.shape

        # init_mode='random' dengan seed
        gen_r1, _ = run_style_transfer(content, style, vgg_model, num_steps=1, init_mode="random", seed=123)
        gen_r2, _ = run_style_transfer(content, style, vgg_model, num_steps=1, init_mode="random", seed=123)
        assert torch.allclose(gen_r1, gen_r2)

    def test_model_parameters_remain_frozen(self, vgg_model, dummy_tensors):
        """Memverifikasi parameter model VGG-19 tidak pernah termutasi atau menerima gradien."""
        content, style = dummy_tensors
        param_snapshots = [p.clone() for p in vgg_model.parameters()]

        run_style_transfer(
            content_tensor=content,
            style_tensor=style,
            model=vgg_model,
            num_steps=3,
            optimizer_type="adam",
        )

        for p_before, p_after in zip(param_snapshots, vgg_model.parameters()):
            assert torch.equal(p_before, p_after)
            assert p_after.grad is None

    def test_input_validation_errors(self, vgg_model, dummy_tensors):
        """Memvalidasi penanganan exception saat diberikan input parameter yang salah."""
        content, style = dummy_tensors

        # Resolusi < 16x16
        small_t = torch.randn(1, 3, 8, 8)
        with pytest.raises(ValueError, match="Dimensi spasial content_tensor minimal 16x16"):
            run_style_transfer(small_t, style, vgg_model)

        # Batch size berbeda
        multi_batch_style = torch.randn(2, 3, 16, 16)
        with pytest.raises(ValueError, match="Batch size content_tensor"):
            run_style_transfer(content, multi_batch_style, vgg_model)

        # num_steps < 1 atau bertipe bool
        with pytest.raises(TypeError, match="num_steps harus bertipe int"):
            run_style_transfer(content, style, vgg_model, num_steps=True)
        with pytest.raises(ValueError, match="num_steps harus >= 1"):
            run_style_transfer(content, style, vgg_model, num_steps=0)

        # learning_rate invalid
        with pytest.raises(ValueError, match="learning_rate harus berupa numerik finite > 0"):
            run_style_transfer(content, style, vgg_model, learning_rate=-0.1)

        # alpha / beta negatif
        with pytest.raises(ValueError, match="alpha harus berupa numerik finite >= 0"):
            run_style_transfer(content, style, vgg_model, alpha=-1.0)
        with pytest.raises(ValueError, match="beta harus berupa numerik finite >= 0"):
            run_style_transfer(content, style, vgg_model, beta=-5.0)

        # optimizer_type tidak dikenali
        with pytest.raises(ValueError, match="optimizer_type harus 'lbfgs' atau 'adam'"):
            run_style_transfer(content, style, vgg_model, optimizer_type="rmsprop")

        # init_mode tidak dikenali
        with pytest.raises(ValueError, match="init_mode harus 'content', 'style', atau 'random'"):
            run_style_transfer(content, style, vgg_model, init_mode="zeros")

        # init_mode='style' dengan ukuran spasial berbeda
        diff_style = torch.randn(1, 3, 32, 32)
        with pytest.raises(ValueError, match="Ukuran spasial style_tensor"):
            run_style_transfer(content, diff_style, vgg_model, init_mode="style")
