"""Integration test pipeline NST lintas-modul — Stage 5: S05-03.

Rantai yang diuji (seluruhnya di memori, tanpa file dan tanpa unduhan bobot):
PIL.Image -> preprocess_image (A2) -> get_vgg19_model & get_features (A1)
-> gram_matrix & fungsi loss (A3) -> run_style_transfer (A3)
-> deprocess_image (A2).
"""

import math

import numpy as np
import pytest
import torch
from PIL import Image

from src.feature_extractor import (
    CONTENT_LAYERS,
    STYLE_LAYERS,
    get_features,
    get_vgg19_model,
)
from src.losses import (
    compute_content_loss,
    compute_style_loss,
    compute_total_loss,
    gram_matrix,
)
from src.optimization import run_style_transfer
from src.preprocessing import deprocess_image, preprocess_image

# Ukuran sengaja tidak persegi agar urutan (H, W) vs (W, H) ikut teruji.
SIZE = (32, 48)  # (H, W) sesuai D-005


@pytest.fixture(scope='module')
def vgg_model():
    """VGG-19 beku tanpa bobot pretrained (weights=None) agar cepat."""
    torch.manual_seed(42)
    return get_vgg19_model(device='cpu', weights=None)


@pytest.fixture(scope='module')
def images():
    """Pasangan gambar sintetis: konten bergradasi, gaya berupa noise."""
    rng = np.random.default_rng(42)
    height, width = 40, 60
    ramp = np.linspace(0, 255, width, dtype=np.float32)
    content = np.stack(
        [np.tile(ramp, (height, 1)),
         np.tile(ramp[::-1], (height, 1)),
         np.full((height, width), 128.0, dtype=np.float32)],
        axis=-1,
    ).astype(np.uint8)
    style = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(content), Image.fromarray(style)


@pytest.fixture(scope='module')
def tensors(images):
    """Tensor konten dan gaya hasil preprocess_image (A2)."""
    content_img, style_img = images
    return (preprocess_image(content_img, size=SIZE),
            preprocess_image(style_img, size=SIZE))


class TestPreprocessToFeatures:
    """Sambungan A2 -> A1: tensor preprocessing diterima feature extractor."""

    def test_preprocess_output_contract(self, tensors):
        for tensor in tensors:
            assert tensor.shape == (1, 3, *SIZE)
            assert tensor.dtype == torch.float32
            assert torch.isfinite(tensor).all()

    def test_features_layers_and_shapes(self, vgg_model, tensors):
        content, _ = tensors
        features = get_features(content, vgg_model)
        assert set(features) == set(CONTENT_LAYERS) | set(STYLE_LAYERS)
        # conv4_2 berada setelah 3 max-pool: resolusi turun 8 kali.
        assert features['conv4_2'].shape == (1, 512, SIZE[0] // 8,
                                             SIZE[1] // 8)
        assert features['conv1_1'].shape == (1, 64, *SIZE)


class TestFeaturesToLosses:
    """Sambungan A1 -> A3: peta fitur VGG-19 dipakai fungsi loss."""

    def test_gram_matrix_shapes(self, vgg_model, tensors):
        _, style = tensors
        features = get_features(style, vgg_model)
        for layer in STYLE_LAYERS:
            channels = features[layer].shape[1]
            gram = gram_matrix(features[layer])
            assert gram.shape == (1, channels, channels)
            assert torch.allclose(gram, gram.transpose(1, 2))

    def test_losses_and_gradient_reach_image(self, vgg_model, tensors):
        content, style = tensors
        with torch.no_grad():
            content_feats = get_features(content, vgg_model)
            style_grams = {
                layer: gram_matrix(feat)
                for layer, feat in get_features(style, vgg_model).items()
                if layer in STYLE_LAYERS
            }

        generated = content.clone().requires_grad_(True)
        gen_feats = get_features(generated, vgg_model)
        c_loss = compute_content_loss(gen_feats['conv4_2'],
                                      content_feats['conv4_2'])
        s_loss = compute_style_loss(gen_feats, style_grams)
        total = compute_total_loss(c_loss, s_loss, alpha=1.0, beta=1e4)

        # Gambar hasil = gambar konten, jadi content loss nol.
        assert c_loss.item() == pytest.approx(0.0, abs=1e-6)
        assert s_loss.item() > 0.0
        assert total.item() == pytest.approx(
            c_loss.item() + 1e4 * s_loss.item(), rel=1e-5)

        total.backward()
        assert generated.grad is not None
        assert generated.grad.abs().sum().item() > 0.0
        assert all(p.grad is None for p in vgg_model.parameters())


class TestEndToEndPipeline:
    """Rantai lengkap PIL -> optimasi -> PIL untuk Adam dan L-BFGS."""

    @pytest.mark.parametrize('optimizer_type, learning_rate, num_steps', [
        ('adam', 0.05, 10),
        ('lbfgs', 1.0, 5),
    ])
    def test_pipeline_produces_rgb_image(self, vgg_model, tensors,
                                         optimizer_type, learning_rate,
                                         num_steps):
        content, style = tensors
        params_before = [p.clone() for p in vgg_model.parameters()]

        generated, history = run_style_transfer(
            content, style, vgg_model,
            num_steps=num_steps,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            init_mode='content',
        )

        assert generated.shape == content.shape
        assert not generated.requires_grad
        assert len(history) == num_steps
        assert all(math.isfinite(loss) and loss >= 0.0 for loss in history)
        assert history[-1] < history[0]

        image = deprocess_image(generated)
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image.size == (SIZE[1], SIZE[0])  # PIL memakai (W, H)

        for before, after in zip(params_before, vgg_model.parameters()):
            assert torch.equal(before, after)

    @pytest.mark.parametrize('init_mode', ['content', 'style'])
    def test_lbfgs_accepts_non_contiguous_input(self, vgg_model, tensors,
                                                init_mode):
        """Regresi S05-03: output preprocess_image tidak kontigu."""
        content, style = tensors
        assert not content.is_contiguous()
        generated, history = run_style_transfer(
            content, style, vgg_model, num_steps=2,
            optimizer_type='lbfgs', init_mode=init_mode,
        )
        assert generated.is_contiguous()
        assert len(history) == 2

    def test_deprocess_roundtrip_of_preprocess(self, images):
        content_img, _ = images
        resized = content_img.resize((SIZE[1], SIZE[0]),
                                     Image.Resampling.LANCZOS)
        restored = deprocess_image(preprocess_image(resized, size=SIZE))
        diff = np.abs(np.asarray(restored, dtype=np.int16)
                      - np.asarray(resized, dtype=np.int16))
        assert diff.max() <= 1


class TestErrorHandling:
    """Temuan #2 S05-01: parameter mode non-string harus TypeError."""

    @pytest.mark.parametrize('value', [1, None, ['adam']])
    def test_optimizer_type_non_string(self, vgg_model, tensors, value):
        content, style = tensors
        with pytest.raises(TypeError, match='optimizer_type harus bertipe'):
            run_style_transfer(content, style, vgg_model, num_steps=1,
                               optimizer_type=value)

    @pytest.mark.parametrize('value', [0, None, ('content',)])
    def test_init_mode_non_string(self, vgg_model, tensors, value):
        content, style = tensors
        with pytest.raises(TypeError, match='init_mode harus bertipe'):
            run_style_transfer(content, style, vgg_model, num_steps=1,
                               init_mode=value)

    def test_uppercase_mode_still_valid(self, vgg_model, tensors):
        content, style = tensors
        _, history = run_style_transfer(content, style, vgg_model,
                                        num_steps=1, optimizer_type='ADAM',
                                        learning_rate=0.05,
                                        init_mode='Content')
        assert len(history) == 1
