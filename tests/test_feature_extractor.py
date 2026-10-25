"""Unit test komprehensif untuk modul ``src.feature_extractor`` (Stage 3: S03-01 & S03-02)."""

from __future__ import annotations

import inspect
import unittest
from unittest import mock

from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights

from src.feature_extractor import (
    CONTENT_LAYERS,
    DEFAULT_LAYERS,
    MAX_FEATURE_LAYER_INDEX,
    STYLE_LAYERS,
    get_features,
    get_vgg19_model,
)
from src.preprocessing import preprocess_image


class TestFeatureExtractor(unittest.TestCase):
    """Suite pengujian kontrak arsitektur D-006 untuk ekstraktor fitur VGG-19."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.model = get_vgg19_model(device="cpu", weights=None)

    def test_default_signature_and_layer_constants(self) -> None:
        """Memastikan signature default memakai VGG19_Weights.DEFAULT dan konstanta lengkap."""
        sig = inspect.signature(get_vgg19_model)
        self.assertEqual(sig.parameters["device"].default, "cpu")
        self.assertIs(sig.parameters["weights"].default, VGG19_Weights.DEFAULT)

        expected_layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",
            "28": "conv5_1",
        }
        self.assertEqual(DEFAULT_LAYERS, expected_layers)
        self.assertEqual(CONTENT_LAYERS, ("conv4_2",))
        self.assertEqual(
            STYLE_LAYERS,
            ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"),
        )
        self.assertEqual(MAX_FEATURE_LAYER_INDEX, 28)

    def test_model_truncation_eval_mode_frozen_weights_and_non_inplace_relu(self) -> None:
        """Memastikan model dipotong 29 layer, mode eval, bobot beku, dan ReLU inplace=False."""
        self.assertIsInstance(self.model, nn.Sequential)
        self.assertEqual(len(self.model), MAX_FEATURE_LAYER_INDEX + 1)
        self.assertFalse(self.model.training)

        params = list(self.model.parameters())
        self.assertGreater(len(params), 0)
        for param in params:
            self.assertFalse(param.requires_grad)

        relu_layers = [m for m in self.model.modules() if isinstance(m, nn.ReLU)]
        self.assertGreater(len(relu_layers), 0)
        for relu in relu_layers:
            self.assertFalse(relu.inplace)

        for layer_idx in DEFAULT_LAYERS:
            self.assertIn(layer_idx, self.model._modules)
            self.assertIsInstance(self.model._modules[layer_idx], nn.Conv2d)

    def test_feature_shapes_square_non_square_and_batched(self) -> None:
        """Memvalidasi dimensi tensor keluaran (N, C_l, H_l, W_l) untuk berbagai ukuran input."""
        cases = [
            ((1, 3, 64, 64), {
                "conv1_1": (1, 64, 64, 64),
                "conv2_1": (1, 128, 32, 32),
                "conv3_1": (1, 256, 16, 16),
                "conv4_1": (1, 512, 8, 8),
                "conv4_2": (1, 512, 8, 8),
                "conv5_1": (1, 512, 4, 4),
            }),
            ((1, 3, 64, 128), {
                "conv1_1": (1, 64, 64, 128),
                "conv2_1": (1, 128, 32, 64),
                "conv3_1": (1, 256, 16, 32),
                "conv4_1": (1, 512, 8, 16),
                "conv4_2": (1, 512, 8, 16),
                "conv5_1": (1, 512, 4, 8),
            }),
            ((2, 3, 32, 48), {
                "conv1_1": (2, 64, 32, 48),
                "conv2_1": (2, 128, 16, 24),
                "conv3_1": (2, 256, 8, 12),
                "conv4_1": (2, 512, 4, 6),
                "conv4_2": (2, 512, 4, 6),
                "conv5_1": (2, 512, 2, 3),
            }),
        ]

        for input_shape, expected_shapes in cases:
            with self.subTest(input_shape=input_shape):
                tensor = torch.randn(input_shape, dtype=torch.float32)
                features = get_features(tensor, self.model)
                self.assertEqual(set(features.keys()), set(DEFAULT_LAYERS.values()))
                for name, expected_shape in expected_shapes.items():
                    feat = features[name]
                    self.assertEqual(tuple(feat.shape), expected_shape)
                    self.assertEqual(feat.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(feat).all())

    def test_conv_outputs_not_mutated_inplace_by_subsequent_relu(self) -> None:
        """Membuktikan aktivasi Conv2d yang disimpan tidak terpotong oleh ReLU di bawahnya."""
        tensor = torch.randn(1, 3, 32, 32, dtype=torch.float32)
        direct_conv1_1 = self.model[0](tensor).clone()
        features = get_features(tensor, self.model)

        torch.testing.assert_close(features["conv1_1"], direct_conv1_1)
        self.assertTrue((features["conv1_1"] < 0).any())

    def test_autograd_gradient_flows_to_input_image_and_not_vgg_parameters(self) -> None:
        """Memastikan backward pass mengisi grad pada image_tensor sementara bobot VGG tetap beku."""
        generated = torch.randn(1, 3, 32, 32, dtype=torch.float32, requires_grad=True)
        target_content = torch.randn(1, 512, 4, 4, dtype=torch.float32)

        params_before = [p.detach().clone() for p in self.model.parameters()]
        features = get_features(generated, self.model)

        loss = (
            torch.nn.functional.mse_loss(features["conv4_2"], target_content)
            + sum(features[name].square().mean() for name in STYLE_LAYERS)
        )
        loss.backward()

        self.assertIsNotNone(generated.grad)
        assert generated.grad is not None
        self.assertEqual(tuple(generated.grad.shape), (1, 3, 32, 32))
        self.assertTrue(torch.isfinite(generated.grad).all())
        self.assertGreater(float(generated.grad.abs().sum().item()), 0.0)

        for param, before in zip(self.model.parameters(), params_before):
            self.assertFalse(param.requires_grad)
            self.assertIsNone(param.grad)
            torch.testing.assert_close(param, before)

    def test_custom_layers_and_early_exit_stops_execution(self) -> None:
        """Memastikan ekstraksi subset layer kustom berhenti lebih awal (early exit)."""
        tensor = torch.randn(1, 3, 32, 32, dtype=torch.float32)
        custom_layers = {"0": "early_style", "5": "second_style"}

        with mock.patch.object(
            self.model[10], "forward", wraps=self.model[10].forward
        ) as mocked_later_layer:
            features = get_features(tensor, self.model, layers=custom_layers)
            mocked_later_layer.assert_not_called()

        self.assertEqual(set(features.keys()), {"early_style", "second_style"})
        self.assertEqual(tuple(features["early_style"].shape), (1, 64, 32, 32))
        self.assertEqual(tuple(features["second_style"].shape), (1, 128, 16, 16))

    def test_integration_with_stage2_preprocess_image(self) -> None:
        """Menguji integrasi langsung antara preprocess_image (Stage 2) dan get_features (Stage 3)."""
        pil_img = Image.new("RGB", (96, 64), color=(120, 180, 210))
        tensor = preprocess_image(pil_img, size=(64, 96), device="cpu")
        self.assertEqual(tuple(tensor.shape), (1, 3, 64, 96))

        features = get_features(tensor, self.model)
        self.assertEqual(tuple(features["conv1_1"].shape), (1, 64, 64, 96))
        self.assertEqual(tuple(features["conv4_2"].shape), (1, 512, 8, 12))
        self.assertEqual(tuple(features["conv5_1"].shape), (1, 512, 4, 6))

    def test_invalid_inputs_raise_expected_exceptions(self) -> None:
        """Memvalidasi penolakan input tidak valid dengan TypeError dan ValueError."""
        valid_tensor = torch.randn(1, 3, 32, 32, dtype=torch.float32)

        with self.assertRaises(TypeError):
            get_vgg19_model(device=123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            get_vgg19_model(device="   ")
        with self.assertRaises(ValueError):
            get_vgg19_model(device="invalid_device_xyz")
        with self.assertRaises(TypeError):
            get_vgg19_model(weights="invalid")  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            get_features([1, 2, 3], self.model)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            get_features(valid_tensor, "not_a_module")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            get_features(valid_tensor, self.model, layers=["0", "5"])  # type: ignore[arg-type]

        invalid_tensors = [
            torch.randn(3, 32, 32),
            torch.randn(1, 1, 3, 32, 32),
            torch.empty(0, 3, 32, 32),
            torch.randn(1, 1, 32, 32),
            torch.randn(1, 4, 32, 32),
            torch.randn(1, 3, 15, 32),
            torch.randn(1, 3, 32, 8),
            torch.ones(1, 3, 32, 32, dtype=torch.uint8),
            torch.ones(1, 3, 32, 32, dtype=torch.int64),
        ]
        for bad_tensor in invalid_tensors:
            with self.subTest(shape=tuple(bad_tensor.shape), dtype=bad_tensor.dtype):
                with self.assertRaises(ValueError):
                    get_features(bad_tensor, self.model)

        nan_tensor = valid_tensor.clone()
        nan_tensor[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            get_features(nan_tensor, self.model)

        inf_tensor = valid_tensor.clone()
        inf_tensor[0, 0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            get_features(inf_tensor, self.model)

        invalid_layer_dicts = [
            {},
            {"": "conv1_1"},
            {"0": ""},
            {"0": "dup", "5": "dup"},
            {"999": "missing_layer"},
        ]
        for bad_layers in invalid_layer_dicts:
            with self.subTest(bad_layers=bad_layers):
                with self.assertRaises(ValueError):
                    get_features(valid_tensor, self.model, layers=bad_layers)

        meta_model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, device="meta"))
        with self.assertRaises(ValueError):
            get_features(valid_tensor, meta_model, layers={"0": "conv1_1"})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA tidak tersedia")
    def test_cuda_device_placement_when_available(self) -> None:
        """Memverifikasi model dan ekstraksi fitur pada perangkat CUDA apabila tersedia."""
        cuda_model = get_vgg19_model(device="cuda", weights=None)
        cuda_tensor = torch.randn(1, 3, 32, 32, device="cuda", requires_grad=True)
        features = get_features(cuda_tensor, cuda_model)
        self.assertEqual(features["conv4_2"].device.type, "cuda")
        features["conv4_2"].sum().backward()
        self.assertIsNotNone(cuda_tensor.grad)


if __name__ == "__main__":
    unittest.main()
