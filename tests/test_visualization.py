"""Unit test untuk src/visualization.py dan main.py — Stage 6: S06-01."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import main as nst_main
from src.preprocessing import preprocess_image
from src.visualization import (
    plot_loss_history,
    save_comparison_grid,
    save_image,
)


def _make_sample_pil(
    height: int = 32, width: int = 48, seed: int = 123
) -> Image.Image:
    """Membuat gambar PIL RGB sintetis berukuran (width, height)."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class TestSaveImage:
    """Pengujian fungsi save_image pada src/visualization.py."""

    def test_save_from_4d_and_3d_tensor(self, tmp_path: Path) -> None:
        """Memverifikasi penyimpanan dari tensor 4D dan 3D ke berkas PNG."""
        pil_img = _make_sample_pil(32, 48)
        tensor_4d = preprocess_image(pil_img, size=(32, 48))
        tensor_3d = tensor_4d.squeeze(0)

        out_4d = tmp_path / "nested" / "out_4d.png"
        out_3d = tmp_path / "out_3d.jpg"

        save_image(tensor_4d, out_4d)
        save_image(tensor_3d, str(out_3d))

        assert out_4d.is_file()
        assert out_3d.is_file()

        with Image.open(out_4d) as loaded:
            assert loaded.mode == "RGB"
            assert loaded.size == (48, 32)

    def test_save_from_pil_image(self, tmp_path: Path) -> None:
        """Memverifikasi penyimpanan langsung dari objek PIL.Image."""
        pil_img = _make_sample_pil(24, 36)
        out_path = tmp_path / "from_pil.png"
        save_image(pil_img, out_path)

        with Image.open(out_path) as loaded:
            assert loaded.size == (36, 24)
            assert loaded.mode == "RGB"

    def test_invalid_inputs_raise_errors(self, tmp_path: Path) -> None:
        """Memverifikasi validasi tipe input dan ekstensi output_path."""
        pil_img = _make_sample_pil(16, 16)
        with pytest.raises(TypeError, match="torch.Tensor atau PIL.Image"):
            bad_img: object = [1, 2, 3]
            save_image(bad_img, tmp_path / "bad.png")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="output_path harus bertipe"):
            save_image(pil_img, 12345)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Ekstensi output_path"):
            save_image(pil_img, tmp_path / "bad.txt")

        with pytest.raises(ValueError, match="string kosong"):
            save_image(pil_img, "   ")


class TestPlotLossHistory:
    """Pengujian fungsi plot_loss_history pada src/visualization.py."""

    def test_creates_non_empty_plot_file(self, tmp_path: Path) -> None:
        """Memverifikasi pembuatan berkas grafik PNG dari riwayat loss."""
        losses = [1250.5, 980.2, 740.0, 520.8, 410.1]
        out_path = tmp_path / "graphs" / "loss_curve.png"
        plot_loss_history(losses, out_path)

        assert out_path.is_file()
        assert out_path.stat().st_size > 1000

    def test_invalid_loss_history_raises(self, tmp_path: Path) -> None:
        """Memverifikasi penolakan loss_history kosong atau tidak valid."""
        out = tmp_path / "loss.png"
        with pytest.raises(TypeError, match="sequence numerik"):
            plot_loss_history("123", out)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="tidak boleh kosong"):
            plot_loss_history([], out)

        with pytest.raises(TypeError, match="int atau float"):
            plot_loss_history([True, 1.0], out)

        with pytest.raises(ValueError, match="finite dan >= 0"):
            plot_loss_history([10.0, -1.0], out)

        with pytest.raises(ValueError, match="finite dan >= 0"):
            plot_loss_history([10.0, float("nan")], out)


class TestSaveComparisonGrid:
    """Pengujian fungsi save_comparison_grid pada src/visualization.py."""

    def test_creates_comparison_grid_file(self, tmp_path: Path) -> None:
        """Memverifikasi pembuatan panel perbandingan 3 kolom."""
        c_img = _make_sample_pil(32, 32)
        s_img = _make_sample_pil(32, 32)
        g_tensor = preprocess_image(c_img, size=(32, 32))
        out_path = tmp_path / "comparison.png"

        save_comparison_grid(c_img, s_img, g_tensor, out_path)
        assert out_path.is_file()
        assert out_path.stat().st_size > 1000


class TestMainPipelineCLI:
    """Pengujian konfigurasi dan eksekusi CLI main.py."""

    def test_load_config_and_resolve_device(self) -> None:
        """Memverifikasi pembacaan config/config.yaml dan resolve_device."""
        cfg = nst_main.load_config()
        assert cfg["project"]["name"] == "neural-style-transfer"
        assert isinstance(cfg["preprocessing"]["size"], tuple)
        assert len(cfg["preprocessing"]["size"]) == 2

        dev = nst_main.resolve_device("cpu")
        assert dev == torch.device("cpu")

        with pytest.raises(ValueError, match="device_spec harus"):
            nst_main.resolve_device("tpu")

    def test_main_cli_end_to_end_offline(self, tmp_path: Path) -> None:
        """Memverifikasi eksekusi penuh main() dengan --no-pretrained."""
        content_path = tmp_path / "content.png"
        style_path = tmp_path / "style.png"
        grad = np.tile(
            np.linspace(0, 255, 48, dtype=np.uint8), (40, 1)
        )
        content_arr = np.stack([grad, grad, grad], axis=-1)
        Image.fromarray(content_arr, mode="RGB").save(content_path)
        _make_sample_pil(40, 48, seed=202).save(style_path)

        out_img = tmp_path / "outputs" / "stylized.png"
        comp_img = tmp_path / "outputs" / "comparison.png"
        loss_plot = tmp_path / "outputs" / "loss.png"

        result = nst_main.main(
            [
                "--content",
                str(content_path),
                "--style",
                str(style_path),
                "--output-image",
                str(out_img),
                "--comparison-image",
                str(comp_img),
                "--loss-graph",
                str(loss_plot),
                "--size",
                "32",
                "48",
                "--steps",
                "5",
                "--optimizer",
                "lbfgs",
                "--lr",
                "1.0",
                "--device",
                "cpu",
                "--no-pretrained",
            ]
        )

        assert out_img.is_file()
        assert comp_img.is_file()
        assert loss_plot.is_file()
        assert len(result["loss_history"]) == 5
        assert result["final_loss"] < result["initial_loss"]

        with Image.open(out_img) as saved_img:
            assert saved_img.size == (48, 32)
            assert saved_img.mode == "RGB"
