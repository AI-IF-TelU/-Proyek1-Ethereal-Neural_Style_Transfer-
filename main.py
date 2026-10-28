"""Titik masuk utama (CLI & End-to-End Pipeline) Neural Style Transfer.

Mengintegrasikan seluruh modul dari Stage 2 s.d. Stage 6:
1. ``src.preprocessing`` (A2): Pemuatan gambar & normalisasi tensor ImageNet.
2. ``src.feature_extractor`` (A1): Ekstraksi fitur VGG-19 (`features[:29]`).
3. ``src.losses`` & ``src.optimization`` (A3): Perhitungan loss & optimasi
   piksel (L-BFGS / Adam) dengan clamping ruang ImageNet.
4. ``src.visualization`` (A1): Penyimpanan citra hasil, panel perbandingan,
   dan grafik konvergensi loss.
"""

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union
from urllib.error import URLError

import torch
import torch.nn as nn
import yaml
from torchvision.models import VGG19_Weights

from src.feature_extractor import DEFAULT_LAYERS, get_vgg19_model
from src.losses import DEFAULT_STYLE_WEIGHTS
from src.optimization import run_style_transfer
from src.preprocessing import preprocess_image
from src.visualization import (
    plot_loss_history,
    save_comparison_grid,
    save_image,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def _resolve_repo_path(path_value: Union[str, Path]) -> Path:
    """Mengubah path relatif terhadap akar repositori menjadi path absolut."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_device(device_spec: str = "auto") -> torch.device:
    """Menentukan perangkat komputasi PyTorch berdasarkan spesifikasi string.

    Args:
        device_spec: ``'auto'``, ``'cpu'``, atau ``'cuda'``.

    Returns:
        Objek ``torch.device`` yang siap digunakan.

    Raises:
        TypeError: Jika ``device_spec`` bukan ``str``.
        ValueError: Jika nilai ``device_spec`` tidak dikenali atau meminta
            ``'cuda'`` saat CUDA tidak tersedia.
    """
    if not isinstance(device_spec, str):
        raise TypeError(
            "device_spec harus bertipe str, diterima "
            f"{type(device_spec).__name__}"
        )
    spec = device_spec.strip().lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Perangkat 'cuda' diminta tetapi tidak tersedia")
        return torch.device("cuda")
    raise ValueError(
        f"device_spec harus 'auto', 'cpu', atau 'cuda', diterima "
        f"'{device_spec}'"
    )


def _calibrate_offline_vgg(model: nn.Sequential, seed: int = 42) -> None:
    """Mengalibrasi bobot filter VGG-19 mode offline secara deterministik.

    Torchvision menginisialisasi ``vgg19(weights=None)`` dengan ``fan_out``
    sehingga varians aktivasi menyusut drastis pada layer pertama (3 -> 64).
    Kalibrasi ``fan_in`` menjaga skala aktivasi tetap O(1) di seluruh 29 layer
    ketika dijalankan pada lingkungan offline tanpa unduhan bobot ImageNet.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                fan_in = (
                    module.in_channels
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                )
                std = (2.0 / fan_in) ** 0.5
                weight_cpu = torch.randn(
                    module.weight.shape,
                    generator=gen,
                    dtype=module.weight.dtype,
                ) * std
                module.weight.copy_(weight_cpu.to(module.weight.device))
                if module.bias is not None:
                    module.bias.zero_()


def _load_vgg19_with_fallback(
    device: torch.device,
    use_pretrained: bool,
    seed: int = 42,
) -> tuple[nn.Sequential, str]:
    """Memuat model VGG-19 beku dengan fallback otomatis jika jaringan offline.

    Args:
        device: Perangkat target (``cpu`` atau ``cuda``).
        use_pretrained: Apakah mencoba memuat ``VGG19_Weights.DEFAULT``.
        seed: Seed deterministik untuk fallback offline.

    Returns:
        Tuple ``(model, mode_description)``.
    """
    if use_pretrained:
        try:
            model = get_vgg19_model(
                device=device, weights=VGG19_Weights.DEFAULT
            )
            return model, "Pretrained ImageNet (VGG19_Weights.DEFAULT)"
        except (URLError, OSError, RuntimeError) as exc:
            print(
                "[INFO] Unduhan bobot VGG-19 tidak tersedia di lingkungan "
                f"offline ({type(exc).__name__}); beralih ke filter VGG-19 "
                "terkalibrasi (deterministic fan_in)."
            )

    model = get_vgg19_model(device=device, weights=None)
    _calibrate_offline_vgg(model, seed=seed)
    return model, "Offline Calibrated VGG-19 (weights=None, fan_in)"


def load_config(
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Membaca berkas konfigurasi YAML dan melengkapi nilai default.

    Args:
        config_path: Path menuju berkas konfigurasi YAML.

    Returns:
        Dictionary konfigurasi terstruktur.

    Raises:
        FileNotFoundError: Jika berkas konfigurasi tidak ditemukan.
        ValueError: Jika isi YAML bukan pemetaan dictionary yang valid.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Berkas konfigurasi tidak ditemukan: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Isi berkas YAML harus berupa pemetaan (dict)")

    project_cfg = raw.get("project", {}) or {}
    paths_cfg = raw.get("paths", {}) or {}
    prep_cfg = raw.get("preprocessing", {}) or {}
    model_cfg = raw.get("model", {}) or {}
    opt_cfg = raw.get("optimization", {}) or {}

    raw_size = prep_cfg.get("size", [256, 256])
    if (
        not isinstance(raw_size, (list, tuple))
        or len(raw_size) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in raw_size)
    ):
        raise ValueError(
            "preprocessing.size di YAML harus berupa [height, width] integer"
        )

    return {
        "project": {
            "name": str(project_cfg.get("name", "neural-style-transfer")),
            "seed": int(project_cfg.get("seed", 42)),
        },
        "paths": {
            "content_image": paths_cfg.get(
                "content_image", "assets/content/content_sample.png"
            ),
            "style_image": paths_cfg.get(
                "style_image", "assets/style/style_sample.png"
            ),
            "output_image": paths_cfg.get(
                "output_image", "outputs/images/stylized_output.png"
            ),
            "comparison_image": paths_cfg.get(
                "comparison_image", "outputs/images/comparison_grid.png"
            ),
            "loss_graph": paths_cfg.get(
                "loss_graph", "outputs/graphs/loss_curve.png"
            ),
        },
        "preprocessing": {
            "size": (int(raw_size[0]), int(raw_size[1])),
        },
        "model": {
            "device": str(model_cfg.get("device", "auto")),
            "pretrained": bool(model_cfg.get("pretrained", True)),
            "content_layer": str(model_cfg.get("content_layer", "conv4_2")),
            "style_weights": dict(
                model_cfg.get("style_weights", DEFAULT_STYLE_WEIGHTS)
            ),
        },
        "optimization": {
            "optimizer_type": str(opt_cfg.get("optimizer_type", "lbfgs")),
            "num_steps": int(opt_cfg.get("num_steps", 50)),
            "learning_rate": float(opt_cfg.get("learning_rate", 0.5)),
            "alpha": float(opt_cfg.get("alpha", 1.0)),
            "beta": float(opt_cfg.get("beta", 10000.0)),
            "init_mode": str(opt_cfg.get("init_mode", "content")),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Membangun parser argumen CLI untuk pipeline Neural Style Transfer."""
    parser = argparse.ArgumentParser(
        description="Pipeline End-to-End Neural Style Transfer (VGG-19 Beku)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path berkas konfigurasi YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--content",
        type=str,
        default=None,
        help="Override path gambar konten (Content Image)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        help="Override path gambar referensi gaya (Style Image)",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default=None,
        help="Override path penyimpanan gambar hasil NST",
    )
    parser.add_argument(
        "--comparison-image",
        type=str,
        default=None,
        help="Override path penyimpanan panel perbandingan 3-kolom",
    )
    parser.add_argument(
        "--loss-graph",
        type=str,
        default=None,
        help="Override path penyimpanan grafik penurunan loss",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Ukuran gambar target (HEIGHT WIDTH) sesuai kebijakan D-005",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Jumlah langkah iterasi optimasi piksel (num_steps)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Bobot Content Loss (alpha >= 0)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Bobot Style Loss (beta >= 0)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=("lbfgs", "adam"),
        default=None,
        help="Jenis optimizer piksel ('lbfgs' atau 'adam')",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate untuk optimizer",
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        choices=("content", "style", "random"),
        default=None,
        help="Mode inisialisasi kanvas ('content', 'style', 'random')",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Perangkat eksekusi ('auto', 'cpu', atau 'cuda')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed untuk reprodusibilitas",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Gunakan bobot acak VGG-19 (weights=None) untuk uji offline",
    )
    return parser


def run_pipeline(config: Mapping[str, Any]) -> dict[str, Any]:
    """Menjalankan pipeline Neural Style Transfer sesuai konfigurasi.

    Args:
        config: Dictionary konfigurasi hasil ``load_config`` dan override CLI.

    Returns:
        Dictionary ringkasan hasil eksekusi berisi tensor hasil, riwayat loss,
        metrik penurunan loss, waktu eksekusi, dan path berkas keluaran.
    """
    seed = int(config["project"]["seed"])
    torch.manual_seed(seed)

    device = resolve_device(config["model"]["device"])
    size = tuple(config["preprocessing"]["size"])

    content_path = _resolve_repo_path(config["paths"]["content_image"])
    style_path = _resolve_repo_path(config["paths"]["style_image"])
    output_image_path = _resolve_repo_path(config["paths"]["output_image"])
    comparison_path = _resolve_repo_path(config["paths"]["comparison_image"])
    loss_graph_path = _resolve_repo_path(config["paths"]["loss_graph"])

    start_time = time.perf_counter()

    # 1. Stage 2: Preprocessing Gambar Konten & Gaya
    content_tensor = preprocess_image(content_path, size=size, device=device)
    style_tensor = preprocess_image(style_path, size=size, device=device)

    # 2. Stage 3: Load Model VGG-19 Beku
    use_pretrained = bool(config["model"].get("pretrained", True))
    model, weights_desc = _load_vgg19_with_fallback(
        device=device, use_pretrained=use_pretrained, seed=seed
    )

    # 3. Stage 4: Loop Optimasi Piksel Gambar
    opt_cfg = config["optimization"]
    generated_tensor, loss_history = run_style_transfer(
        content_tensor=content_tensor,
        style_tensor=style_tensor,
        model=model,
        num_steps=int(opt_cfg["num_steps"]),
        alpha=float(opt_cfg["alpha"]),
        beta=float(opt_cfg["beta"]),
        optimizer_type=str(opt_cfg["optimizer_type"]),
        learning_rate=float(opt_cfg["learning_rate"]),
        init_mode=str(opt_cfg["init_mode"]),
        style_weights=config["model"].get(
            "style_weights", DEFAULT_STYLE_WEIGHTS
        ),
        content_layer=str(
            config["model"].get("content_layer", DEFAULT_LAYERS["21"])
        ),
        seed=seed,
    )

    # 4. Stage 6: Visualisasi & Penyimpanan Output
    save_image(generated_tensor, output_image_path)
    plot_loss_history(
        loss_history,
        loss_graph_path,
        title=(
            f"Kurva Konvergensi Total Loss "
            f"({opt_cfg['optimizer_type'].upper()}, "
            f"{len(loss_history)} Steps)"
        ),
    )
    save_comparison_grid(
        content_tensor,
        style_tensor,
        generated_tensor,
        comparison_path,
        title=(
            f"Neural Style Transfer — alpha={opt_cfg['alpha']:g}, "
            f"beta={opt_cfg['beta']:g} ({opt_cfg['optimizer_type'].upper()})"
        ),
    )

    elapsed_sec = time.perf_counter() - start_time
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    reduction_pct = (
        ((initial_loss - final_loss) / initial_loss) * 100.0
        if initial_loss > 0.0
        else 0.0
    )

    print("=== Ringkasan Eksekusi Neural Style Transfer ===")
    print(f"Perangkat       : {device} ({weights_desc})")
    print(f"Ukuran (H, W)   : {size}")
    print(
        f"Optimizer       : {opt_cfg['optimizer_type'].upper()} "
        f"(steps={len(loss_history)}, lr={opt_cfg['learning_rate']}, "
        f"init={opt_cfg['init_mode']})"
    )
    print(
        f"Bobot Loss      : alpha={opt_cfg['alpha']}, "
        f"beta={opt_cfg['beta']}"
    )
    print(f"Loss Awal       : {initial_loss:,.4f}")
    print(f"Loss Akhir      : {final_loss:,.4f} (-{reduction_pct:.2f}%)")
    print(f"Waktu Eksekusi  : {elapsed_sec:.2f} detik")
    print(f"Gambar Hasil    : {output_image_path}")
    print(f"Panel Komparasi : {comparison_path}")
    print(f"Grafik Loss     : {loss_graph_path}")

    return {
        "generated_tensor": generated_tensor,
        "loss_history": loss_history,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "reduction_pct": reduction_pct,
        "elapsed_sec": elapsed_sec,
        "output_image_path": output_image_path,
        "comparison_image_path": comparison_path,
        "loss_graph_path": loss_graph_path,
    }


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Membaca argumen CLI, memuat konfigurasi, dan menjalankan pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)

    if args.content is not None:
        config["paths"]["content_image"] = args.content
    if args.style is not None:
        config["paths"]["style_image"] = args.style
    if args.output_image is not None:
        config["paths"]["output_image"] = args.output_image
    if args.comparison_image is not None:
        config["paths"]["comparison_image"] = args.comparison_image
    if args.loss_graph is not None:
        config["paths"]["loss_graph"] = args.loss_graph
    if args.size is not None:
        config["preprocessing"]["size"] = (args.size[0], args.size[1])
    if args.steps is not None:
        config["optimization"]["num_steps"] = args.steps
    if args.alpha is not None:
        config["optimization"]["alpha"] = args.alpha
    if args.beta is not None:
        config["optimization"]["beta"] = args.beta
    if args.optimizer is not None:
        config["optimization"]["optimizer_type"] = args.optimizer
    if args.lr is not None:
        config["optimization"]["learning_rate"] = args.lr
    if args.init_mode is not None:
        config["optimization"]["init_mode"] = args.init_mode
    if args.device is not None:
        config["model"]["device"] = args.device
    if args.seed is not None:
        config["project"]["seed"] = args.seed
    if args.no_pretrained:
        config["model"]["pretrained"] = False

    return run_pipeline(config)


if __name__ == "__main__":
    main()
