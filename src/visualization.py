"""Modul visualisasi dan penyimpanan hasil Neural Style Transfer (Stage 6).

Menyediakan utilitas untuk:
1. Menyimpan tensor gambar hasil NST (atau ``PIL.Image.Image``) ke berkas PNG
   atau JPEG melalui ``deprocess_image`` (`save_image`).
2. Memplot dan menyimpan grafik kurva penurunan *Total Loss* per iterasi
   menggunakan backend non-interaktif Matplotlib (`plot_loss_history`).
3. Menyusun gambar perbandingan 3-panel (*Content* vs *Style* vs *Stylized*)
   untuk evaluasi visual dan dokumentasi akhir (`save_comparison_grid`).
"""

import math
import os
import tempfile
from pathlib import Path
from typing import Sequence, Union

# Pastikan direktori konfigurasi Matplotlib dapat ditulis di lingkungan sandbox
_mpl_dir = Path.home() / ".matplotlib"
if "MPLCONFIGDIR" not in os.environ and not os.access(_mpl_dir, os.W_OK):
    os.environ["MPLCONFIGDIR"] = str(
        Path(tempfile.gettempdir()) / "nst_matplotlib_cache"
    )
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.preprocessing import deprocess_image  # noqa: E402

SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")
SUPPORTED_PLOT_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".svg",
)


def _validate_output_path(
    output_path: Union[str, Path],
    allowed_extensions: tuple[str, ...],
) -> Path:
    """Memvalidasi path tujuan keluaran dan membuat folder induknya.

    Args:
        output_path: Path berkas tujuan bertipe ``str`` atau ``Path``.
        allowed_extensions: Tuple ekstensi huruf kecil yang diizinkan.

    Returns:
        Objek ``Path`` yang telah tervalidasi.

    Raises:
        TypeError: Jika ``output_path`` bukan ``str`` atau ``Path``.
        ValueError: Jika path kosong, mengarah ke direktori, atau ekstensinya
            tidak didukung.
    """
    if not isinstance(output_path, (str, Path)):
        raise TypeError(
            "output_path harus bertipe str atau Path, diterima "
            f"{type(output_path).__name__}"
        )
    if isinstance(output_path, str) and not output_path.strip():
        raise ValueError("output_path tidak boleh berupa string kosong")

    path = Path(output_path)
    if path.exists() and path.is_dir():
        raise ValueError(f"output_path mengarah ke direktori: {path}")

    ext = path.suffix.lower()
    if ext not in allowed_extensions:
        allowed_str = ", ".join(allowed_extensions)
        raise ValueError(
            f"Ekstensi output_path '{ext}' tidak didukung; "
            f"gunakan salah satu dari: {allowed_str}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _to_rgb_pil(image: Union[torch.Tensor, Image.Image]) -> Image.Image:
    """Mengonversi Tensor PyTorch atau PIL.Image menjadi PIL.Image mode RGB.

    Args:
        image: Tensor 4D ``(1, 3, H, W)`` / 3D ``(3, H, W)`` terstandarisasi
            ImageNet, atau objek ``PIL.Image.Image``.

    Returns:
        Objek ``PIL.Image.Image`` dalam mode ``'RGB'``.

    Raises:
        TypeError: Jika ``image`` bukan ``torch.Tensor`` atau ``PIL.Image``.
        ValueError: Jika dimensi gambar atau tensor tidak valid.
    """
    if isinstance(image, torch.Tensor):
        return deprocess_image(image)
    if isinstance(image, Image.Image):
        if image.width <= 0 or image.height <= 0:
            raise ValueError("PIL.Image memiliki dimensi kosong (<= 0)")
        return image.convert("RGB") if image.mode != "RGB" else image.copy()
    raise TypeError(
        "image harus bertipe torch.Tensor atau PIL.Image.Image, diterima "
        f"{type(image).__name__}"
    )


def save_image(
    tensor: Union[torch.Tensor, Image.Image],
    output_path: Union[str, Path],
) -> None:
    """Menyimpan tensor gambar hasil NST (atau PIL.Image) ke berkas gambar.

    Jika masukan berupa ``torch.Tensor`` dalam ruang ternormalisasi ImageNet,
    fungsi ini secara otomatis memanggil ``deprocess_image(tensor)`` terlebih
    dahulu sebelum menyimpan ke ``output_path``.

    Args:
        tensor: Tensor gambar 4D ``(1, 3, H, W)`` atau 3D ``(3, H, W)``,
            atau objek ``PIL.Image.Image``.
        output_path: Lokasi berkas keluaran (``.png``, ``.jpg``, ``.jpeg``).

    Returns:
        None. Berkas gambar disimpan pada ``output_path``.

    Raises:
        TypeError: Jika tipe ``tensor`` atau ``output_path`` tidak valid.
        ValueError: Jika bentuk tensor, nilai piksel, atau ekstensi berkas
            keluaran tidak valid.
    """
    path = _validate_output_path(output_path, SUPPORTED_IMAGE_EXTENSIONS)
    pil_image = _to_rgb_pil(tensor)
    pil_image.save(path)


def plot_loss_history(
    loss_history: Sequence[float],
    output_path: Union[str, Path],
    title: str = "Kurva Konvergensi Total Loss — Neural Style Transfer",
) -> None:
    """Membuat dan menyimpan grafik kurva penurunan Total Loss per iterasi.

    Args:
        loss_history: Urutan nilai loss numerik non-negatif per iterasi.
        output_path: Lokasi penyimpanan berkas grafik (mis. ``.png``).
        title: Judul grafik yang ditampilkan di bagian atas plot.

    Returns:
        None. Berkas grafik disimpan pada ``output_path``.

    Raises:
        TypeError: Jika ``loss_history`` bukan sequence numerik, atau
            ``output_path`` / ``title`` bertipe tidak valid.
        ValueError: Jika ``loss_history`` kosong, memuat nilai ``NaN``/``Inf``
            atau negatif, atau ekstensi ``output_path`` tidak didukung.
    """
    if isinstance(loss_history, (str, bytes)) or not isinstance(
        loss_history, Sequence
    ):
        raise TypeError(
            "loss_history harus berupa sequence numerik (list/tuple), "
            f"diterima {type(loss_history).__name__}"
        )
    if len(loss_history) == 0:
        raise ValueError("loss_history tidak boleh kosong")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title harus berupa string non-kosong")

    cleaned_losses: list[float] = []
    for idx, val in enumerate(loss_history):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(
                f"Elemen loss_history[{idx}] harus berupa int atau float, "
                f"diterima {type(val).__name__}"
            )
        float_val = float(val)
        if not math.isfinite(float_val) or float_val < 0.0:
            raise ValueError(
                f"Elemen loss_history[{idx}] harus finite dan >= 0, "
                f"diterima {val}"
            )
        cleaned_losses.append(float_val)

    path = _validate_output_path(output_path, SUPPORTED_PLOT_EXTENSIONS)
    steps = list(range(1, len(cleaned_losses) + 1))
    initial_loss = cleaned_losses[0]
    final_loss = cleaned_losses[-1]
    reduction_pct = (
        ((initial_loss - final_loss) / initial_loss) * 100.0
        if initial_loss > 0.0
        else 0.0
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(
        steps,
        cleaned_losses,
        color="#1f77b4",
        linewidth=2.0,
        marker="o" if len(steps) <= 25 else None,
        markersize=4,
        label="Total Loss",
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Iterasi (Step)", fontsize=10)
    ax.set_ylabel("Total Loss", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    summary_text = (
        f"Initial Loss (Step 1): {initial_loss:,.2f}\n"
        f"Final Loss (Step {len(steps)}): {final_loss:,.2f}\n"
        f"Penurunan: {reduction_pct:.2f}%"
    )
    ax.text(
        0.97,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f8f9fa",
            "edgecolor": "#adb5bd",
            "alpha": 0.9,
        },
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison_grid(
    content_image: Union[torch.Tensor, Image.Image],
    style_image: Union[torch.Tensor, Image.Image],
    generated_image: Union[torch.Tensor, Image.Image],
    output_path: Union[str, Path],
    title: str = "Perbandingan Neural Style Transfer",
) -> None:
    """Menyusun dan menyimpan panel perbandingan 3 kolom (Content/Style/NST).

    Args:
        content_image: Tensor atau ``PIL.Image`` gambar konten asli.
        style_image: Tensor atau ``PIL.Image`` gambar referensi gaya.
        generated_image: Tensor atau ``PIL.Image`` gambar hasil stylization.
        output_path: Lokasi penyimpanan berkas gambar gabungan.
        title: Judul utama pada bagian atas gambar perbandingan.

    Returns:
        None. Berkas panel perbandingan disimpan pada ``output_path``.

    Raises:
        TypeError: Jika salah satu gambar atau ``output_path`` tidak valid.
        ValueError: Jika dimensi gambar atau ekstensi berkas tidak valid.
    """
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title harus berupa string non-kosong")

    path = _validate_output_path(output_path, SUPPORTED_IMAGE_EXTENSIONS)
    content_pil = _to_rgb_pil(content_image)
    style_pil = _to_rgb_pil(style_image)
    gen_pil = _to_rgb_pil(generated_image)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), dpi=150)
    panels = (
        (content_pil, "1. Content Image"),
        (style_pil, "2. Style Image"),
        (gen_pil, "3. Stylized Output (NST)"),
    )
    for ax, (img, subtitle) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(subtitle, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


__all__ = [
    "SUPPORTED_IMAGE_EXTENSIONS",
    "SUPPORTED_PLOT_EXTENSIONS",
    "save_image",
    "plot_loss_history",
    "save_comparison_grid",
]
