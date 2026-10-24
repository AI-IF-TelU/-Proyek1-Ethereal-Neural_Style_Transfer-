"""Pembacaan gambar S02-02; transformasi tensor belum diimplementasikan."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError


def load_image(
    image_path: str | Path, size: tuple[int, int] | None = None
) -> Image.Image:
    """Baca JPEG/PNG menjadi gambar RGB mandiri dengan ukuran asli.

    Parameter size dicadangkan untuk S02-03. Nilai selain None ditolak,
    bukan diabaikan. RGBA dikonversi dengan membuang kanal alpha.
    File sumber ditutup sebelum hasil dikembalikan.

    Raises:
        NotImplementedError: Resize diminta sebelum S02-03.
        FileNotFoundError: File tidak ditemukan.
        ValueError: Format bukan JPEG/PNG atau file tidak dikenali.
        OSError: File tidak dapat diakses atau decoding gagal.
    """
    if size is not None:
        raise NotImplementedError("Resize belum tersedia; gunakan size=None.")

    path = Path(image_path)
    try:
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Format gambar harus JPEG atau PNG.")
            image.load()
            return image.convert("RGB")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path}") from error
    except UnidentifiedImageError as error:
        raise ValueError(f"File bukan gambar yang valid: {path}") from error
