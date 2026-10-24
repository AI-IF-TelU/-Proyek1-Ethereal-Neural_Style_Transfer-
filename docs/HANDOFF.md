# DOKUMEN SERAH TERIMA TUGAS (HANDOFF LOG)

Dokumen ini mencatat riwayat resmi serah terima antar-tahap dan persetujuan hasil (*acceptance*) dari Ketua Tim/PIC.

---

## Template Format Serah Terima
- **Task ID:** [Contoh: S02-02]
- **Dari / Ke:** [Nama Anggota Pengirim] -> [Nama Anggota Penerima / Ketua]
- **File Berubah:** [Daftar file kode dan pengujian]
- **Kontrak / API:** [Nama fungsi, parameter input, dan tipe output]
- **Test Status:** [Hasil eksekusi unit test]
- **Known Issues:** [Isu yang diketahui jika ada]
- **Acceptance:** [Status persetujuan: APPROVED / REVISION]
- **Tanggal / Waktu WIB:** [Timestamp sistem]

---

## 📋 Riwayat Serah Terima Resmi

### Handoff #1: S02-02 (Implementasi Image Loader)
- **Task ID:** `S02-02`
- **Dari / Ke:** Marthin M. Tambunan (A2) ➔ Samuel Aaron (A1 - Ketua / Arsitek)
- **File Berubah:**
  - `src/preprocessing.py`
  - `tests/test_preprocessing.py`
  - `docs/DECISIONS.md` (`D-004`)
  - `logs/A2/S02-02-LOG.md`
- **Kontrak / API:** `load_image(image_path: str | Path, size: tuple[int, int] | None = None) -> Image.Image` (RGB mode, Pillow, size non-None raises NotImplementedError).
- **Test Status:** **8/8 PASS** (`python -m unittest tests/test_preprocessing.py` lolos tanpa kegagalan).
- **Known Issues:** Aset gambar contoh disimpan secara lokal untuk pengujian manual; folder `assets/` tetap bersih sesuai strategi repo ringan.
- **Acceptance:** **APPROVED (Disetujui)** oleh Samuel Aaron (A1).
- **Langkah Berikutnya:** A2 resmi diberikan izin memulai `S02-03: Image Resize, Tensor & Normalisasi VGG`.
- **Tanggal / Waktu:** 2026-10-25 02:35 WIB

