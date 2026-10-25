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

---

### Handoff #2: S02-03 (Image Resize, Tensor Conversion & Normalisasi VGG-19)
- **Task ID:** `S02-03`
- **Dari / Ke:** Marthin M. Tambunan (A2) ➔ Samuel Aaron (A1 - Ketua / Arsitek)
- **Git Commit Referensi:** `345c0d00383425f062949d902c542640fb46cece`
- **File Berubah:**
  - `src/preprocessing.py`
  - `tests/test_preprocessing.py`
  - `docs/ARCHITECTURE.md`
  - `docs/planning/A2_PLAN.md`
  - `docs/planning/PROGRESS_TRACKER.md`
  - `logs/A2/S02-03-LOG.md`
- **Kontrak / API (Sesuai Kebijakan Tunggal D-005):**
  - `load_image(image_path: str | Path, size: tuple[int, int] | None = None) -> Image.Image` (`size=(H, W)`, `LANCZOS`).
  - `preprocess_image(image: Image.Image | str | Path, size: tuple[int, int] | None = (512, 512), device: torch.device | str = "cpu") -> torch.Tensor` (Output 4D `(1, 3, H, W)` `float32`, normalisasi ImageNet, bebas *double resize*).
  - `deprocess_image(tensor: torch.Tensor) -> Image.Image` (Menerima 3D/4D, `detach().cpu().clone()` tanpa mutasi autograd, clamp `[0, 1]`, `.round().to(torch.uint8)`).
- **Test Status:** **19 passed, 1 skipped (CUDA), 35 subtests passed** pada `pytest` & `unittest`, ditambah *smoke test* 6 kasus ukuran pada aset nyata dengan selisih piksel *round-trip* `0`.
- **Known Issues:** Tidak ada; *native exit* sementara saat quoting PowerShell sudah teratasi ketika kasus diisolasi dan dijalankan secara berurutan.
- **Acceptance:** **APPROVED (Disetujui)** oleh Samuel Aaron (A1).
- **Langkah Berikutnya:** A2 resmi diberikan izin menyelesaikan `S02-04: Unit Testing & Handoff Preprocessing` untuk menutup Stage 2 dan menyerahkan pipeline ke Stage 3 (A1).
- **Tanggal / Waktu:** 2026-10-25 15:36 WIB
