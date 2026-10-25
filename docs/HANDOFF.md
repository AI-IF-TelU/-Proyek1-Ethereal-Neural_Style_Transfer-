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

---

### Handoff #3: S02-04 (Opsi B CPU, Menunggu Review A1)
- **Task ID:** `S02-04`
- **Dari / Ke:** Marthin M. Tambunan (A2) -> Samuel Aaron (A1 - Ketua / Arsitek)
- **Otorisasi pengguna:** `eksekusi opsi  b saja`; scope CPU diterima pengguna, bukan acceptance A1 atas hasil akhir.
- **Berkas pekerjaan:** Test worker menambah 2 metode dan memperluas 3 metode di `tests/test_preprocessing.py`; dokumentasi pada `docs/COMPLEXITY_ANALYSIS.md`, `docs/ARCHITECTURE.md`, `docs/HANDOFF.md`, `docs/planning/A2_PLAN.md`, `docs/planning/PROGRESS_TRACKER.md`, dan `logs/A2/S02-04-LOG.md`. Tidak ada edit runtime; pembaruan dokumentasi tidak mengedit tests/src.
- **Kontrak / API:** D-005 tetap; klarifikasi `ValueError` untuk file tidak dikenali versus `OSError` untuk kegagalan decode format dikenali mencerminkan implementasi yang ada, bukan persetujuan manusia baru.
- **Test Status:** Python `.venv`: `-m pip check` bersih; `-m pytest tests/test_preprocessing.py -v -rs`: 21 passed, 1 skipped (CUDA), 43 subtests passed, 12.42s; `-m unittest tests.test_preprocessing -v`: 22 tests, OK (skipped=1). Hanya preprocessing, bukan seluruh repo.
- **Lingkungan aktual:** Python 3.14.3, pip 25.3, torch 2.14.1+cpu, numpy 2.5.4, Pillow 13.0.0, pytest 9.1.1 pada `2026-10-25T15:57:36.642816+07:00`. Pin requirements tetap tidak diubah.
- **Smoke aktual:** Enam kasus in-memory berurutan untuk `nature.jpg` dan `stylenature.png`, masing-masing `(512, 512)`, `(256, 512)`, dan `None`, lulus BCHW float32 CPU finite dan inverse RGB ukuran tepat dengan maxpixelerror 0 terhadap referensi decoded RGB/Lanczos. Shape asli `(1, 3, 3560, 2848)` dan `(1, 3, 788, 1400)`. Waktu `2026-10-25T15:59:45.121415+07:00` sampai `2026-10-25T15:59:51.060205+07:00`.
- **Known Issues / klarifikasi riwayat:** Native crash terdahulu tidak tereproduksi; akar penyebab belum terbukti. Pernyataan penyelesaian pada Handoff #2 dipertahankan sebagai riwayat, bukan bukti penyebab quoting atau perbaikan definitif. Probe corruption awal sengaja menghasilkan `OSError: image file is truncated`; regression sekarang lulus dengan exception yang diharapkan. CUDA belum terverifikasi.
- **Batas bukti:** Hasil aktual test worker diberikan pengguna; rincian pada [S02-04-LOG](../logs/A2/S02-04-LOG.md). Tidak ada ekspor lengkap percakapan/konsol yang otomatis tersedia; smoke in-memory tidak menghasilkan bukti ekspor file secara otomatis.
- **Acceptance:** **PENDING REVIEW A1**. Stage 2 dan S02-04 tetap `IN_PROGRESS`; Stage 3/S03 tetap `PENDING`. Belum ada approval completion, commit, atau push untuk pembaruan ini.
- **Langkah Berikutnya:** A1 mereview bukti CPU dan batas CUDA sebelum acceptance/penutupan Stage 2.
