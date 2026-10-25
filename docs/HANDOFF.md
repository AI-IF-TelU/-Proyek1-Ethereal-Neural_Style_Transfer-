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

### Handoff #3: S02-04 (Penutupan Stage 2: Unit Testing, Analisis Kompleksitas & Handoff Preprocessing)
- **Task ID:** `S02-04`
- **Dari / Ke:** Marthin M. Tambunan (A2) ➔ Samuel Aaron (A1 - Ketua / Arsitek)
- **Git Commit Referensi:** `287badeb4d8bb9957a7cf9ef4abe65cfb92b7843`
- **Otorisasi pengguna:** `eksekusi opsi  b saja`; scope CPU diterima pengguna dan telah diaudit oleh A1.
- **Berkas pekerjaan:** Test worker menambah 2 metode dan memperluas 3 metode di `tests/test_preprocessing.py`; dokumentasi pada `docs/COMPLEXITY_ANALYSIS.md`, `docs/ARCHITECTURE.md`, `docs/HANDOFF.md`, `docs/planning/A2_PLAN.md`, `docs/planning/PROGRESS_TRACKER.md`, dan `logs/A2/S02-04-LOG.md`. Tidak ada edit runtime pada `src/preprocessing.py`.
- **Kontrak / API:** D-005 tetap utuh; klarifikasi `ValueError` untuk file tidak dikenali versus `OSError` untuk kegagalan decode format dikenali (termasuk PNG *truncated*) mencerminkan implementasi aktual.
- **Test Status:** Python `.venv`: `-m pip check` bersih; `-m pytest tests/test_preprocessing.py -v -rs`: **21 passed, 1 skipped (CUDA), 43 subtests passed** (12.42s); `-m unittest tests.test_preprocessing -v`: **22 tests, OK (skipped=1)**. Termasuk uji keamanan gradien `loss.backward()` setelah `deprocess_image` dan uji tanpa *redundant resize* pada path ukuran identik.
- **Lingkungan aktual:** Python 3.14.3, pip 25.3, torch 2.14.1+cpu, numpy 2.5.4, Pillow 13.0.0, pytest 9.1.1 pada `2026-10-25T15:57:36.642816+07:00`. Pin requirements tetap tidak diubah.
- **Smoke aktual:** Enam kasus in-memory berurutan untuk `nature.jpg` dan `stylenature.png`, masing-masing `(512, 512)`, `(256, 512)`, dan `None`, lulus BCHW float32 CPU finite dan inverse RGB ukuran tepat dengan maxpixelerror 0 terhadap referensi decoded RGB/Lanczos. Shape asli `(1, 3, 3560, 2848)` dan `(1, 3, 788, 1400)`. Waktu `2026-10-25T15:59:45.121415+07:00` sampai `2026-10-25T15:59:51.060205+07:00`.
- **Known Issues / klarifikasi riwayat:** Native crash terdahulu tidak tereproduksi; regression truncated PNG sekarang lulus dengan `OSError` yang diharapkan. Pengujian perangkat CUDA dilewati secara otomatis (`skipUnless`) karena lingkungan lokal menggunakan PyTorch CPU.
- **Batas bukti:** Rincian tercatat pada [S02-04-LOG](../logs/A2/S02-04-LOG.md) dan diaudit pada [S02-04-REVIEW-LOG](../logs/A1/S02-04-REVIEW-LOG.md).
- **Acceptance:** **APPROVED (Disetujui)** oleh Samuel Aaron (A1) — **Stage 2 Resmi Ditutup (`COMPLETED`)**.
- **Langkah Berikutnya:** Tongkat estafet resmi berpindah kepada Samuel Aaron (A1) untuk mengeksekusi **Stage 3 (`S03-01` & `S03-02`: VGG-19 Feature Extractor)** berdasarkan kontrak arsitektur `D-006`.
- **Tanggal / Waktu:** 2026-10-25 16:25 WIB

---

### Handoff #4: S03-01 & S03-02 (Penutupan Stage 3: VGG-19 Feature Extractor & Serah Terima ke Stage 4 / A3)
- **Task ID:** `S03-01` & `S03-02`
- **Dari / Ke:** Samuel Aaron (A1 - Ketua / Arsitek) ➔ Hafiz Fajar Ramadhan (A3 - Loss Function & Optimization Engineer)
- **File Berubah:**
  - `src/feature_extractor.py`
  - `tests/test_feature_extractor.py`
  - `docs/COMPLEXITY_ANALYSIS.md`
  - `docs/DECISIONS.md` (`D-006`)
  - `docs/ARCHITECTURE.md`
  - `docs/planning/A1_PLAN.md`
  - `docs/planning/PROGRESS_TRACKER.md`
  - `logs/A1/S03-01-ARCH-SPEC-LOG.md`
  - `logs/A1/S03-01-LOG.md`
  - `logs/A1/S03-02-LOG.md`
- **Kontrak / API (Sesuai Keputusan D-006):**
  - `DEFAULT_LAYERS: dict[str, str]` = `{"0": "conv1_1", "5": "conv2_1", "10": "conv3_1", "19": "conv4_1", "21": "conv4_2", "28": "conv5_1"}`
  - `CONTENT_LAYERS: tuple[str, ...]` = `("conv4_2",)`
  - `STYLE_LAYERS: tuple[str, ...]` = `("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")`
  - `get_vgg19_model(device: torch.device | str = "cpu", weights: VGG19_Weights | None = VGG19_Weights.DEFAULT) -> nn.Sequential` (Memuat `vgg19.features[:29]`, mengganti `ReLU(inplace=True)` menjadi `ReLU(inplace=False)`, membekukan seluruh bobot `requires_grad=False`, dan mengunci `.eval()`).
  - `get_features(image_tensor: torch.Tensor, model: nn.Module, layers: dict[str, str] | None = None) -> dict[str, torch.Tensor]` (Mengekstrak peta fitur tanpa `torch.no_grad()` sehingga gradien *autograd* mengalir ke `image_tensor` saat `requires_grad=True`).
- **Test Status:**
  - `tests/test_feature_extractor.py`: **8 passed, 1 skipped (CUDA), 17 subtests passed**.
  - Gabungan Stage 2 + Stage 3 (`pytest tests/test_feature_extractor.py tests/test_preprocessing.py -v -rs`): **29 passed, 2 skipped (CUDA), 60 subtests passed** (15.67s).
- **Known Issues:** Tidak ada; pengujian penempatan perangkat CUDA dilewati secara otomatis (`skipUnless`) pada mesin CPU.
- **Acceptance:** **APPROVED (Disetujui)** oleh Samuel Aaron (A1) — **Stage 3 Resmi Ditutup (`COMPLETED`)**.
- **Langkah Berikutnya:** Tongkat estafet resmi diserahkan kepada **Hafiz Fajar Ramadhan (A3)** untuk memulai **Stage 4 (`S04-01` s.d. `S04-05`: Gram Matrix, Content/Style Loss & Pixel Optimization)** pada `src/losses.py` dan `src/optimization.py`.
- **Tanggal / Waktu:** 2026-10-25 16:54 WIB
