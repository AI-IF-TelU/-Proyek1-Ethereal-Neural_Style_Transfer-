# 📙 Panduan & Rencana Kerja Anggota A3
### Hafiz Fajar Ramadhan (NIM: 103012430027)
**Peran:** Loss Function & Optimization Engineer

---

## 🎯 Tanggung Jawab Utama
1. Mengimplementasikan kalkulasi **Gram Matrix** dari nol (*from scratch*).
2. Menyusun fungsi kerugian: **Content Loss**, **Style Loss**, dan **Total Loss** (`src/losses.py`).
3. Mengembangkan algoritma perulangan optimasi piksel gambar (`src/optimization.py`).
4. Mengelola eksperimen hyperparameter loss ($\alpha, \beta$) dan optimizer (L-BFGS / Adam).

---

## 📋 Rincian Langkah Kerja (Stage 4: Loss & Optimization — `COMPLETED` ✅)

### S04-01: Implementasi Matematis Gram Matrix
* **Tujuan:** Membuat fungsi `gram_matrix(tensor)` dari nol menggunakan operasi matriks PyTorch (`torch.bmm`).
* **Formula:** Untuk tensor fitur $F$ berukuran $(B, C, H \times W)$, $G = F \times F^T$, dinormalisasi dengan membagi terhadap total elemen $C \times H \times W$.
* **Target Berkas:** `src/losses.py`, `tests/test_losses.py`
* **Status:** `COMPLETED` ✅ (7 unit test lulus; disetujui oleh A1 pada Handoff #5).

### S04-02: Formulasi Content Loss (Kebijakan D-007)
* **Status:** `COMPLETED` ✅ (Teruji 20/20 test losses lulus; disetujui oleh A1 pada Handoff #6).
* **Tujuan:** Membuat fungsi `compute_content_loss(gen_features: torch.Tensor, content_features: torch.Tensor) -> torch.Tensor` serta merapikan `gram_matrix` dengan `.reshape(...)`.
* **Target Berkas:** `src/losses.py`, `tests/test_losses.py`, `logs/A3/S04-02-LOG.md`

### S04-03: Formulasi Style Loss & Total Loss (Kebijakan D-007)
* **Status:** `COMPLETED` ✅ (Teruji 20/20 test losses lulus; disetujui oleh A1 pada Handoff #6).
* **Tujuan:** Membuat konstanta `DEFAULT_STYLE_WEIGHTS`, fungsi `compute_style_loss`, dan `compute_total_loss` pada `src/losses.py`.
* **Target Berkas:** `src/losses.py`, `tests/test_losses.py`, `logs/A3/S04-03-LOG.md`

### S04-04: Optimization Loop Piksel Gambar (Kebijakan D-008)
* **Status:** `COMPLETED` ✅ (Teruji 9/9 test optimization lulus; disetujui oleh A1 pada Handoff #7).
* **Tujuan:** Menyusun modul `src/optimization.py` (`clamp_imagenet_tensor_` dan `run_style_transfer` mendukung optimizer `lbfgs` dan `adam` serta prekomputasi target sekali di awal).
* **Target Berkas:** `src/optimization.py`, `logs/A3/S04-04-LOG.md`

### S04-05: Testing, Analisis Kompleksitas & Handoff Stage 4
* **Status:** `COMPLETED` ✅ (Disetujui oleh A1 pada Handoff #7 — **Stage 4 Resmi Selesai**).
* **Tujuan:**
  1. Menyusun unit test komprehensif pada `tests/test_optimization.py` (`9/9 passed`; total repositori `58 passed, 2 skipped CUDA, 60 subtests passed`).
  2. Memperbarui bagian *Losses & Optimization* pada `docs/COMPLEXITY_ANALYSIS.md`.
  3. Mencatat log `logs/A3/S04-05-LOG.md` dan menyelesaikan serah terima penutupan Stage 4 pada `docs/HANDOFF.md` untuk diserahkan kepada Anggota A4 (Stage 5: QA).
* **Target Berkas:** `tests/test_optimization.py`, `docs/COMPLEXITY_ANALYSIS.md`, `logs/A3/S04-05-LOG.md`, `docs/HANDOFF.md`
