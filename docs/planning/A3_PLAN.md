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

## 📋 Rincian Langkah Kerja (Stage 4: Loss & Optimization)

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
* **Status:** `IN_PROGRESS` 🟡 (Implementasi teruji 9 unit test passed; menunggu review A1).
* **Tujuan:** Menyusun modul `src/optimization.py` di mana parameter yang dioptimasi adalah piksel gambar hasil (`generated_tensor.requires_grad_(True)`), bukan bobot jaringan VGG-19.
* **Kontrak & Spesifikasi Teknis (`D-008`):**
  1. **Fungsi Clamping Ruang Ternormalisasi ImageNet (`clamp_imagenet_tensor_`):**
     * Karena tensor dari `preprocess_image` berada di ruang ternormalisasi ImageNet ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$), batas piksel valid RGB $[0.0, 1.0]$ di ruang ImageNet adalah:
       $$\text{min\_norm}_c = \frac{0.0 - \mu_c}{\sigma_c}, \quad \text{max\_norm}_c = \frac{1.0 - \mu_c}{\sigma_c}$$
     * Jangan melakukan `.clamp_(0.0, 1.0)` mentah karena akan memotong piksel gelap ($< 0$ di ruang ImageNet). Gunakan *clamping* per-channel ke $[\text{min\_norm}_c, \text{max\_norm}_c]$ di dalam `with torch.no_grad():` pada setiap iterasi.
  2. **Prekomputasi Target Sekali di Awal (`with torch.no_grad():`):**
     * Ekstrak fitur konten target (`content_features = get_features(content_tensor, model)[content_layer].detach()`) dan Gram Matrix gaya target (`style_grams = {l: gram_matrix(feat.detach()) for l, feat in get_features(style_tensor, model).items() if l in active_style_weights}`) **cukup satu kali** sebelum loop dimulai.
  3. **Signature Fungsi Utama `run_style_transfer`:**
     ```python
     def run_style_transfer(
         content_tensor: torch.Tensor,
         style_tensor: torch.Tensor,
         model: torch.nn.Module,
         num_steps: int = 300,
         alpha: float = 1.0,
         beta: float = 1e4,
         optimizer_type: str = "lbfgs",
         learning_rate: float = 1.0,
         init_mode: str = "content",
         style_weights: Mapping[str, float] | None = None,
         content_layer: str = "conv4_2",
         seed: int = 42,
     ) -> tuple[torch.Tensor, list[float]]:
     ```
     * Mendukung `init_mode`: `"content"` (klon `content_tensor`), `"style"` (klon `style_tensor`), atau `"random"` (noise di ruang ImageNet dengan `torch.manual_seed(seed)`).
     * Mendukung `optimizer_type`: `"lbfgs"` (`torch.optim.LBFGS([generated], lr=learning_rate, max_iter=1)` dengan fungsi `closure()`) dan `"adam"` (`torch.optim.Adam([generated], lr=learning_rate)`).
     * Mengembalikan `(generated.detach(), loss_history)` di mana `generated.shape == content_tensor.shape` dan `len(loss_history) == num_steps`.
* **Target Berkas:** `src/optimization.py`, `logs/A3/S04-04-LOG.md`

### S04-05: Testing, Analisis Kompleksitas & Handoff Stage 4
* **Status:** `IN_PROGRESS` 🟡 (Analisis kompleksitas & 58 passed repo; menunggu review A1).
* **Tujuan:**
  1. Menyusun unit test komprehensif pada `tests/test_optimization.py` (uji penurunan loss pada optimizer `adam` dan `lbfgs`, uji `init_mode` `"content"`/`"style"`/`"random"`, uji `clamp_imagenet_tensor_` menjaga rentang per-channel ImageNet, uji bobot VGG-19 tetap beku tidak berubah selama loop, dan uji validasi error input).
  2. Memperbarui bagian *Losses & Optimization* pada `docs/COMPLEXITY_ANALYSIS.md`.
  3. Mencatat log `logs/A3/S04-05-LOG.md` dan menyiapkan serah terima penutupan Stage 4 pada `docs/HANDOFF.md` untuk diserahkan kepada Anggota A4 (Stage 5: QA).
* **Target Berkas:** `tests/test_optimization.py`, `docs/COMPLEXITY_ANALYSIS.md`, `logs/A3/S04-05-LOG.md`, `docs/HANDOFF.md`

---

## 💬 Template Prompt untuk AI Agent A3 (S04-04 & S04-05)

Saat Hafiz berinteraksi dengan AI di VS Code untuk menuntaskan Stage 4 (`S04-04` dan `S04-05`), gunakan template ini:

```text
Saya adalah Hafiz Fajar Ramadhan (Anggota A3 - Loss & Optimization Engineer).
Proyek: Neural Style Transfer (PyTorch, VGG-19 Pretrained).
Referensi Dokumen: docs/planning/PROGRESS_TRACKER.md, docs/ARCHITECTURE.md (Bagian 2.D), docs/DECISIONS.md (D-008), dan docs/planning/A3_PLAN.md.

Langkah Aktif: S04-04 (Optimization Loop Piksel) & S04-05 (Testing, Analisis Kompleksitas & Handoff Stage 4)
Tujuan:
1. Mengimplementasikan `src/optimization.py` (`clamp_imagenet_tensor_` dan `run_style_transfer` mendukung optimizer `lbfgs` dan `adam`, `init_mode` `content`/`style`/`random`, serta prekomputasi fitur target sekali di awal) sesuai Keputusan D-008.
2. Menyusun unit test komprehensif pada `tests/test_optimization.py` dan memverifikasi `pytest tests/ -v -rs` lulus 100%.
3. Melengkapi analisis kompleksitas waktu dan ruang untuk modul `losses.py` dan `optimization.py` pada `docs/COMPLEXITY_ANALYSIS.md`.
4. Mencatat log resmi di `logs/A3/S04-04-LOG.md` dan `logs/A3/S04-05-LOG.md` serta memperbarui `docs/planning/PROGRESS_TRACKER.md` dan `docs/HANDOFF.md` (menunggu review penutupan Stage 4 oleh A1).
Jangan mengubah modul `src/preprocessing.py` atau `src/feature_extractor.py`.
```
