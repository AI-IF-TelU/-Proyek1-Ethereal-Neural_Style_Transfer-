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
* **Status:** `IN_PROGRESS` 🟡 (Teruji 20/20 test losses lulus; menunggu review A1).
* **Tujuan:** Membuat fungsi `compute_content_loss(gen_features: torch.Tensor, content_features: torch.Tensor) -> torch.Tensor`.
* **Kontrak & Formula (`D-007`):**
  1. Memvalidasi `gen_features` dan `content_features` sebagai `torch.Tensor` 4D `(B, C, H, W)` bertipe floating-point yang *finite*, memiliki `shape` dan `device` yang identik (lempar `TypeError` jika bukan tensor, `ValueError` jika shape/dtype/device/finite tidak valid).
  2. Menghitung *Mean Squared Error* (MSE) antara `gen_features` dan `content_features.detach()`:
     $$\mathcal{L}_{\text{content}} = \frac{1}{B \cdot C \cdot H \cdot W} \sum (F_{\text{gen}} - P_{\text{content}})^2 = \text{F.mse\_loss}(F_{\text{gen}}, P_{\text{content}}.\text{detach}())$$
     Penggunaan `.detach()` pada target memastikan gradien hanya mengalir ke `gen_features`.
  3. Sekaligus merapikan `tensor.view(...)` pada `gram_matrix` menjadi `tensor.reshape(...)` agar tahan terhadap tensor *non-contiguous*.
* **Target Berkas:** `src/losses.py`, `tests/test_losses.py`, `logs/A3/S04-02-LOG.md`

### S04-03: Formulasi Style Loss & Total Loss (Kebijakan D-007)
* **Status:** `IN_PROGRESS` 🟡 (Teruji 20/20 test losses lulus; menunggu review A1).
* **Tujuan:** Membuat fungsi `compute_style_loss` dan `compute_total_loss` pada `src/losses.py`.
* **Kontrak & Formula (`D-007`):**
  1. **Konstanta Bobot Layer Default:**
     `DEFAULT_STYLE_WEIGHTS: dict[str, float] = {"conv1_1": 0.2, "conv2_1": 0.2, "conv3_1": 0.2, "conv4_1": 0.2, "conv5_1": 0.2}`
  2. **`compute_style_loss(gen_features_dict: dict[str, torch.Tensor], style_gram_dict: dict[str, torch.Tensor], layer_weights: dict[str, float] | None = None) -> torch.Tensor`:**
     * Jika `layer_weights is None`, gunakan `DEFAULT_STYLE_WEIGHTS`.
     * Validasi `gen_features_dict`, `style_gram_dict`, dan `layer_weights` berupa `dict` non-kosong; seluruh kunci pada `layer_weights` wajib ada di `gen_features_dict` dan `style_gram_dict`, serta bobot $w_l \ge 0$ (bukan `bool`, *finite*).
     * Untuk setiap layer $l \in \text{layer\_weights}$:
       * Hitung `gen_gram = gram_matrix(gen_features_dict[l])` berukuran `(B, C, C)`.
       * Ambil target `target_entry = style_gram_dict[l].detach()`. Jika `target_entry` masih berupa tensor fitur 4D `(B, C, H, W)`, konversi otomatis dengan `gram_matrix(target_entry)`; jika sudah 3D `(B, C, C)`, validasi *finite* dan kecocokan shape `(B, C, C)` serta `device`.
       * Akumulasikan $w_l \cdot \text{F.mse\_loss}(\text{gen\_gram}, \text{target\_gram})$.
  3. **`compute_total_loss(content_loss: torch.Tensor, style_loss: torch.Tensor, alpha: float = 1.0, beta: float = 1e4) -> torch.Tensor`:**
     * Memvalidasi `content_loss` dan `style_loss` sebagai `torch.Tensor` skalar (`ndim == 0` atau `numel() == 1`) floating-point yang *finite* pada `device` yang sama.
     * Memvalidasi `alpha` dan `beta` sebagai angka `int | float` (bukan `bool`), *finite*, dan $\ge 0$.
     * Mengembalikan $\alpha \cdot \mathcal{L}_{\text{content}} + \beta \cdot \mathcal{L}_{\text{style}}$.
* **Target Berkas:** `src/losses.py`, `tests/test_losses.py`, `logs/A3/S04-03-LOG.md`

### S04-04: Optimization Loop Piksel Gambar
* **Tujuan:** Menyusun loop optimasi di mana target optimasi adalah piksel gambar (`gen_image.requires_grad_(True)`), bukan bobot jaringan.
* **Optimizer:** Menggunakan optimizer **L-BFGS** dengan fungsi `closure()` atau **Adam**.
* **Target Berkas:** `src/optimization.py`
* **Fitur Penting:** Clamping piksel atau menjaga stabilitas ruang terstandarisasi ImageNet dan mencatat riwayat loss per iterasi.

### S04-05: Testing & Tuning Hyperparameter
* **Tujuan:** Menguji numerik loss, memastikan gradien mengalir hanya ke tensor gambar target, dan memvalidasi `tests/test_losses.py` & `tests/test_optimization.py`.
* **Acceptance Criteria:** Seluruh unit test loss dan loop optimasi lulus `pytest`.

---

## 💬 Template Prompt untuk AI Agent A3 (S04-02 & S04-03)

Saat Hafiz berinteraksi dengan AI di VS Code untuk melanjutkan `S04-02` dan `S04-03`, gunakan template ini:

```text
Saya adalah Hafiz Fajar Ramadhan (Anggota A3 - Loss & Optimization Engineer).
Proyek: Neural Style Transfer (PyTorch, VGG-19 Pretrained).
Referensi Dokumen: docs/planning/PROGRESS_TRACKER.md, docs/ARCHITECTURE.md (Bagian 2.C), dan docs/DECISIONS.md (D-007).

Langkah Aktif: S04-02 (Content Loss) & S04-03 (Style Loss & Total Loss)
Tujuan: Melengkapi `src/losses.py` dengan `DEFAULT_STYLE_WEIGHTS`, `compute_content_loss`, `compute_style_loss`, dan `compute_total_loss` sesuai Keputusan D-007, serta mengganti `.view()` pada `gram_matrix` menjadi `.reshape()`.
Batasan:
- Kerjakan hanya `src/losses.py`, `tests/test_losses.py`, `logs/A3/S04-02-LOG.md`, `logs/A3/S04-03-LOG.md`, dan `docs/planning/PROGRESS_TRACKER.md`.
- Jangan mengubah modul `preprocessing.py` atau `feature_extractor.py`.
```
