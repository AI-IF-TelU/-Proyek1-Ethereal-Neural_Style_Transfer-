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
* **Tujuan:** Membuat fungsi `gram_matrix(tensor)` dari nol menggunakan operasi matriks PyTorch (`torch.matmul` atau `torch.bmm`).
* **Formula:** Untuk tensor fitur $F$ berukuran $(B, C, H \times W)$, $G = F \times F^T$, dinormalisasi dengan membagi terhadap total elemen $C \times H \times W$.
* **Target Berkas:** `src/losses.py`
* **Acceptance Criteria:** Matriks output berukuran $(C, C)$ atau $(B, C, C)$, simetris, dan gradien dapat mengalir.

### S04-02: Formulasi Content Loss
* **Tujuan:** Membuat fungsi `compute_content_loss(gen_features, content_features)`.
* **Formula:** Menggunakan *Mean Squared Error* (MSE) antar feature map layer target (`conv4_2`).
* **Target Berkas:** `src/losses.py`

### S04-03: Formulasi Style Loss
* **Tujuan:** Membuat fungsi `compute_style_loss(gen_features, style_gram_dict, layer_weights)`.
* **Formula:** Menghitung MSE antara Gram Matrix gambar target dan Gram Matrix gambar gaya seni pada beberapa layer (`conv1_1` s.d. `conv5_1`), kemudian dijumlahkan dengan bobot layer.
* **Target Berkas:** `src/losses.py`

### S04-04: Optimization Loop Piksel Gambar
* **Tujuan:** Menyusun loop optimasi di mana target optimasi adalah piksel gambar (`gen_image.requires_grad_(True)`), bukan bobot jaringan.
* **Optimizer:** Menggunakan optimizer **L-BFGS** dengan fungsi `closure()` atau **Adam**.
* **Target Berkas:** `src/optimization.py`
* **Fitur Penting:** Clamping piksel ke rentang valid $[0, 1]$ setelah setiap update, mencatat riwayat loss per iterasi.

### S04-05: Testing & Tuning Hyperparameter
* **Tujuan:** Menguji numerik loss, memastikan gradien mengalir hanya ke tensor gambar target, dan memvalidasi `tests/test_losses.py` & `tests/test_optimization.py`.
* **Acceptance Criteria:** Seluruh unit test loss dan loop optimasi lulus `pytest`.

---

## 💬 Template Prompt untuk AI Agent A3

Saat Hafiz berinteraksi dengan AI di VS Code, gunakan template ini:

```text
Saya adalah Hafiz Fajar Ramadhan (Anggota A3 - Loss & Optimization Engineer).
Proyek: Neural Style Transfer (PyTorch, VGG-19 Pretrained).
Referensi Dokumen: docs/planning/PROGRESS_TRACKER.md dan docs/ARCHITECTURE.md.

Langkah Aktif: [Misal: S04-01 - Implementasi Gram Matrix dari Nol]
Tujuan: Membuat fungsi kalkulasi Gram Matrix untuk ekstraksi gaya tekstur.
Batasan:
- Wajib dibangun secara mandiri menggunakan PyTorch tensor operations, dilarang menyalin pustaka NST jadi.
- Harus mempertahankan dimensi batch dan mendukung autograd.
- Kerjakan hanya file src/losses.py dan tests/test_losses.py.

Berikan:
1. Penjelasan matematis Gram Matrix dan korelasi channel fitur.
2. Kode implementasi gram_matrix(feature_tensor) dengan normalisasi.
3. Unit test untuk membuktikan simetri matriks dan output shape C x C.
```
