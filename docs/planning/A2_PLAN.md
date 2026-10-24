# 📗 Panduan & Rencana Kerja Anggota A2
### Marthin M. Tambunan (NIM: 103012580055)
**Peran:** Data Pipeline & Preprocessing Engineer

---

## 🎯 Tanggung Jawab Utama
1. Menyiapkan gambar konten dan gambar gaya seni pada folder `assets/`.
2. Mengembangkan modul `src/preprocessing.py` (load gambar, resize, tensor, dan normalisasi).
3. Menyusun fungsi *post-processing* (mengembalikan tensor ke gambar yang dapat ditampilkan/disimpan).
4. Membuat dan memvalidasi `tests/test_preprocessing.py`.

---

## 📋 Rincian Langkah Kerja (Stage 2: Data Pipeline)

### S02-01: Perencanaan Format Data & Pasangan Gambar
* **Tujuan:** Menentukan spesifikasi gambar masukan (format PNG/JPG, resolusi default misal $512 \times 512$, format warna RGB). Menyiapkan sampel gambar di `assets/content/` dan `assets/style/`.
* **Status:** `READY` (Dapat langsung dikerjakan sekarang!)
* **Acceptance Criteria:** Terdapat minimal 1 gambar konten dan 1 gambar gaya di folder aset yang valid dan tidak rusak.

### S02-02: Implementasi Image Loader
* **Tujuan:** Membuat fungsi `load_image(image_path, size=None)` menggunakan pustaka Pillow (`PIL.Image`).
* **Target Berkas:** `src/preprocessing.py`
* **Batasan:** Memastikan gambar dikonversi ke mode RGB (menangani gambar dengan channel RGBA atau Grayscale). Penanganan pesan error jika file gambar tidak ditemukan.

### S02-03: Resize, Konversi Tensor & Normalisasi VGG-19
* **Tujuan:** 
  1. Mengubah ukuran gambar sesuai target dimensi.
  2. Mengonversi gambar menjadi Tensor PyTorch dengan dimensi batch: `(1, 3, H, W)`.
  3. Menerapkan normalisasi standar ImageNet: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
  4. Membuat fungsi kebalikannya: `deprocess_image(tensor)` untuk un-normalize, clamping ke rentang $[0, 1]$, dan konversi kembali ke `PIL.Image`.
* **Target Berkas:** `src/preprocessing.py`

### S02-04: Preprocessing Testing & Handoff
* **Tujuan:** Menyusun unit test untuk menguji kesesuaian shape tensor, nilai piksel sebelum dan sesudah normalisasi, serta penanganan file tidak valid.
* **Target Berkas:** `tests/test_preprocessing.py`
* **Acceptance Criteria:** Perintah `pytest tests/test_preprocessing.py -v` lulus 100%.

---

## 💬 Template Prompt untuk AI Agent A2

Saat Marthin berinteraksi dengan AI di VS Code, gunakan template ini:

```text
Saya adalah Marthin M. Tambunan (Anggota A2 - Data Pipeline Engineer).
Proyek: Neural Style Transfer (PyTorch, VGG-19 Pretrained).
Referensi Dokumen: docs/planning/PROGRESS_TRACKER.md dan docs/ARCHITECTURE.md.

Langkah Aktif: [Misal: S02-03 - Preprocessing Gambar & Normalisasi]
Tujuan: Membuat fungsi preprocess_image() dan deprocess_image().
Batasan: 
- Input harus menghasilkan tensor 4D [1, C, H, W].
- Normalisasi harus menggunakan mean dan std ImageNet VGG-19.
- Kerjakan hanya file src/preprocessing.py dan tests/test_preprocessing.py. Jangan menyentuh modul model atau loss.

Berikan:
1. Penjelasan alur transformasi gambar.
2. Kode lengkap src/preprocessing.py dengan type hints dan docstrings.
3. Unit test lengkap pada tests/test_preprocessing.py.
```
