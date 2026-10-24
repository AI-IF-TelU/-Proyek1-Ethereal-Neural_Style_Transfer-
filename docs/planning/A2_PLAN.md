# 📗 Panduan & Rencana Kerja Anggota A2
### Marthin M. Tambunan (NIM: 103012580055)
**Peran:** Data Pipeline & Preprocessing Engineer

---

## 🎯 Tanggung Jawab Utama
1. Menyiapkan gambar konten dan gambar gaya seni untuk pengujian lokal.
2. Mengembangkan modul `src/preprocessing.py` (load gambar, resize, konversi tensor, dan normalisasi).
3. Menyusun fungsi *deprocessing* (mengembalikan tensor ke gambar RGB yang dapat ditampilkan/disimpan).
4. Membuat dan memvalidasi `tests/test_preprocessing.py`.

---

## 📋 Rincian Langkah Kerja (Stage 2: Data Pipeline)

### S02-01: Perencanaan Format Data & Pasangan Gambar
* **Tujuan:** Menentukan spesifikasi gambar masukan (format PNG/JPG, resolusi default $512 \times 512$, format warna RGB).
* **Status:** `COMPLETED` ✅

### S02-02: Implementasi Image Loader
* **Tujuan:** Membuat fungsi `load_image(image_path, size=None)` menggunakan pustaka Pillow (`PIL.Image`).
* **Target Berkas:** `src/preprocessing.py`
* **Status:** `COMPLETED` ✅ (Disetujui oleh A1 pada Handoff #1).

### S02-03: Resize, Konversi Tensor & Normalisasi VGG-19 (Kebijakan D-005)
* **Tujuan & Kebijakan Tunggal Parameter `size`:**
  1. **Satu Nama Parameter (`size`):** Baik `load_image` maupun `preprocess_image` menggunakan nama parameter **`size: tuple[int, int] | None`** (bukan `target_size`).
  2. **Urutan Dimensi `size = (height, width)` = `(H, W)`:** Selaras dengan tensor PyTorch `(1, 3, H, W)`. Saat memanggil Pillow `image.resize`, balik menjadi `(width, height)` (`image.resize((width, height), Image.Resampling.LANCZOS)`).
  3. **Validasi `size`:** Harus `None` atau tuple 2 integer murni (`not isinstance(v, bool)`) dengan `H > 0, W > 0`, selain itu lempar `ValueError`.
  4. **Alur Resize Tanpa Double Resize:**
     * `load_image(image_path, size=None)`: jika `size=(H, W)` diberikan, resize gambar dengan `LANCZOS` sehingga `img.height == H` dan `img.width == W`.
     * `preprocess_image(image, size=(512, 512), device="cpu") -> torch.Tensor`: jika `image` berupa path, panggil `load_image(image, size=size)`. Jika berupa `PIL.Image`, pastikan RGB dan hanya resize jika `size is not None` dan `(image.height, image.width) != size`. Konversi ke tensor 4D `(1, 3, H, W)` `torch.float32` dan terapkan normalisasi standar ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  5. **Deprocessing:** Buat fungsi `deprocess_image(tensor: torch.Tensor) -> Image.Image` untuk menghapus dimensi batch, denormalisasi, clamping ke rentang $[0.0, 1.0]$, dan konversi kembali ke `PIL.Image` RGB.
* **Target Berkas:** `src/preprocessing.py`, `tests/test_preprocessing.py`
* **Status:** `READY`

### S02-04: Preprocessing Testing & Handoff
* **Tujuan:** Memastikan seluruh unit test pada `tests/test_preprocessing.py` (termasuk pengujian ukuran non-persegi `size=(H, W)` agar tinggi dan lebar terbukti tidak tertukar) lulus 100%.
* **Target Berkas:** `tests/test_preprocessing.py`

---

## 💬 Template Prompt untuk AI Agent A2 (S02-03)

Saat Marthin berinteraksi dengan AI di VS Code, gunakan template ini:

```text
Saya adalah Marthin M. Tambunan (Anggota A2 - Data Pipeline Engineer).
Proyek: Neural Style Transfer (PyTorch, VGG-19 Pretrained).
Referensi Dokumen: docs/ARCHITECTURE.md (Bagian 2.A) dan docs/DECISIONS.md (Keputusan D-005).

Langkah Aktif: S02-03 - Image Resize, Tensor Conversion & Normalisasi VGG-19.
Tujuan:
1. Mengaktifkan parameter size=(H, W) pada load_image(image_path, size=None) dengan Image.Resampling.LANCZOS.
2. Membuat fungsi preprocess_image(image, size=(512, 512), device="cpu") -> torch.Tensor.
3. Membuat fungsi deprocess_image(tensor: torch.Tensor) -> PIL.Image.Image.

Kebijakan Wajib (D-005):
- Gunakan nama parameter `size` (bukan `target_size`) di semua fungsi.
- `size` SELALU berarti `(height, width) = (H, W)`. Balik menjadi `(W, H)` saat memanggil `PIL.Image.resize((W, H), Image.Resampling.LANCZOS)`.
- Hindari double resize jika preprocess_image menerima path atau PIL.Image yang sudah berukuran (H, W).
- Validasi ketat parameter `size` (tolak <= 0, bool, atau panjang tuple != 2 dengan ValueError).
- Normalisasi menggunakan mean=[0.485, 0.456, 0.406] dan std=[0.229, 0.224, 0.225], output 4D [1, 3, H, W] float32.
- Kerjakan hanya src/preprocessing.py dan tests/test_preprocessing.py (perbarui test_resize_not_silently_ignored menjadi pengujian resize (H, W) non-persegi dan validasi size).
```
