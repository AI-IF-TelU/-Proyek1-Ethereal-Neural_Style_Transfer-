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
* **Status:** `COMPLETED` ✅ (Disetujui oleh A1 pada Handoff #2).

### S02-04: Preprocessing Testing & Handoff (Penutupan Stage 2)
* **Status:** `IN_PROGRESS` (opsi B CPU telah diverifikasi; acceptance A1 masih pending pada Handoff #3).
* **Tujuan:**
   1. Memverifikasi regresi preprocessing CPU sesuai opsi B pengguna; CUDA dilewati dan belum terverifikasi, bukan klaim seluruh suite repositori lulus 100%.
  2. Mendokumentasikan analisis kompleksitas waktu dan memori final untuk modul `preprocessing.py` pada `docs/COMPLEXITY_ANALYSIS.md` dan `logs/A2/S02-04-LOG.md`.
  3. Menyelesaikan catatan serah terima (*handoff*) Stage 2 dari A2 kepada A1 (Stage 3: VGG-19 Feature Extractor).
* **Target Berkas:** `tests/test_preprocessing.py`, `docs/COMPLEXITY_ANALYSIS.md`, `logs/A2/S02-04-LOG.md`
* **Hasil aktual:** Pytest 21 passed, 1 skipped (CUDA), 43 subtests passed (12.42s); unittest 22 tests OK (skipped=1); pip check bersih; enam smoke aset aktual in-memory berurutan lulus. Rincian lingkungan, waktu, dan batas bukti pada log S02-04; pin requirements tidak diubah.
* **Sinkronisasi dokumentasi:** `docs/HANDOFF.md`, `docs/planning/PROGRESS_TRACKER.md`, rencana ini, dan klarifikasi exception minimal `docs/ARCHITECTURE.md`. Tidak ada perubahan runtime; acceptance A1 belum diberikan. Stage 2 tetap `IN_PROGRESS`, Stage 3/S03 tetap `PENDING`.

---

## 💬 Template Prompt untuk AI Agent A2 (S02-04)

Template planning historis berikut dipertahankan sebagai referensi, bukan instruksi eksekusi ulang atau approval penutupan. Eksekusi aktual mengikuti opsi B CPU dan menunggu review A1:

```text
Saya adalah Marthin M. Tambunan (Anggota A2 - Data Pipeline Engineer).
Proyek: Neural Style Transfer (PyTorch, VGG-19 Pretrained).
Referensi Dokumen: docs/ARCHITECTURE.md (Bagian 2.A), docs/DECISIONS.md (D-005), dan logs/A2/S02-04-LOG.md.

Langkah Aktif: S02-04 - Preprocessing Testing, Analisis Kompleksitas & Handoff Stage 2.
Tujuan:
1. Menjalankan verifikasi akhir test suite `tests/test_preprocessing.py` dan memastikan tidak ada celah kontrak pada `src/preprocessing.py`.
2. Memperbarui bagian Preprocessing pada `docs/COMPLEXITY_ANALYSIS.md` dengan analisis Big-O waktu dan memori aktual untuk `load_image`, `preprocess_image`, dan `deprocess_image`.
3. Melengkapi `logs/A2/S02-04-LOG.md` sebagai bukti penutupan Stage 2 untuk diserahkan kepada A1 (Stage 3).
Jangan menyentuh modul `feature_extractor.py`, `losses.py`, atau `optimization.py`.
```
