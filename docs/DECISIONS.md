# DECISIONS (Architecture Decision Records)

| ID | Keputusan | Alasan | Alternatif | Disetujui |
|---|---|---|---|---|
| D-001 | VGG-19 pretrained frozen | Standar NST Gatys et al. | Train from scratch | Seluruh Tim (A1 Lead) |
| D-002 | Optimasi pixel generated image | Mempertahankan bobot model tetap beku | Fine-tune weights | Seluruh Tim (A1 Lead) |
| D-003 | Modular struktur (5 modul src) | Kolaborasi multi-agent terisolasi | Monolitik satu skrip | Seluruh Tim (A1 Lead) |
| D-004 | Batas scope image loader S02-02 | Penolakan parameter size hingga S02-03 | Resize di loader | A1 (Samuel) & A2 (Marthin) |
| D-005 | Kebijakan Tunggal Parameter `size=(H, W)` & Preprocessing Tensor S02-03 | Mencegah tertukarnya `(W, H)` Pillow vs `(H, W)` PyTorch serta menyamakan nama parameter `size` | Memakai `target_size` terpisah atau urutan `(W, H)` | A1 (Samuel) & A2 (Marthin) |

---

## D-004: Batas loader S02-02 (Superseded pada bagian `size` oleh D-005 di S02-03)

- **Konteks:** Signature `load_image(image_path, size=None)` sudah direncanakan, tetapi resize termasuk S02-03 dan perilaku parameter belum dijelaskan.
- **Keputusan (Saat S02-02):** Hasil berupa `PIL.Image` RGB mandiri, ukuran asli saat `size=None`. Nilai `size` selain None ditolak dengan `NotImplementedError` sampai S02-03.
- **Alternatif:** Mengimplementasikan resize pada loader pada tahap S02-02; tidak dipilih untuk mematuhi *Stage-Gate Sequential*.
- **Persetujuan:** Disetujui oleh A2 dan diverifikasi/diterima oleh A1 (Samuel Aaron).
- **Dampak:** `src/preprocessing.py`, `tests/test_preprocessing.py`; tidak mengubah modul model/loss.
- **Pencatatan Sistem:** 2026-10-25T01:57:30+07:00 (WIB).

---

## D-005: Kebijakan Tunggal Parameter `size=(H, W)`, Resize & Normalisasi Tensor S02-03

- **Konteks:**
  1. Pada S02-02 fungsi `load_image` menggunakan nama parameter `size`, sedangkan draf awal S02-03 sempat menyebut `target_size` pada `preprocess_image`. Dua nama berbeda untuk konsep yang sama berisiko menimbulkan kebingungan.
  2. Terdapat perbedaan konvensi urutan dimensi antara pustaka Pillow (`PIL.Image.resize` mengharapkan `(width, height) = (W, H)`) dan PyTorch Tensor (`[N, C, H, W]` menggunakan `(height, width) = (H, W)`). Jika tidak diatur dalam satu kebijakan tegas, dimensi tinggi dan lebar akan tertukar pada gambar non-persegi.
- **Keputusan Teknis (Satu Kebijakan Tunggal):**
  1. **Penyeragaman Nama Parameter (`size`):**
     - Baik `load_image` maupun `preprocess_image` wajib menggunakan nama parameter yang sama: **`size: tuple[int, int] | None`**. Istilah `target_size` dihapus dari kontrak.
  2. **Konvensi Urutan Dimensi `size = (height, width)` = `(H, W)`:**
     - Di seluruh fungsi publik `src/preprocessing.py`, tuple `size` **selalu berarti `(height, width)` = `(H, W)`** selaras dengan dimensi spasial tensor PyTorch `(1, 3, H, W)`.
     - Saat memanggil `PIL.Image.resize(...)` secara internal, fungsi wajib membalik tuple `(height, width)` menjadi `(width, height)`:
       `image.resize((width, height), Image.Resampling.LANCZOS)`.
     - Dengan aturan ini, untuk `size=(256, 512)`:
       - `load_image(path, size=(256, 512))` mengembalikan `PIL.Image` dengan `img.height == 256` dan `img.width == 512` (`img.size == (512, 256)`).
       - `preprocess_image(img, size=(256, 512))` mengembalikan Tensor dengan `tensor.shape == (1, 3, 256, 512)`.
  3. **Validasi Ketat Parameter `size`:**
     - `size` harus bernilai `None` atau `tuple` berisi tepat 2 integer murni (`int`, bukan `bool`) dengan `height > 0` dan `width > 0`.
     - Input tidak valid (misal nilai $\le 0$, panjang tuple $\neq 2$, tipe `list`/`int` tunggal/`float`/`bool`) wajib ditolak dengan `ValueError`.
  4. **Kebijakan Alur Resize (Mencegah Double Resize):**
     - Metode interpolasi: *Direct exact resize* ke `(H, W)` menggunakan `Image.Resampling.LANCZOS` (tanpa *cropping*), sehingga gambar konten dan gaya memiliki ukuran tensor `(1, 3, H, W)` yang identik jika diberi `size` yang sama.
     - Pada `load_image(image_path, size=None)`:
       - `size=None`: ukuran asli dipertahankan.
       - `size=(H, W)`: gambar di-resize ke tinggi `H` dan lebar `W`.
     - Pada `preprocess_image(image, size=(512, 512), device="cpu") -> torch.Tensor`:
       - Default `size=(512, 512)`.
       - Jika `image` berupa `str | Path`: memanggil `load_image(image, size=size)` (resize cukup terjadi sekali di `load_image`).
       - Jika `image` berupa `PIL.Image.Image`: memastikan mode `"RGB"`, lalu jika `size is not None` dan `(image.height, image.width) != size`, melakukan resize `LANCZOS` ke `(H, W)`. Jika `size is None`, mempertahankan ukuran gambar apa adanya.
       - Mengonversi ke tensor `torch.float32` skala `[0.0, 1.0]`, menormalisasi dengan konstanta ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$), menambahkan dimensi batch `(1, 3, H, W)`, dan memindahkan ke `device`.
  5. **Kontrak `deprocess_image(tensor: torch.Tensor) -> Image.Image`:**
     - Menerima tensor 4D `(1, 3, H, W)` atau 3D `(3, H, W)` (selain itu lempar `ValueError`).
     - Tidak mengubah tensor input secara *in-place* (`detach().cpu().clone()`).
     - Denormalisasi ($x \times \sigma + \mu$), *clamping* ke `[0.0, 1.0]`, konversi ke `uint8` `[0, 255]`, dan mengembalikan `PIL.Image.Image` RGB dengan `height == H` dan `width == W`.
  6. **Transisi Unit Test S02-02 ke S02-03:**
     - Tes `test_resize_not_silently_ignored` dari S02-02 yang menguji `NotImplementedError` diperbarui pada S02-03 menjadi pengujian resize ukuran non-persegi `(H, W)` dan validasi `ValueError` untuk `size` tidak valid.
- **Alternatif:** Menggunakan urutan `(W, H)` milik Pillow pada `load_image` dan `(H, W)` pada `preprocess_image`; ditolak keras karena pasti menyebabkan bug tertukarnya tinggi dan lebar.
- **Persetujuan:** Disetujui oleh Samuel Aaron (A1 - Ketua / Arsitek) dan Marthin M. Tambunan (A2).
- **Dampak:** `docs/ARCHITECTURE.md`, `src/preprocessing.py`, `tests/test_preprocessing.py`.
- **Pencatatan Sistem:** 2026-10-25T03:52:00+07:00 (WIB).
