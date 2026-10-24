# DECISIONS (Architecture Decision Records)

| ID | Keputusan | Alasan | Alternatif | Disetujui |
|---|---|---|---|---|
| D-001 | VGG-19 pretrained frozen | Standar NST Gatys et al. | Train from scratch | Seluruh Tim (A1 Lead) |
| D-002 | Optimasi pixel generated image | Mempertahankan bobot model tetap beku | Fine-tune weights | Seluruh Tim (A1 Lead) |
| D-003 | Modular struktur (5 modul src) | Kolaborasi multi-agent terisolasi | Monolitik satu skrip | Seluruh Tim (A1 Lead) |
| D-004 | Batas scope image loader S02-02 | Penolakan parameter size hingga S02-03 | Resize di loader | A1 (Samuel) & A2 (Marthin) |
| D-005 | Spesifikasi Preprocessing & Normalisasi S02-03 | Standardisasi input tensor VGG-19 | Normalisasi custom | A1 (Samuel) & A2 (Marthin) |

---

## D-004: Batas loader S02-02

- **Konteks:** Signature `load_image(image_path, size=None)` sudah direncanakan, tetapi resize termasuk S02-03 dan perilaku parameter belum dijelaskan.
- **Keputusan:** Hasil berupa `PIL.Image` RGB mandiri, ukuran asli saat `size=None`. Nilai `size` selain None ditolak dengan `NotImplementedError` sampai S02-03.
- **Alternatif:** Mengimplementasikan resize pada loader sekarang; tidak dipilih untuk mematuhi *Stage-Gate Sequential*.
- **Persetujuan:** Disetujui oleh A2 dan diverifikasi/diterima oleh A1 (Samuel Aaron).
- **Dampak:** `src/preprocessing.py`, `tests/test_preprocessing.py`; tidak mengubah modul model/loss.
- **Pencatatan Sistem:** 2026-10-25T01:57:30+07:00 (WIB).

---

## D-005: Kontrak Spesifikasi Preprocessing & Normalisasi Tensor S02-03

- **Konteks:** Tahap S02-03 memerlukan konversi gambar PIL menjadi PyTorch Tensor yang kompatibel dengan VGG-19 serta fungsi deprocessing untuk mengembalikan tensor menjadi gambar.
- **Keputusan Teknis:**
  1. **Fungsi `preprocess_image(image, target_size=(512, 512), device="cpu") -> torch.Tensor`:**
     - Menerima `PIL.Image.Image`, `str`, atau `pathlib.Path`.
     - Mengubah ukuran menjadi `target_size` (default $512 \times 512$).
     - Mengonversi ke tensor dengan dimensi batch 4D: `(1, 3, H, W)` bertipe `torch.float32`.
     - Melakukan normalisasi standar ImageNet: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
  2. **Fungsi `deprocess_image(tensor: torch.Tensor) -> Image.Image`:**
     - Menghapus batch dimension: `(3, H, W)`.
     - Melakukan denormalisasi ($x \times \sigma + \mu$).
     - Melakukan pembatasan nilai (*clamping*) ke rentang $[0.0, 1.0]$.
     - Mengonversi kembali ke `PIL.Image.Image` (RGB uint8).
  3. **Pembaruan `load_image(image_path, size=None)`:**
     - Jika `size is not None`, gambar di-resize menggunakan Pillow (`image.resize(size, Image.Resampling.LANCZOS)`).
- **Alternatif:** Normalisasi manual di dalam loop optimasi; ditolak karena melanggar modularitas.
- **Persetujuan:** Disetujui oleh Samuel Aaron (A1 - Ketua / Arsitek) dan Marthin M. Tambunan (A2).
- **Dampak:** Berkas `src/preprocessing.py`, `tests/test_preprocessing.py`.
- **Pencatatan Sistem:** 2026-10-25T03:30:00+07:00 (WIB).
