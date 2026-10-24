# S02-03-ARCH-SPEC: Perumusan Kebijakan Tunggal `size=(H, W)`, Resize & Kontrak Tensor (D-005)

## Metadata
- **Nama Kelompok:** Proyek 1 - Ethereal Neural Style Transfer
- **Metodologi:** SDLC + Stage-Gate Sequential + Vibe Engineering
- **Tools:** VS Code, Git, Python, PyTorch
- **Tanggal Sesi:** 2026-10-25
- **Anggota:** A1 — Samuel Aaron (Ketua, Prompt Engineer & Software Architect)
- **Target Modul:** Penyelarasan Kontrak Arsitektur Preprocessing Tensor (`docs/ARCHITECTURE.md`) dan Pencatatan Keputusan D-005 (`docs/DECISIONS.md`)
- **AI Model:** Gemini Assistant
- **Tautan Sesi Chat:** Sesi aktif VS Code AI Assistant
- **Step ID:** S02-03 Architecture Specification & Gate Approval
- **Phase:** Architecture Specification, Policy Alignment & Decision Making

---

## 0. TAHAP PLANNING & AUDIT KONSISTENSI KONTRAK

### Prompt Kelompok
> "Sebagai Software Architect, cek lagi mengenai keselarasan arti dari `size` dan `target_size` agar tidak tertukar. Detail resize pada preprocessing cek juga sehingga mengikuti satu kebijakan yang jelas sebelum Anggota A2 melakukan implementasi S02-03."

### Analisis Risiko & Temuan Audit oleh A1:
1. **Inkonsistensi Nama Parameter (`size` vs `target_size`):**
   * Pada `load_image` (S02-02) digunakan nama parameter `size`, sedangkan draf awal `preprocess_image` menggunakan `target_size`. Penggunaan dua istilah berbeda untuk ukuran gambar berpotensi menimbulkan kebingungan.
2. **Risiko Tertukarnya Dimensi Pillow `(W, H)` vs PyTorch `(H, W)`:**
   * Pustaka Pillow `PIL.Image.resize` mengharapkan tuple `(width, height) = (W, H)`.
   * Sebaliknya, PyTorch Tensor `[N, C, H, W]` menggunakan urutan `(height, width) = (H, W)`.
   * Jika `size=(256, 512)` diteruskan mentah ke `image.resize(size)`, hasil tensornya menjadi `(1, 3, 512, 256)` (tinggi dan lebar terbalik!).
3. **Potensi *Double Resize*:**
   * Jika `load_image` melakukan resize dan `preprocess_image` juga melakukan resize tanpa pemeriksaan, gambar berisiko diinterpolasi dua kali.

---

## 1. TAHAP PERUMUSAN KEPUTUSAN ARSITEKTUR TUNGGAL (D-005)

### Kebijakan Tunggal yang Ditetapkan:
1. **Satu Nama Parameter (`size`):**
   * Seluruh fungsi di `src/preprocessing.py` (`load_image` dan `preprocess_image`) wajib menggunakan nama parameter **`size: tuple[int, int] | None`**. Istilah `target_size` dihapus sepenuhnya.
2. **Konvensi Urutan Dimensi `size = (height, width)` = `(H, W)`:**
   * Di seluruh API `src/preprocessing.py`, `size` **selalu berarti `(height, width) = (H, W)`** mengikuti konvensi PyTorch `(1, 3, H, W)`.
   * Saat memanggil `PIL.Image.resize` secara internal, tuple wajib dibalik menjadi `(width, height)`:
     `image.resize((width, height), Image.Resampling.LANCZOS)`.
   * Dengan demikian, input `size=(256, 512)` menjamin `pil_img.height == 256`, `pil_img.width == 512`, dan `tensor.shape == (1, 3, 256, 512)`.
3. **Validasi Ketat Parameter `size`:**
   * Hanya menerima `None` atau `tuple` berisi tepat 2 integer murni (`not isinstance(v, bool)`) dengan `H > 0` dan `W > 0`. Selain itu wajib melempar `ValueError`.
4. **Kebijakan Alur Resize (Tanpa Double Resize):**
   * Metode interpolasi: *Direct exact resize* ke `(H, W)` menggunakan `Image.Resampling.LANCZOS` (tanpa *cropping*).
   * `load_image(image_path, size=None) -> Image.Image`: jika `size=(H, W)` diberikan, melakukan resize ke tinggi `H` dan lebar `W`.
   * `preprocess_image(image, size=(512, 512), device='cpu') -> torch.Tensor`:
     * Jika `image` berupa `str | Path`: memanggil `load_image(image, size=size)` (resize hanya terjadi 1 kali).
     * Jika `image` berupa `PIL.Image.Image`: memastikan mode `"RGB"`, dan hanya melakukan resize jika `size is not None` serta `(image.height, image.width) != size`.
     * Mengonversi ke tensor `torch.float32` `[0.0, 1.0]`, menormalisasi dengan ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$), dan menghasilkan tensor 4D `(1, 3, H, W)` pada `device`.
5. **Fungsi `deprocess_image(tensor: torch.Tensor) -> Image.Image`:**
   * Menerima tensor 4D `(1, 3, H, W)` atau 3D `(3, H, W)`, bekerja tanpa mengubah tensor input (*out-of-place* via `detach().cpu().clone()`), denormalisasi, *clamping* ke `[0.0, 1.0]`, dan mengembalikan `PIL.Image` RGB (`height == H, width == W`).

---

## 2. PENYELARASAN DOKUMEN MASTER

1. **Pembaruan [docs/DECISIONS.md](../../docs/DECISIONS.md):** Mencatat kebijakan tunggal `D-005` secara lengkap beserta aturan transisi unit test dari S02-02 ke S02-03.
2. **Pembaruan [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md):** Menyelaraskan Bagian 2.A dengan signature `size=(H, W)`.
3. **Pembaruan [docs/planning/A2_PLAN.md](../../docs/planning/A2_PLAN.md) & [logs/A2/S02-03-LOG.md](../A2/S02-03-LOG.md):** Menyesuaikan prompt planning dan persetujuan S02-03 agar A2 langsung mengikuti kebijakan tunggal ini.

---

## 3. TESTING, REVIEW & HANDOFF
- **Validasi:** Seluruh dokumen master (`DECISIONS.md`, `ARCHITECTURE.md`, `A2_PLAN.md`, `S02-03-LOG.md`) telah diperiksa dan 100% bebas dari istilah `target_size` maupun ambiguitas `(W, H)` vs `(H, W)`.
- **Handoff:** Tahap S02-03 siap diimplementasikan oleh Anggota A2 (Marthin M. Tambunan) tanpa risiko salah tafsir dimensi.
