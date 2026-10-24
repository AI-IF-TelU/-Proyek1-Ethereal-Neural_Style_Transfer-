# S02-03-ARCH-SPEC: Perumusan Kontrak Preprocessing Tensor & Keputusan D-005

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
- **Phase:** Architecture Specification & Decision Making

---

## 0. TAHAP PLANNING (Analisis Kebutuhan Kontrak S02-03)

### Prompt Kelompok
> "Sebagai Software Architect, periksa apakah kontrak teknis untuk tahap S02-03 (Resize, Tensor Conversion & Normalisasi ImageNet) sudah cukup jelas dan selaras antara `docs/DECISIONS.md` dan `docs/ARCHITECTURE.md`.
> Rancang spesifikasi fungsi, parameter input, tipe kembalian, dan konstanta normalisasi ImageNet agar Anggota A2 memiliki dasar kontrak resmi sebelum meminta kode implementasi ke AI-nya."

### Alasan Khusus
Mencegah terjadinya ambiguitas antarmuka (*interface mismatch*): Memastikan A2 menghasilkan tensor 4D `(1, 3, H, W)` bertipe `torch.float32` dengan normalisasi ImageNet standar yang siap dikonsumsi langsung oleh modul `feature_extractor.py` (milik A1) pada Stage 3.

---

## 1. TAHAP PERUMUSAN KEPUTUSAN ARSITEKTUR (D-005)

### Poin-Poin Keputusan yang Ditetapkan:
1. **Fungsi `preprocess_image(image, target_size=(512, 512), device='cpu') -> torch.Tensor`:**
   * Menerima format `PIL.Image.Image`, `str`, atau `pathlib.Path`.
   * Melakukan resize ke `target_size` (default $512 \times 512$).
   * Konversi tensor 4D `(1, 3, H, W)` dengan dtype `torch.float32`.
   * Normalisasi standar ImageNet: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
2. **Fungsi `deprocess_image(tensor: torch.Tensor) -> Image.Image`:**
   * Menghapus dimensi batch `(3, H, W)`.
   * Denormalisasi ($x \times \sigma + \mu$).
   * Clamping ke rentang $[0.0, 1.0]$.
   * Konversi ke `PIL.Image.Image` (RGB uint8).
3. **Pembaruan `load_image(image_path, size=None)`:**
   * Parameter `size` resmi diaktifkan untuk meneruskan ukuran resize berkualitas tinggi (`Image.Resampling.LANCZOS`).

---

## 2. PENYELARASAN DOKUMEN MASTER

1. **Pembaruan [docs/DECISIONS.md](../../docs/DECISIONS.md):**
   * Dicatat keputusan **`D-005`** dengan status *Disetujui oleh Samuel Aaron (A1) & Marthin M. Tambunan (A2)*.
2. **Pembaruan [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md):**
   * Bagian 2.A diperluas memuat signature fungsi lengkap, tipe parameter, dan penanganan exception.
3. **Pemberian Persetujuan Gerbang (*Gate Approval*):**
   * A1 secara resmi memberikan izin kepada A2 untuk membuka tahap S02-03.

---

## 3. TESTING, REVIEW & HANDOFF
- **Validasi:** Kontrak fungsi diuji kesesuaiannya dengan kebutuhan input layer VGG-19 ImageNet.
- **Hasil Review:** Selaras 100% dan bebas konflik arsitektur.
- **File Hasil:** `docs/DECISIONS.md`, `docs/ARCHITECTURE.md`, `logs/A1/S02-03-ARCH-SPEC-LOG.md`.
- **Handoff:** Tahap S02-03 berstatus `READY` untuk dieksekusi oleh A2.
