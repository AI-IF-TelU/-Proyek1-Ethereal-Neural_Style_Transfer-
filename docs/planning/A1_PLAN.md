# 📘 Panduan & Rencana Kerja Anggota A1
### Samuel Aaron (NIM: 103012580024)
**Peran:** Ketua, Prompt Engineer & Software Architect

---

## 🎯 Tanggung Jawab Utama
1. Mengarahkan keselarasan logika seluruh arsitektur proyek.
2. Membangun modul ekstraksi fitur VGG-19 beku (`src/feature_extractor.py`).
3. Memimpin integrasi end-to-end (`main.py`) dan evaluasi final model.

---

## 📋 Rincian Langkah Kerja (Step-by-Step)

### Stage 0: Planning & Architecture
* **S00-01: Requirement Analysis**
  * *Tujuan:* Menganalisis PRD dan SRS untuk menetapkan fitur wajib, batasan, dan kriteria keberhasilan.
  * *Status:* `COMPLETED` ✅
* **S00-02: Architecture & Algorithm Design**
  * *Tujuan:* Merancang diagram arsitektur, modul-modul utama, kontrak antarmuka, dan dimensi tensor.
  * *Status:* `COMPLETED` ✅

### Stage 1: Project Scaffolding
* **S01-01: Project Structure**
  * *Tujuan:* Menyiapkan folder repositori, Git environment, dan file placeholder.
  * *Status:* `COMPLETED` ✅
* **S01-02: Project Standards**
  * *Tujuan:* Menetapkan standar PEP 8, konfigurasi testing `pytest`, dan dependensi `requirements.txt`.
  * *Status:* `COMPLETED` ✅

### Stage 3: VGG-19 Feature Extraction (Aktif — Stage 2 Selesai)
* **S03-01: Feature Extractor Implementation (Kebijakan D-006)**
  * *Tujuan:*
    1. Memuat blok konvolusi `torchvision.models.vgg19(weights=weights).features` (default `VGG19_Weights.DEFAULT`, mendukung `weights=None` untuk unit test cepat/offline).
    2. Memotong (*truncate*) sekuens `features[:29]` hanya hingga layer terakhir yang dibutuhkan (`conv5_1` di indeks `'28'`), mengganti seluruh `nn.ReLU(inplace=True)` menjadi `nn.ReLU(inplace=False)` demi keamanan *autograd* dan keutuhan aktivasi layer `Conv2d`.
    3. Membekukan seluruh parameter (`requires_grad = False`), mengunci mode `.eval()`, dan memindahkan model ke `device`.
    4. Menyediakan `get_features(image_tensor, model, layers=None)` yang memvalidasi tensor 4D `(N, 3, H, W)` bertipe floating dan mengekstrak dictionary fitur untuk layer konten (`'21': 'conv4_2'`) dan gaya (`'0': 'conv1_1'`, `'5': 'conv2_1'`, `'10': 'conv3_1'`, `'19': 'conv4_1'`, `'28': 'conv5_1'`) **tanpa** membungkus *forward pass* dalam `torch.no_grad()` agar gradien dari *loss* tetap mengalir ke tensor gambar hasil (`generated_img`).
  * *Target Berkas:* `src/feature_extractor.py`
  * *Dependensi:* Stage 2 selesai (`COMPLETED`).
  * *Status:* `READY` 🟢 (Spesifikasi D-006 telah disetujui dan siap diimplementasikan).
* **S03-02: Feature Extractor Testing & Handoff Stage 3**
  * *Tujuan:*
    1. Menyusun unit test komprehensif pada `tests/test_feature_extractor.py` untuk menguji pembekuan bobot (`requires_grad=False`), mode `.eval()`, penggantian `ReLU(inplace=False)`, bentuk tensor fitur keluaran (`conv1_1`..`conv5_1`, `conv4_2`) pada dimensi persegi & non-persegi, aliran gradien ke `image_tensor` (`image_tensor.grad is not None`) sementara parameter VGG tetap `grad is None`, serta validasi error input.
    2. Memperbarui analisis kompleksitas waktu dan ruang bagian Feature Extractor pada `docs/COMPLEXITY_ANALYSIS.md`.
    3. Mencatat log `logs/A1/S03-01-LOG.md` & `logs/A1/S03-02-LOG.md` serta serah terima ke A3 (Stage 4) di `docs/HANDOFF.md`.
  * *Target Berkas:* `tests/test_feature_extractor.py`, `docs/COMPLEXITY_ANALYSIS.md`, `docs/HANDOFF.md`
  * *Status:* `PENDING`

### Stage 6: Integration & Evaluation
* **S06-01: Module Integration**
  * *Tujuan:* Menggabungkan preprocessing (A2), feature extractor (A1), loss & optimizer (A3) ke dalam titik masuk utama.
  * *Target Berkas:* `main.py`
  * *Acceptance Criteria:* Perintah `python main.py` berjalan dari awal hingga selesai tanpa runtime crash.
* **S06-02: Evaluation & Experimentation**
  * *Tujuan:* Melakukan uji coba transfer gaya, tuning rasio $\alpha/\beta$, dan menghasilkan plot grafik penurunan loss.
  * *Target Berkas:* `outputs/images/` dan `outputs/graphs/`

---

## 💬 Template Prompt untuk AI Agent A1

Saat Anda berinteraksi dengan AI di VS Code, gunakan template ini agar AI tetap fokus:

```text
Saya adalah Samuel Aaron (Anggota A1 - Arsitek & Lead).
Proyek: Neural Style Transfer (PyTorch, VGG-19 Frozen, Vibe Coding).
Referensi Dokumen: docs/planning/PROGRESS_TRACKER.md, docs/ARCHITECTURE.md, dan docs/DECISIONS.md (D-006).

Langkah Aktif: [Misal: S03-01 & S03-02 - Ekstraksi Fitur VGG-19]
Tujuan: Mengimplementasikan dan menguji src/feature_extractor.py dan tests/test_feature_extractor.py sesuai kontrak D-006.
Batasan: Bobot VGG-19 wajib dibekukan (requires_grad = False) dan ReLU inplace=False. Jangan mengubah modul preprocessing (milik A2) atau losses (milik A3).
```
