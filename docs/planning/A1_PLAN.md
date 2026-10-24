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

### Stage 3: VGG-19 Feature Extraction (Menunggu Stage 2 Selesai)
* **S03-01: Feature Extractor Implementation**
  * *Tujuan:* Memuat VGG-19 pretrained dari `torchvision.models`, membekukan parameter (`requires_grad = False`), dan memotong layer untuk konten (`conv4_2`) dan gaya (`conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`).
  * *Target Berkas:* `src/feature_extractor.py`
  * *Dependensi:* Stage 2 selesai (data loader A2 teruji).
  * *Acceptance Criteria:* Model mengembalikan dictionary tensor fitur sesuai layer yang diminta tanpa mengubah bobot VGG-19.
* **S03-02: Feature Extractor Testing**
  * *Tujuan:* Menjalankan unit test untuk memvalidasi shape tensor output fitur dan memastikan gradien tidak mengalir ke bobot VGG.
  * *Target Berkas:* `tests/test_feature_extractor.py`
  * *Acceptance Criteria:* `pytest tests/test_feature_extractor.py` lolos 100%.

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
Referensi Dokumen: docs/planning/PROGRESS_TRACKER.md dan docs/ARCHITECTURE.md.

Langkah Aktif: [Misal: S03-01 - Ekstraksi Fitur VGG-19]
Tujuan: [Tuliskan apa yang ingin dibuat/diuji]
Batasan: Bobot VGG-19 wajib dibekukan (requires_grad = False). Jangan mengubah modul preprocessing (milik A2) atau losses (milik A3).

Berikan:
1. Rencana implementasi dan kontrak fungsi get_features().
2. Analisis tensor shape input dan output.
3. Kode implementasi untuk src/feature_extractor.py dan unit test-nya.
```
