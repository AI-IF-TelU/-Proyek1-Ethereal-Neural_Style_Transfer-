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

### Stage 0: Planning & Architecture (`COMPLETED` ✅)
* **S00-01: Requirement Analysis** — `COMPLETED` ✅
* **S00-02: Architecture & Algorithm Design** — `COMPLETED` ✅

### Stage 1: Project Scaffolding (`COMPLETED` ✅)
* **S01-01: Project Structure** — `COMPLETED` ✅
* **S01-02: Project Standards** — `COMPLETED` ✅

### Stage 3: VGG-19 Feature Extraction (`COMPLETED` ✅)
* **S03-01: Feature Extractor Implementation (Kebijakan D-006)** — `COMPLETED` ✅ ([logs/A1/S03-01-LOG.md](../../logs/A1/S03-01-LOG.md))
* **S03-02: Feature Extractor Testing & Handoff Stage 3** — `COMPLETED` ✅ ([logs/A1/S03-02-LOG.md](../../logs/A1/S03-02-LOG.md))

### Stage 6: Integration & Evaluation (`COMPLETED` ✅)
* **S06-01: Module Integration (Kebijakan D-009)**
  * *Tujuan:* Menggabungkan preprocessing (A2), feature extractor (A1), loss & optimizer (A3), dan visualisasi (A1) ke dalam titik masuk utama `main.py` serta `config/config.yaml`.
  * *Target Berkas:* `src/visualization.py`, `config/config.yaml`, `main.py`, `tests/test_visualization.py`, `logs/A1/S06-01-LOG.md`
  * *Status:* `COMPLETED` ✅ (`pytest`: 90 passed, 2 skipped CUDA; `0 E501` pada seluruh berkas Stage 6).
* **S06-02: Evaluation & Experimentation**
  * *Tujuan:* Menjalankan eksperimen transfer gaya *end-to-end* (`256x256`, L-BFGS 50 langkah, $\alpha=1.0, \beta=10^4$), melakukan *tuning* `learning_rate` (`1.0` vs `0.5`), dan menyimpan artefak citra hasil serta grafik penurunan *loss*.
  * *Target Berkas:* `assets/content/content_sample.png`, `assets/style/style_sample.png`, `outputs/images/stylized_output.png`, `outputs/images/comparison_grid.png`, `outputs/graphs/loss_curve.png`, `logs/A1/S06-02-LOG.md`
  * *Status:* `COMPLETED` ✅ (Penurunan *Total Loss* monoton `-61.34%` dari `9.8481` ke `3.8076`; diserahkan ke Anggota A4 pada **Handoff #10** untuk Stage 7).
