# 📊 Master Progress Tracker — Neural Style Transfer (NST)

Dokumen ini adalah **pusat kendali dan status resmi (*Single Source of Truth*)** untuk seluruh tahapan kerja proyek. Setiap anggota tim wajib memperbarui status langkahnya di dokumen ini dan menyertakan referensi Git commit saat handoff.

---

## 👥 Informasi Tim & Peran

| ID | Nama Anggota | NIM | Peran Utama | File Rencana Kerja |
| :---: | :--- | :---: | :--- | :--- |
| **A1** | **Samuel Aaron** | `103012580024` | Ketua, Prompt Engineer & Arsitek | [A1_PLAN.md](A1_PLAN.md) |
| **A2** | **Marthin M. Tambunan** | `103012580055` | Data Pipeline & Preprocessing Engineer | [A2_PLAN.md](A2_PLAN.md) |
| **A3** | **Hafiz Fajar Ramadhan** | `103012430027` | Loss Function & Optimization Engineer | [A3_PLAN.md](A3_PLAN.md) |
| **A4** | **Muhammad Rafiq Abdurrasyid** | `103012400240` | QA, Dokumentasi & Error Handler | [A4_PLAN.md](A4_PLAN.md) |

---

## 🚦 Status Global per Stage (Stage-Gate Sequential)

| Stage | Nama Tahapan | PIC Utama | Status | Syarat Lolos (*Gate*) |
| :---: | :--- | :---: | :---: | :--- |
| **Stage 0** | Planning & Design | A1 (Lead) & Tim | `COMPLETED` | Seluruh dokumen master di `docs/` disetujui. |
| **Stage 1** | Project Scaffolding | A1 | `COMPLETED` | Struktur folder, virtualenv, dan testing awal siap. |
| **Stage 2** | Data Pipeline | A2 | `COMPLETED` | Preprocessing selesai dan lolos `test_preprocessing.py`. |
| **Stage 3** | VGG-19 Feature Extractor | A1 | `COMPLETED` | Layer VGG-19 beku tervalidasi bentuk tensornya. |
| **Stage 4** | Loss & Optimization | A3 | `COMPLETED` | Gram Matrix, Content/Style Loss teruji numerik. |
| **Stage 5** | QA & Dokumentasi | A4 | `IN_PROGRESS` | Integration test lulus tanpa regresi, log terverifikasi. |
| **Stage 6** | Integrasi & Evaluasi | A1 & Tim | `PENDING` | NST berjalan end-to-end menghasilkan gambar valid. |
| **Stage 7** | Finalisasi Proyek | Seluruh Tim | `PENDING` | Laporan akhir, repositori bersih, siap presentasi. |

---

## 📋 Matriks Rinci 23 Langkah Kerja (*Step-by-Step Tracker*)

> **Keterangan Status:**
> * `COMPLETED` : Langkah selesai, diuji, dan disetujui.
> * `IN_PROGRESS`: Sedang aktif dikerjakan oleh PIC saat ini.
> * `READY`      : Seluruh dependensi selesai, siap dikerjakan oleh PIC.
> * `PENDING`    : Menunggu tahap sebelumnya selesai (*blocked by dependency*).

| Step ID | PIC | Nama Langkah Aktivitas | Dependensi | Status | Unit / Smoke Test | Tautan Log Interaksi |
| :--- | :---: | :--- | :---: | :---: | :--- | :--- |
| **S00-01** | A1 | Analisis Kebutuhan (PRD & SRS) | - | `COMPLETED` | Validasi dokumen | [S00-02](../../logs/A1/S00-02-LOG.md) |
| **S00-02** | A1 | Perancangan Arsitektur & Kontrak Modul | S00-01 | `COMPLETED` | Mermaid review | [S00-02](../../logs/A1/S00-02-LOG.md) |
| **S00-03** | A4 | Kebijakan Logging AI & Format Log | S00-02 | `COMPLETED` | Pytest skrip logging: 8 passed; disetujui A1 | [S00-03](../../logs/A4/S00-03-LOG.md) / [Review A1](../../logs/A1/S05-01-REVIEW-LOG.md) |
| **S01-01** | A1 | Struktur Folder & Git Scaffolding | Stage 0 | `COMPLETED` | Folder & Git tree | [S01-01](../../logs/A1/S01-01-LOG.md) |
| **S01-02** | A1 | Coding Standards & Pytest Config | S01-01 | `COMPLETED` | `pytest -q` (pass) | [S01-01](../../logs/A1/S01-01-LOG.md) |
| **S02-01** | A2 | Perencanaan Format Data & Pasangan Gambar | Stage 1 | `COMPLETED` | Kedua aset valid; A2 meminta lanjut S02-02 | [Catatan transisi](../../logs/A2/S02-02-LOG.md) |
| **S02-02** | A2 | Implementasi Image Loader (Pillow/RGB) | S02-01 | `COMPLETED` | 8 tes unittest lulus; disetujui A1 | [S02-02](../../logs/A2/S02-02-LOG.md) |
| **S02-03** | A2 | Image Resize, Tensor & Normalisasi VGG | S02-02 | `COMPLETED` | Pytest: 19 passed, 1 skipped (CUDA), 35 subtests passed; smoke 6 kasus lulus; disetujui A1 | [S02-03](../../logs/A2/S02-03-LOG.md) |
| **S02-04** | A2 | Unit Testing & Handoff Preprocessing | S02-03 | `COMPLETED` | Opsi B CPU: pytest 21 passed, 1 skipped CUDA, 43 subtests; 6 smoke lulus; disetujui A1 | [S02-04](../../logs/A2/S02-04-LOG.md) / [Review A1](../../logs/A1/S02-04-REVIEW-LOG.md) |
| **S03-01** | A1 | Load VGG-19 Pretrained & Freezing Bobot | Stage 2 | `COMPLETED` | VGG-19[:29] frozen, ReLU inplace=False | [S03-01](../../logs/A1/S03-01-LOG.md) |
| **S03-02** | A1 | Ekstraksi Layer Content & Style | S03-01 | `COMPLETED` | `pytest`: 8 passed, 1 skipped (CUDA), 17 subtests passed | [S03-02](../../logs/A1/S03-02-LOG.md) |
| **S04-01** | A3 | Implementasi Matematis Gram Matrix | Stage 3 | `COMPLETED` | Pytest: 7 passed; disetujui A1 | [S04-01](../../logs/A3/S04-01-LOG.md) / [Review A1](../../logs/A1/S04-01-REVIEW-LOG.md) |
| **S04-02** | A3 | Formulasi Content Loss (MSE Fitur) | S04-01 | `COMPLETED` | Pytest: 20 passed (TestContentLoss); disetujui A1 | [S04-02](../../logs/A3/S04-02-LOG.md) / [Review A1](../../logs/A1/S04-02-03-REVIEW-LOG.md) |
| **S04-03** | A3 | Formulasi Style Loss (Multi-layer Gram) | S04-02 | `COMPLETED` | Pytest: 20 passed (TestStyleLoss & TestTotalLoss); disetujui A1 | [S04-03](../../logs/A3/S04-03-LOG.md) / [Review A1](../../logs/A1/S04-02-03-REVIEW-LOG.md) |
| **S04-04** | A3 | Optimization Loop Piksel (L-BFGS/Adam) | S04-03 | `COMPLETED` | Pytest: 9 passed (TestRunStyleTransfer); disetujui A1 | [S04-04](../../logs/A3/S04-04-LOG.md) / [Review A1](../../logs/A1/S04-04-05-REVIEW-LOG.md) |
| **S04-05** | A3 | Testing & Tuning Hyperparameter Loss | S04-04 | `COMPLETED` | Pytest: 58 passed repositori; analisis kompleksitas selesai; disetujui A1 | [S04-05](../../logs/A3/S04-05-LOG.md) / [Review A1](../../logs/A1/S04-04-05-REVIEW-LOG.md) |
| **S05-01** | A4 | Audit Kepatuhan Kode & Standar PEP 8 | Stage 4 | `COMPLETED` | pycodestyle: 228 E501, 2 E901; pytest 66 passed; disetujui A1 | [S05-01](../../logs/A4/S05-01-LOG.md) / [Review A1](../../logs/A1/S05-01-REVIEW-LOG.md) |
| **S05-02** | A4 | Validasi & Konsolidasi Log Interaksi AI | S05-01 | `IN_PROGRESS` (menunggu review A1) | 23 log diaudit; `docs/LOG_INTERAKSI.md` jadi indeks resmi; JSONL 4 → 25 entri | [S05-02](../../logs/A4/S05-02-LOG.md) / [Indeks](../LOG_INTERAKSI.md) |
| **S05-03** | A4 | Integration Testing & Error Troubleshooting | S05-02 | `IN_PROGRESS` (menunggu review A1) | `test_integration.py` 16 passed; pytest 82 passed, 2 skipped; 0 E901 | [S05-03](../../logs/A4/S05-03-LOG.md) |
| **S06-01** | A1 | Integrasi End-to-End Pipeline NST | Stage 5 | `PENDING` | `python main.py` | [logs/A1/](../../logs/A1/) |
| **S06-02** | A1 | Evaluasi Output Gambar & Grafik Loss | S06-01 | `PENDING` | Visual inspection | [logs/A1/](../../logs/A1/) |
| **S07-01** | A4 | Finalisasi Laporan Proyek & Dokumentasi | Stage 6 | `PENDING` | Audit dokumen | [logs/A4/](../../logs/A4/) |
| **S07-02** | Semua| Final Acceptance & Persiapan Presentasi | S07-01 | `PENDING` | Demo rehearsal | Seluruh Tim |

---

## 🔄 Aturan Protokol Perubahan Planning di Tengah Jalan

Jika di tengah pengerjaan terjadi kendala teknis (misal: *CUDA Out of Memory*, optimizer L-BFGS lambat, atau perubahan bobot loss):

1. **Dilarang langsung mengubah kode secara sepihak.**
2. Catat usulan perubahan pada [docs/DECISIONS.md](../DECISIONS.md) dengan format *Architecture Decision Record (ADR)*:
   * **Konteks:** Mengapa perubahan dibutuhkan.
   * **Keputusan:** Solusi baru yang disepakati.
   * **Dampak:** File apa saja yang terpengaruh.
   * **Persetujuan:** Nama anggota yang menyetujui.
3. Perbarui status atau langkah pada tabel di atas.
4. Buat satu commit Git khusus untuk perubahan planning:
   ```bash
   git commit -m "docs(planning): update tracking status and record decision for [topik]"
   ```
