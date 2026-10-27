# 📕 Panduan & Rencana Kerja Anggota A4
### Muhammad Rafiq Abdurrasyid (NIM: 103012400240)
**Peran:** Quality Assurance, Dokumentasi & Error Handler

---

## 🎯 Tanggung Jawab Utama
1. Mengelola dan mengaudit riwayat percakapan AI (*Vibe Coding logs*) seluruh anggota pada folder `logs/`.
2. Melakukan *Code Review* terhadap kode yang dihasilkan oleh A1, A2, dan A3 agar patuh pada `docs/CODING_STANDARDS.md`.
3. Menjalankan *Integration Testing* lintas-modul dan menyelesaikan kendala error/bug (*Troubleshooting*).
4. Menyusun laporan akhir kelompok dan materi kesiapan presentasi pada Stage 7.

---

## 📋 Rincian Langkah Kerja

### Stage 0: Dokumentasi & Logging Policy
* **S00-03: AI Logging Policy & Scaffolding Log**
  * *Tujuan:* Menyiapkan folder `logs/A1` s.d. `logs/A4`, template [LOG_TEMPLATE.md](../../logs/LOG_TEMPLATE.md), dan skrip [scripts/log_interaction.py](../../scripts/log_interaction.py).
  * *Status:* `COMPLETED` ✅
  * *Catatan Wajib Saat Memulai Stage 5:* Buat berkas `logs/A4/S00-03-LOG.md` agar bukti pekerjaan Stage 0 milik A4 tercatat secara eksplisit di dalam folder `logs/A4/`.

---

### Stage 5: QA, Code Review & Integration Testing (AKTIF — Stage 2, 3, dan 4 Selesai!)

> **Panduan Eksekusi Stage 5 untuk A4 (Muhammad Rafiq Abdurrasyid):**
> Seluruh modul inti telah selesai dibangun dan diuji per-unit oleh A2 (`src/preprocessing.py`), A1 (`src/feature_extractor.py`), dan A3 (`src/losses.py` & `src/optimization.py`) dengan total **58 passed, 2 skipped (CUDA), 60 subtests passed**. Tugas A4 di Stage 5 adalah melakukan audit kualitas menyeluruh tanpa merusak kontrak API yang sudah disepakati pada `D-001` s.d. `D-008`.

#### S05-01: Audit Kepatuhan Kode & Standar PEP 8
* **Status:** `READY` 🟢
* **Tujuan:**
  1. Memeriksa kepatuhan seluruh berkas produksi di `src/` (`preprocessing.py`, `feature_extractor.py`, `losses.py`, `optimization.py`) dan pengujian di `tests/` terhadap `docs/CODING_STANDARDS.md`:
     * Konvensi penamaan `snake_case` untuk fungsi/variabel, `PascalCase` untuk kelas, dan `UPPER_CASE` untuk konstanta.
     * Kelengkapan *type hints* dan *docstrings* (Args, Returns, Raises) pada seluruh fungsi publik.
     * Kepatuhan batas modul (*separation of concerns*) dan tidak adanya *hardcoded local path*.
  2. Mencatat hasil audit per-modul secara terstruktur pada **`logs/A4/S05-01-LOG.md`** (beserta **`logs/A4/S00-03-LOG.md`** untuk melengkapi arsip Stage 0 A4).
* **Target Berkas:** `logs/A4/S00-03-LOG.md`, `logs/A4/S05-01-LOG.md`

#### S05-02: Validasi & Konsolidasi Log Interaksi AI (Bobot Rubrik 25%)
* **Status:** `READY` 🟢 (Dapat dikerjakan langsung bersama/setelah `S05-01`)
* **Tujuan:**
  1. Mengaudit kelengkapan berkas log di `logs/A1/`, `logs/A2/`, `logs/A3/`, dan `logs/A4/` terhadap format `logs/LOG_TEMPLATE.md` (Metadata, `0. TAHAP PLANNING`, `1. TAHAP INISIASI / IMPLEMENTATION`, `2. TAHAP PENANGANAN EROR / DEBUGGING`, `3. TESTING, REVIEW & HANDOFF`).
  2. Memperbarui **`docs/LOG_INTERAKSI.md`** dengan indeks komprehensif seluruh log interaksi tim (tabel ringkasan per anggota A1..A4, tautan ke setiap file log di `logs/`, ringkasan kasus *debugging* utama yang berhasil diselesaikan di tiap stage, dan keputusan arsitektur `D-001` s.d. `D-008`).
  3. Mencatat hasil validasi dan konsolidasi log pada **`logs/A4/S05-02-LOG.md`**.
* **Target Berkas:** `docs/LOG_INTERAKSI.md`, `logs/A4/S05-02-LOG.md`

#### S05-03: Integration Testing & Handoff Stage 5 ke Stage 6 (A1)
* **Status:** `READY` 🟢 (Dapat dikerjakan langsung setelah `S05-02`)
* **Tujuan:**
  1. Menjalankan seluruh suite pengujian (`pytest tests/ -v -rs`) dan memverifikasi integrasi *end-to-end* lintas-modul di memori:
     `PIL.Image` $\rightarrow$ `preprocess_image` (A2) $\rightarrow$ `get_vgg19_model` & `get_features` (A1) $\rightarrow$ `gram_matrix`, `compute_content_loss`, `compute_style_loss`, `compute_total_loss` (A3) $\rightarrow$ `run_style_transfer` (A3) $\rightarrow$ `deprocess_image` (A2) menghasilkan `PIL.Image` RGB yang valid dan penurunan nilai loss yang konvergen.
  2. (Opsional / Direkomendasikan) Menambahkan pengujian integrasi *end-to-end* lintas-modul (`TestEndToEndPipelineIntegration`) pada berkas `tests/test_integration.py` (atau di dalam suite `tests/`) sehingga rantai penuh Stage 2 + Stage 3 + Stage 4 + Deprocess teruji secara otomatis oleh `pytest`.
  3. Mencatat log eksekusi pengujian integrasi pada **`logs/A4/S05-03-LOG.md`**, memperbarui status `S05-01` s.d. `S05-03` pada `docs/planning/PROGRESS_TRACKER.md` dan `docs/planning/A4_PLAN.md` menjadi `IN_PROGRESS (menunggu review A1)`, serta menambahkan draf **Handoff #8 (Penutupan Stage 5 $\rightarrow$ Stage 6)** pada `docs/HANDOFF.md`.
* **Target Berkas:** `tests/test_integration.py` (opsional), `logs/A4/S05-03-LOG.md`, `docs/HANDOFF.md`, `docs/planning/PROGRESS_TRACKER.md`, `docs/planning/A4_PLAN.md`

---

### Stage 7: Finalisasi Proyek & Acceptance (Setelah Stage 6 Selesai)
* **S07-01: Final Documentation & Report**
  * *Tujuan:* Memfinalisasi laporan proyek komprehensif (`docs/FINAL_REPORT.md`), merapikan `README.md`, dan menyusun rekap kontribusi anggota.
* **S07-02: Final Acceptance & Demo Checklist**
  * *Tujuan:* Menyusun `docs/DEMO_CHECKLIST.md` dan memastikan proyek siap didemonstrasikan di depan dosen/asisten.

---

## 💬 Template Prompt Siap Pakai untuk AI Agent A4 (Stage 5: `S05-01` s.d. `S05-03`)

Saat **Rafiq (A4)** berinteraksi dengan AI di VS Code, cukup *copy-paste* prompt berikut agar AI langsung mengeksekusi seluruh tugas Stage 5 sesuai aturan proyek:

```text
Saya adalah Muhammad Rafiq Abdurrasyid (Anggota A4 - QA, Dokumentasi & Error Handler, NIM: 103012400240).
Proyek: Tugas Besar Neural Style Transfer (PyTorch, VGG-19 Frozen, Metodologi Stage-Gate Sequential + Vibe Coding).

Sebelum bekerja, BACA TERLEBIH DAHULU berkas berikut di workspace:
1. `docs/planning/PROGRESS_TRACKER.md` dan `docs/planning/A4_PLAN.md`
2. `docs/CODING_STANDARDS.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md` (D-001 s.d. D-008), dan `docs/HANDOFF.md` (Handoff #1 s.d. #7)
3. `logs/LOG_TEMPLATE.md` serta contoh log di `logs/A1/`, `logs/A2/`, dan `logs/A3/`
4. Seluruh kode di `src/` (`preprocessing.py`, `feature_extractor.py`, `losses.py`, `optimization.py`) dan `tests/`

Langkah Aktif: Eksekusi Stage 5 Lengkap (`S00-03-LOG`, `S05-01`, `S05-02`, dan `S05-03`).
Kerjakan secara berurutan dengan ketentuan ketat berikut:
1. Arsip Log Stage 0 (`logs/A4/S00-03-LOG.md`):
   - Buat `logs/A4/S00-03-LOG.md` sesuai format `logs/LOG_TEMPLATE.md` yang mendokumentasikan pekerjaan S00-03 (penyusunan kebijakan logging AI, pembuatan struktur `logs/A1`..`logs/A4`, `logs/LOG_TEMPLATE.md`, dan `scripts/log_interaction.py`).
2. S05-01 — Audit Kepatuhan Kode & Standar PEP 8 (`logs/A4/S05-01-LOG.md`):
   - Audit seluruh berkas di `src/` dan `tests/` terhadap `docs/CODING_STANDARDS.md` dan kontrak `docs/ARCHITECTURE.md` (type hints, docstrings, penamaan snake_case/PascalCase, modularitas, reproducibility).
   - JANGAN mengubah kontrak fungsi publik di `src/` yang sudah lulus 58 unit test. Catat laporan hasil audit per modul secara lengkap di `logs/A4/S05-01-LOG.md`.
3. S05-02 — Validasi & Konsolidasi Log Interaksi AI (`docs/LOG_INTERAKSI.md` & `logs/A4/S05-02-LOG.md`):
   - Periksa seluruh file log di `logs/A1/`, `logs/A2/`, `logs/A3/`, dan `logs/A4/`.
   - Perbarui `docs/LOG_INTERAKSI.md` menjadi dokumen indeks & rekapitulasi resmi seluruh log interaksi AI tim (tabel daftar log per anggota A1..A4 beserta tautannya, ringkasan temuan debugging tiap stage, dan rekap keputusan D-001..D-008).
   - Catat proses audit log ini di `logs/A4/S05-02-LOG.md`.
4. S05-03 — Integration Testing & Handoff Stage 5 (`tests/test_integration.py`, `logs/A4/S05-03-LOG.md`, `docs/HANDOFF.md`):
   - Buat `tests/test_integration.py` yang menguji pipeline end-to-end lintas-modul di memori: `PIL.Image` -> `preprocess_image` -> `get_vgg19_model(weights=None)` & `get_features` -> `run_style_transfer` (Adam & L-BFGS) -> `deprocess_image` menghasilkan `PIL.Image` RGB ukuran tepat dan penurunan loss yang valid.
   - Jalankan `pytest tests/ -v -rs` dan pastikan 100% test lulus tanpa regresi.
   - Catat hasilnya di `logs/A4/S05-03-LOG.md`, perbarui status `S05-01`, `S05-02`, dan `S05-03` di `docs/planning/PROGRESS_TRACKER.md` dan `docs/planning/A4_PLAN.md` menjadi `IN_PROGRESS (menunggu review A1)`, serta tambahkan draf **Handoff #8 (Penutupan Stage 5 -> Stage 6)** pada `docs/HANDOFF.md`.
```
