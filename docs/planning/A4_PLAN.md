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
  * *Tujuan:* Menyiapkan folder `logs/A1` s.d. `logs/A4`, template [LOG_TEMPLATE.md](../../logs/LOG_TEMPLATE.md), skrip [scripts/log_interaction.py](../../scripts/log_interaction.py), serta pengujian [tests/test_log_interaction.py](../../tests/test_log_interaction.py).
  * *Status:* `COMPLETED` ✅ (Disetujui oleh A1 pada `commit 349c627` — 8/8 unit test skrip logging lulus, arsip log tersedia di [logs/A4/S00-03-LOG.md](../../logs/A4/S00-03-LOG.md)).

---

### Stage 5: QA, Code Review & Integration Testing (AKTIF — `S05-01` Selesai, Lanjut `S05-02` & `S05-03`!)

> **Status Stage 5 untuk A4 (Muhammad Rafiq Abdurrasyid):**
> * **`S00-03` & `S05-01` telah resmi DISETUJUI (`APPROVED`) oleh Ketua (A1 - Samuel Aaron).**
> * Keputusan atas temuan audit `S05-01-LOG.md`:
>   1. **Temuan #1 (`main.py` & `src/visualization.py` escaped docstring `\"\"\"`):** Akan diimplementasikan penuh oleh A1 pada Stage 6 (`S06-01`), atau A4 diperbolehkan mengganti `\"\"\"` menjadi `"""` pada `S05-03` agar *linter* bersih dari `E901 SyntaxError`.
>   2. **Temuan #2 (`optimizer_type` & `init_mode` di `src/optimization.py` melempar `AttributeError` alih-alih `TypeError` untuk input non-string):** Sesuai peran A4 sebagai *Error Handler* pada `S05-03`, A4 **diizinkan** menambahkan guard `if not isinstance(..., str): raise TypeError(...)` pada `src/optimization.py` beserta unit test-nya di `tests/test_integration.py` (selama kontrak fungsi publik `D-008` tidak berubah).
>   3. **Temuan #3 s.d. #10 (Kosmetik / Docstring / Info):** Tercatat dengan sangat baik sebagai dokumentasi QA pada laporan akhir.

#### S05-01: Audit Kepatuhan Kode & Standar PEP 8
* **Status:** `COMPLETED` ✅ (Disetujui oleh A1 — `pycodestyle` 228 E501, 2 E901, 10 temuan terdokumentasi di [logs/A4/S05-01-LOG.md](../../logs/A4/S05-01-LOG.md), `pytest`: 66 passed).
* **Berkas Selesai:** [logs/A4/S00-03-LOG.md](../../logs/A4/S00-03-LOG.md), [logs/A4/S05-01-LOG.md](../../logs/A4/S05-01-LOG.md), [scripts/log_interaction.py](../../scripts/log_interaction.py), [tests/test_log_interaction.py](../../tests/test_log_interaction.py).

#### S05-02: Validasi & Konsolidasi Log Interaksi AI (Bobot Rubrik 25%)
* **Status:** `READY` 🟢 (Langsung eksekusi sekarang!)
* **Tujuan:**
  1. Mengaudit kelengkapan seluruh berkas log di `logs/A1/`, `logs/A2/`, `logs/A3/`, dan `logs/A4/` terhadap format [logs/LOG_TEMPLATE.md](../../logs/LOG_TEMPLATE.md) (Metadata, `0. TAHAP PLANNING`, `1. TAHAP INISIASI / IMPLEMENTATION`, `2. TAHAP PENANGANAN EROR / DEBUGGING`, `3. TESTING, REVIEW & HANDOFF`).
  2. Memperbarui **`docs/LOG_INTERAKSI.md`** dengan indeks komprehensif seluruh log interaksi tim (tabel ringkasan per anggota A1..A4, tautan ke setiap file log di `logs/`, ringkasan kasus *debugging* utama yang berhasil diselesaikan di tiap stage, dan keputusan arsitektur `D-001` s.d. `D-008`).
  3. (Opsional / Direkomendasikan) Menambahkan entri ringkasan untuk langkah-langkah A1, A2, dan A3 yang sudah selesai ke dalam `logs/interactions.jsonl` menggunakan `scripts/log_interaction.py` sehingga `logs/interactions.jsonl` mencerminkan seluruh jejak tim A1–A4.
  4. Mencatat hasil validasi dan konsolidasi log pada **`logs/A4/S05-02-LOG.md`**.
* **Target Berkas:** `docs/LOG_INTERAKSI.md`, `logs/A4/S05-02-LOG.md`, `logs/interactions.jsonl`

#### S05-03: Integration Testing & Handoff Stage 5 ke Stage 6 (A1)
* **Status:** `READY` 🟢 (Kerjakan langsung bersama/setelah `S05-02`!)
* **Tujuan:**
  1. Membuat berkas pengujian integrasi **`tests/test_integration.py`** yang menguji rantai *end-to-end* lintas-modul di memori:
     `PIL.Image` $\rightarrow$ `preprocess_image` (A2) $\rightarrow$ `get_vgg19_model(weights=None)` & `get_features` (A1) $\rightarrow$ `gram_matrix`, `compute_content_loss`, `compute_style_loss`, `compute_total_loss` (A3) $\rightarrow$ `run_style_transfer` (`adam` & `lbfgs`, A3) $\rightarrow$ `deprocess_image` (A2) menghasilkan `PIL.Image` RGB yang valid dan penurunan nilai loss yang konvergen.
  2. (Opsional sesuai otorisasi A1 atas Temuan #1 & #2 di `S05-01-LOG.md`): Memperbaiki guard `isinstance(..., str)` untuk `optimizer_type` dan `init_mode` di `src/optimization.py` agar melempar `TypeError` sesuai docstring, serta memperbaiki escaped docstring `\"\"\"` di `main.py` dan `src/visualization.py` bila diinginkan.
  3. Menjalankan seluruh suite pengujian (`pytest tests/ -v -rs`) dan memastikan 100% test lulus tanpa regresi.
  4. Mencatat log eksekusi pengujian integrasi pada **`logs/A4/S05-03-LOG.md`**, memperbarui status `S05-02` dan `S05-03` pada `docs/planning/PROGRESS_TRACKER.md` dan `docs/planning/A4_PLAN.md`, serta menambahkan draf **Handoff #9 (Penutupan Stage 5 $\rightarrow$ Stage 6)** pada `docs/HANDOFF.md`.
* **Target Berkas:** `tests/test_integration.py`, `logs/A4/S05-03-LOG.md`, `docs/HANDOFF.md`, `docs/planning/PROGRESS_TRACKER.md`, `docs/planning/A4_PLAN.md`

---

### Stage 7: Finalisasi Proyek & Acceptance (Setelah Stage 6 Selesai)
* **S07-01: Final Documentation & Report**
  * *Tujuan:* Memfinalisasi laporan proyek komprehensif (`docs/FINAL_REPORT.md`), merapikan `README.md`, dan menyusun rekap kontribusi anggota.
* **S07-02: Final Acceptance & Demo Checklist**
  * *Tujuan:* Menyusun `docs/DEMO_CHECKLIST.md` dan memastikan proyek siap didemonstrasikan di depan dosen/asisten.

---

## 💬 Template Prompt Siap Pakai untuk AI Agent A4 (`S05-02` & `S05-03` Sekaligus)

Saat **Rafiq (A4)** berinteraksi dengan AI di VS Code, cukup *copy-paste* prompt berikut untuk menyelesaikan sisa Stage 5 (`S05-02` dan `S05-03`) sekaligus:

```text
Saya adalah Muhammad Rafiq Abdurrasyid (Anggota A4 - QA, Dokumentasi & Error Handler, NIM: 103012400240).
Samuel (A1 - Ketua) telah menyetujui (ACC) pekerjaan S00-03 dan S05-01 kita (lihat Handoff #8 di `docs/HANDOFF.md`, `logs/A1/S05-01-REVIEW-LOG.md`, dan `docs/planning/A4_PLAN.md`), serta mengizinkan kita langsung menyelesaikan S05-02 dan S05-03 sekaligus untuk menutup Stage 5.

Keputusan A1 atas pertanyaan/temuan S05-01 kita:
1. Temuan #1 (`main.py` & `src/visualization.py` escaped docstring `\"\"\"`): Boleh sekalian diperbaiki menjadi `"""` di S05-03 agar bebas E901 SyntaxError (implementasi penuhnya akan dikerjakan A1 di Stage 6).
2. Temuan #2 (`optimizer_type` & `init_mode` non-string di `src/optimization.py`): Sebagai Error Handler di S05-03, kita diizinkan menambahkan pengecekan `isinstance(..., str)` yang melempar `TypeError` sesuai kontrak docstring, beserta unit test-nya.
3. Konsolidasi `logs/interactions.jsonl`: Boleh menambahkan entri ringkasan untuk log A1, A2, dan A3 yang sudah selesai agar `logs/interactions.jsonl` lengkap untuk seluruh anggota.

Sekarang kerjakan S05-02 dan S05-03 secara berurutan sampai tuntas:
1. S05-02 — Validasi & Konsolidasi Log Interaksi AI (`docs/LOG_INTERAKSI.md`, `logs/A4/S05-02-LOG.md`, `logs/interactions.jsonl`):
   - Audit seluruh berkas log di `logs/A1/`, `logs/A2/`, `logs/A3/`, dan `logs/A4/` terhadap `logs/LOG_TEMPLATE.md`.
   - Perbarui `docs/LOG_INTERAKSI.md` menjadi indeks & rekapitulasi resmi seluruh log interaksi AI tim (tabel daftar log per anggota A1..A4 beserta tautan relatifnya, ringkasan temuan debugging tiap stage, dan rekap keputusan arsitektur D-001..D-008).
   - Buat `logs/A4/S05-02-LOG.md` sesuai `logs/LOG_TEMPLATE.md` dan catat di `logs/interactions.jsonl`.
2. S05-03 — Integration Testing, Error Troubleshooting & Handoff Stage 5 (`tests/test_integration.py`, `logs/A4/S05-03-LOG.md`, `docs/HANDOFF.md`):
   - Buat `tests/test_integration.py` yang menguji pipeline end-to-end lintas-modul di memori: `PIL.Image` -> `preprocess_image` (A2) -> `get_vgg19_model(weights=None)` & `get_features` (A1) -> `gram_matrix`, `compute_content_loss`, `compute_style_loss`, `compute_total_loss` (A3) -> `run_style_transfer` (Adam & L-BFGS, A3) -> `deprocess_image` (A2) menghasilkan `PIL.Image` RGB ukuran tepat dan penurunan loss yang valid.
   - Terapkan perbaikan kecil Temuan #1 & #2 sesuai izin A1 di atas, lalu pastikan `pytest` lulus 100% tanpa regresi dan file milik A4 patuh PEP 8 (`pycodestyle`).
   - Buat `logs/A4/S05-03-LOG.md`, catat di `logs/interactions.jsonl`, perbarui `docs/planning/PROGRESS_TRACKER.md` dan `docs/planning/A4_PLAN.md`, serta tambahkan draf Handoff #9 (Penutupan Stage 5 -> Stage 6) pada `docs/HANDOFF.md`.
```
