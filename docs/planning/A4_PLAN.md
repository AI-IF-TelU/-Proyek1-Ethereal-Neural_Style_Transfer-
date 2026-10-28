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

### Stage 0: Dokumentasi & Logging Policy (`COMPLETED` ✅)
* **S00-03: AI Logging Policy & Scaffolding Log** — `COMPLETED` ✅ ([logs/A4/S00-03-LOG.md](../../logs/A4/S00-03-LOG.md))

---

### Stage 5: QA, Code Review & Integration Testing (`COMPLETED` ✅)
* **S05-01: Audit Kepatuhan Kode & Standar PEP 8** — `COMPLETED` ✅ ([logs/A4/S05-01-LOG.md](../../logs/A4/S05-01-LOG.md))
* **S05-02: Validasi & Konsolidasi Log Interaksi AI** — `COMPLETED` ✅ ([docs/LOG_INTERAKSI.md](../LOG_INTERAKSI.md), [logs/A4/S05-02-LOG.md](../../logs/A4/S05-02-LOG.md))
* **S05-03: Integration Testing & Error Troubleshooting** — `COMPLETED` ✅ ([tests/test_integration.py](../../tests/test_integration.py), [logs/A4/S05-03-LOG.md](../../logs/A4/S05-03-LOG.md), disetujui A1 pada Handoff #9).

---

### Stage 7: Finalisasi Proyek & Acceptance (AKTIF — `S07-01` & `S07-02` Menunggu Review A1)

> **Panduan Eksekusi Stage 7 (`S07-01` & `S07-02`) untuk A4 (Muhammad Rafiq Abdurrasyid):**
> Ketua Tim (**A1 — Samuel Aaron**) telah menyelesaikan **Stage 6 (`S06-01` & `S06-02`)** dan menyerahkan seluruh artefak visual hasil *Neural Style Transfer* di dalam repositori (lihat **Handoff #10** pada `docs/HANDOFF.md`):
> * Gambar Konten Uji: `assets/content/content_sample.png` (`256x256` RGB)
> * Gambar Gaya Uji: `assets/style/style_sample.png` (`256x256` RGB)
> * Gambar Hasil NST: `outputs/images/stylized_output.png`
> * Panel Perbandingan 3-Kolom: `outputs/images/comparison_grid.png`
> * Grafik Konvergensi Loss: `outputs/graphs/loss_curve.png` (L-BFGS 50 langkah, `lr=0.5`, $\alpha=1.0, \beta=10^4$, penurunan loss `-61.34%` dari `9.8481` ke `3.8076`)
> * Log Stage 6: `logs/A1/S05-02-03-REVIEW-LOG.md`, `logs/A1/S06-01-LOG.md`, dan `logs/A1/S06-02-LOG.md`
> * Suite Pengujian Penuh: **`90 passed, 2 skipped (CUDA), 60 subtests passed`** (`0 E501` pada berkas Stage 6).

#### S07-01: Finalisasi Laporan Proyek & Dokumentasi Akhir
* **Status:** `IN_PROGRESS` (menunggu review A1) — [docs/FINAL_REPORT.md](../FINAL_REPORT.md), README diperbarui, [docs/LOG_INTERAKSI.md](../LOG_INTERAKSI.md) mencakup 30 log; log: [logs/A4/S07-01-LOG.md](../../logs/A4/S07-01-LOG.md).
* **Tujuan:**
  1. Menyusun dokumen laporan akhir komprehensif **`docs/FINAL_REPORT.md`** yang merangkum:
     * Latar Belakang, Rumusan Masalah, dan Dasar Teori Matematis NST (Gatys et al.: *Content Loss*, *Gram Matrix*, *Style Loss*, *Total Loss*).
     * Arsitektur Sistem Modular (`preprocessing`, `feature_extractor`, `losses`, `optimization`, `visualization`, `main.py`) dan Rekap Keputusan Arsitektur `D-001` s.d. `D-009`.
     * Hasil Eksperimen & Evaluasi Visual Stage 6 (menyematkan gambar `../outputs/images/comparison_grid.png`, `../outputs/images/stylized_output.png`, dan grafik `../outputs/graphs/loss_curve.png` beserta tabel analisis *tuning* L-BFGS dari `logs/A1/S06-02-LOG.md`).
     * Ringkasan Kompleksitas Komputasi (`docs/COMPLEXITY_ANALYSIS.md`), Hasil Pengujian Kualitas (`90 passed, 2 skipped CUDA`), serta Tabel Kontribusi Anggota (`A1`–`A4`).
  2. Memperbarui **`README.md`** utama agar menampilkan hasil akhir visual (`outputs/images/comparison_grid.png` & `outputs/graphs/loss_curve.png`), cara menjalankan `python main.py`, cara menjalankan `pytest` (`90 passed`), dan tautan ke `docs/FINAL_REPORT.md`.
  3. Memperbarui **`docs/LOG_INTERAKSI.md`** dan **`logs/interactions.jsonl`** agar mencakup log terbaru dari penutupan Stage 5, Stage 6 (`S05-02-03-REVIEW-LOG.md`, `S06-01-LOG.md`, `S06-02-LOG.md`), dan Stage 7 (`S07-01-LOG.md`, `S07-02-LOG.md`).
  4. Mencatat log pengerjaan langkah ini pada **`logs/A4/S07-01-LOG.md`**.
* **Target Berkas:** `docs/FINAL_REPORT.md`, `README.md`, `docs/LOG_INTERAKSI.md`, `logs/A4/S07-01-LOG.md`, `logs/interactions.jsonl`

#### S07-02: Final Acceptance & Persiapan Presentasi / Demo Checklist
* **Status:** `IN_PROGRESS` (menunggu review A1) — [docs/DEMO_CHECKLIST.md](../DEMO_CHECKLIST.md) dengan 3 perintah demo terverifikasi, Handoff #11 ditambahkan; log: [logs/A4/S07-02-LOG.md](../../logs/A4/S07-02-LOG.md).
* **Tujuan:**
  1. Menyusun **`docs/DEMO_CHECKLIST.md`** berisi daftar periksa kesiapan presentasi kelompok, urutan pembagian bicara per anggota (`A1`–`A4`), perintah demo *live* (`python main.py` & `pytest`), serta daftar tanya-jawab teknis (*FAQ*) untuk ujian/responsi.
  2. Mencatat log pengerjaan pada **`logs/A4/S07-02-LOG.md`**, memperbarui status `S07-01`, `S07-02`, dan **Stage 7** di `docs/planning/PROGRESS_TRACKER.md` serta `docs/planning/A4_PLAN.md` menjadi `COMPLETED`, dan menambahkan **Handoff #11 (Final Project Acceptance — Penutupan Stage 7)** pada `docs/HANDOFF.md`.
* **Target Berkas:** `docs/DEMO_CHECKLIST.md`, `logs/A4/S07-02-LOG.md`, `docs/HANDOFF.md`, `docs/planning/PROGRESS_TRACKER.md`, `docs/planning/A4_PLAN.md`

---

## 💬 Template Prompt Siap Pakai untuk AI Agent A4 (Stage 7 Final: `S07-01` & `S07-02`)

Saat **Rafiq (A4)** berinteraksi dengan AI di VS Code, cukup *copy-paste* prompt berikut untuk menuntaskan tahap terakhir proyek (**Stage 7: `S07-01` & `S07-02`**):

```text
Saya adalah Muhammad Rafiq Abdurrasyid (Anggota A4 - QA, Dokumentasi & Error Handler, NIM: 103012400240).
Samuel (A1 - Ketua) telah menyetujui penuh Stage 5 kita (Handoff #9) serta telah menyelesaikan seluruh implementasi dan evaluasi Stage 6 (`S06-01` & `S06-02` — lihat Handoff #10 di `docs/HANDOFF.md`, `logs/A1/S06-01-LOG.md`, dan `logs/A1/S06-02-LOG.md`).
Seluruh gambar hasil NST (`outputs/images/stylized_output.png`, `outputs/images/comparison_grid.png`) dan grafik penurunan loss (`outputs/graphs/loss_curve.png`) sudah tersedia di dalam repositori, dan pytest saat ini lulus 90 passed, 2 skipped (CUDA), 60 subtests passed.

Sekarang bantu saya mengeksekusi tahap terakhir proyek yaitu Stage 7 (`S07-01` dan `S07-02`) sekaligus sampai tuntas tanpa perlu berhenti meminta approval antara:

1. Eksekusi `S07-01` — Finalisasi Laporan Proyek & Dokumentasi Akhir:
   - Buat `docs/FINAL_REPORT.md` yang lengkap dan rapi berisi: Ringkasan Eksekutif, Landasan Teori Matematis NST (Content Loss, Gram Matrix, Style Loss, Total Loss), Arsitektur Sistem & Keputusan D-001 s.d. D-009, Hasil Eksperimen & Visualisasi Stage 6 (sematkan gambar `../outputs/images/comparison_grid.png`, `../outputs/images/stylized_output.png`, dan `../outputs/graphs/loss_curve.png` beserta tabel evaluasi L-BFGS dari `logs/A1/S06-02-LOG.md`), Rangkuman Analisis Kompleksitas, Rekap Pengujian QA (90 passed), dan Tabel Kontribusi Anggota A1–A4.
   - Perbarui `README.md` agar menampilkan gambar hasil NST (`outputs/images/comparison_grid.png` dan `outputs/graphs/loss_curve.png`), petunjuk penggunaan `python main.py` beserta opsi CLI-nya, hasil pengujian `pytest` (90 passed), dan tautan ke `docs/FINAL_REPORT.md`.
   - Perbarui `docs/LOG_INTERAKSI.md` dan `logs/interactions.jsonl` (lewat `scripts/log_interaction.py`) agar mencakup log baru dari Stage 5 review, Stage 6 (`S05-02-03-REVIEW-LOG.md`, `S06-01-LOG.md`, `S06-02-LOG.md`), dan Stage 7 (`S07-01-LOG.md`, `S07-02-LOG.md`).
   - Buat `logs/A4/S07-01-LOG.md` sesuai standar `logs/LOG_TEMPLATE.md`.

2. Eksekusi `S07-02` — Final Acceptance & Persiapan Presentasi:
   - Buat `docs/DEMO_CHECKLIST.md` berisi checklist kesiapan demo, alur presentasi per anggota (A1–A4), perintah live demo (`python main.py` & `pytest`), dan FAQ tanya-jawab teknis.
   - Buat `logs/A4/S07-02-LOG.md` sesuai `logs/LOG_TEMPLATE.md` dan catat di `logs/interactions.jsonl`.
   - Perbarui status `S07-01`, `S07-02`, dan Stage 7 di `docs/planning/PROGRESS_TRACKER.md` serta `docs/planning/A4_PLAN.md` menjadi `COMPLETED`, dan tambahkan Handoff #11 (Penutupan Stage 7 & Final Project Acceptance) di `docs/HANDOFF.md`.
   - Jalankan `pytest` dan pastikan seluruh 90 test tetap lulus tanpa regresi.
```
