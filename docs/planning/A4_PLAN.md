# 📕 Panduan & Rencana Kerja Anggota A4
### Muhammad Rafiq Abdurrasyid (NIM: 103012400240)
**Peran:** Quality Assurance, Dokumentasi & Error Handler

---

## 🎯 Tanggung Jawab Utama
1. Mengelola dan mengaudit riwayat percakapan AI (*Vibe Coding logs*) seluruh anggota pada folder `logs/`.
2. Melakukan *Code Review* terhadap kode yang dihasilkan oleh A1, A2, dan A3 agar patuh pada `docs/CODING_STANDARDS.md`.
3. Menjalankan *Integration Testing* dan menyelesaikan kendala error/bug (*Troubleshooting*).
4. Menyusun laporan akhir kelompok dan materi kesiapan presentasi.

---

## 📋 Rincian Langkah Kerja

### Stage 0: Dokumentasi & Logging Policy
* **S00-03: AI Logging Policy & Scaffolding Log**
  * *Tujuan:* Menyiapkan folder `logs/A1` s.d. `logs/A4`, template [LOG_TEMPLATE.md](../logs/LOG_TEMPLATE.md), dan skrip [scripts/log_interaction.py](../../scripts/log_interaction.py).
  * *Status:* `COMPLETED` ✅

### Stage 5: QA, Code Review & Troubleshooting (Setelah Stage 4)
* **S05-01: Code Review & Quality Audit**
  * *Tujuan:* Memeriksa kepatuhan kode seluruh anggota terhadap PEP 8, kelengkapan *type hints*, *docstrings*, serta memastikan tidak ada kode *hardcoded* yang melanggar kontrak arsitektur.
  * *Target Berkas:* Seluruh berkas di `src/`.
* **S05-02: Konsolidasi & Audit Log Interaksi AI**
  * *Tujuan:* Memastikan setiap anggota (A1, A2, A3) telah mencatat log interaksi AI dengan lengkap (tahap inisiasi, pesan error, prompt perbaikan, dan solusi).
  * *Target Berkas:* `logs/A1/`, `logs/A2/`, `logs/A3/`, `logs/A4/`, dan `logs/interactions.jsonl`.
  * *Acceptance Criteria:* Log lengkap sesuai rubrik penilaian (bobot 25% aman).
* **S05-03: Testing & Troubleshooting**
  * *Tujuan:* Menjalankan seluruh test suite menggunakan `pytest -v`. Menangkap error runtime/memory (jika ada), merumuskan prompt perbaikan ke AI, dan mencatat solusinya ke `docs/LOG_INTERAKSI.md`.

### Stage 7: Finalisasi Proyek & Acceptance
* **S07-01: Final Documentation & Report**
  * *Tujuan:* Memfinalisasi laporan proyek komprehensif, merapikan `README.md`, dan menyusun rekap kontribusi anggota.
* **S07-02: Final Acceptance & Demo Checklist**
  * *Tujuan:* Memastikan proyek siap didemonstrasikan di depan dosen/asisten (skenario live demo NST dan visualisasi grafik loss).

---

## 💬 Template Prompt untuk AI Agent A4

Saat Rafiq berinteraksi dengan AI di VS Code, gunakan template ini:

```text
Saya adalah Muhammad Rafiq Abdurrasyid (Anggota A4 - QA & Error Handler).
Proyek: Neural Style Transfer (Vibe Coding).
Referensi Dokumen: docs/planning/PROGRESS_TRACKER.md dan docs/CODING_STANDARDS.md.

Langkah Aktif: [Misal: S05-03 - Troubleshooting Error & Audit Test Suite]
Tujuan: [Menganalisis kegagalan unit test / error runtime]
Pesan Error Konsol:
[Salin pesan error terminal secara utuh di sini]

Berikan:
1. Analisis akar penyebab error (apakah pelanggaran shape tensor, device mismatch, dsb).
2. Solusi perbaikan minimal tanpa merusak kontrak modul milik anggota lain.
3. Rencana regression test untuk memastikan perbaikan tidak menimbulkan bug baru.
4. Format catatan log interaksi untuk dicatat di folder logs/A4/.
```
