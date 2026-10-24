# TEMPLATE LOG INTERAKSI LLM - NST
# Sesuai contoh yang diberikan

[Nama file: logs/ANGGOTA/STEP-ID-LOG.md]

## Metadata
- Nama Kelompok: [isi]
- Metodologi: SDLC + Stage-Gate Sequential + Vibe Engineering
- Tools: VS Code, Kilo Code, Python, PyTorch
- Tanggal Sesi: [DD/MM/YYYY]
- Anggota: [A1/A2/A3/A4] - [Peran]
- Target Modul: [nama modul]
- AI Model: [nama model]
- Tautan Sesi Chat: [tautan/ekspor]
- Step ID: [contoh S00-02]
- Phase: [planning/implementation/debugging/testing/handoff]

## 0. TAHAP PLANNING
### Prompt Kelompok
`
[teks lengkap]
`
### Alasan Khusus
[alasan]
### Output Model
[ringkasan respons asli]
### Status Planning: [Menunggu Persetujuan / Disetujui / Revisi]

## 1. TAHAP INISIASI / IMPLEMENTATION
### Prompt Kelompok
`
[teks lengkap]
`
### Alasan Khusus
[alasan]
### Output Model
[file berubah, fungsi dibuat, ringkasan]
### Status: [Menunggu Persetujuan / Disetujui / Revisi]

## 2. TAHAP PENANGANAN EROR / DEBUGGING
### Pesan Eror
`
[error aktual atau "Tidak ada error"]
`
### Prompt Perbaikan ke LLM
`
[teks lengkap]
`
### Analisis & Solusi Kelompok
[analisis aktual]
### Pelajaran
[pelajaran]

## 3. TESTING, REVIEW & HANDOFF
- Validasi: [deskripsi]
- Unit/Integration Test: [hasil]
- Hasil Review: [review]
- File Hasil: [path]
- Git Commit: [hash atau "-" ]
- Handoff: [status]
- Status Persetujuan Manusia: [Menunggu/Disetujui/Ditolak]

## Catatan
- Simpan respons AI penuh di ekspor sesi/JSONL
- Jangan mengarang error/testing
- Prompt disimpan utuh
- Timestamp & identitas divalidasi
