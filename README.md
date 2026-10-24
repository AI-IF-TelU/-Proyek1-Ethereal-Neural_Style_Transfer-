# Neural Style Transfer (NST) - Generative AI From Scratch

Proyek kelompok 4 mahasiswa. Mengembangkan NST berbasis VGG-19 pretrained (content/style loss, Gram Matrix). Fokus modular, reproducible, terdokumentasi.

## Struktur
Lihat struktur folder di atas.

## Persyaratan
- Python 3.11/3.12 disarankan (saat ini terdeteksi 3.14 - perlu verifikasi kompatibilitas PyTorch)
- PyTorch, torchvision, pillow, matplotlib, numpy, pytest, pyyaml

## Setup
1. Buat venv: python -m venv .venv
2. Aktifkan (.venv\\Scripts\\activate di Windows)
3. pip install -r requirements.txt

## Menjalankan tes awal
pytest -q

## Pembagian tugas
Lihat docs/TASKS.md

## Aturan AI Agent
Lihat docs/AI_RULES.md
