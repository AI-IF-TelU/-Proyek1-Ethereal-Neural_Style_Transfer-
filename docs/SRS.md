# SRS - Neural Style Transfer

## Functional Requirements
- FR-01: Load & preprocess gambar
- FR-02: Ekstraksi fitur VGG-19 (layer tertentu)
- FR-03: Hitung Content Loss
- FR-04: Hitung Gram Matrix & Style Loss
- FR-05: Total Loss & optimasi (LBFGS/Adam)
- FR-06: Simpan hasil

## Non-Functional Requirements
- NFR-01: Modular, readable
- NFR-02: Reproducible (seed)
- NFR-03: Type hints & docstring
- NFR-04: Unit test
- NFR-05: Dokumentasi lengkap

## Hardware/Software
- Python 3.11/3.12 (rekomendasi), saat ini 3.14 perlu verifikasi
- PyTorch, torchvision, pillow, numpy, matplotlib, pytest, pyyaml

## Validasi Input
- Format gambar valid
- Ukuran sesuai
- Path aman

## Error Handling
- File tidak ditemukan
- Format tidak didukung
- Tensor shape mismatch
