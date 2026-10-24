# PRD - Neural Style Transfer (NST)

## Latar Belakang
Proyek ini bertujuan mengembangkan Neural Style Transfer berbasis VGG-19 pretrained dari scratch secara modular.

## Tujuan
Mengimplementasikan NST dengan Content Loss, Style Loss, Gram Matrix, optimasi terhadap piksel.

## Ruang Lingkup
- Preprocessing gambar
- Feature extraction VGG-19 (frozen)
- Gram Matrix, Content/Style/Total Loss
- Optimasi pixel
- Visualisasi hasil
- Dokumentasi & pengujian

## Fitur Utama
- Transfer gaya antar gambar
- Modular (preprocessing, feature_extractor, losses, optimization)
- Reproducible
- Unit testing

## Input/Output
- Input: content image, style image
- Output: generated image, visualisasi loss

## Batasan
- Hanya VGG-19 pretrained
- Optimasi terhadap piksel, bukan fine-tune VGG
- Implementasi mandiri (gram/loss)
- Tidak menyalin tutorial penuh

## Kriteria Keberhasilan
- Modul terpisah sesuai kontrak
- Gram Matrix & loss terimplementasi benar
- Loss menurun secara wajar
- Hasil tertransfer secara visual
- Pengujian dasar lulus
