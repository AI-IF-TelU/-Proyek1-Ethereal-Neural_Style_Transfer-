# COMPLEXITY_ANALYSIS (Kerangka)

Asumsi: HxW gambar, C channel, iterasi T, fitur F dari layer VGG.

- Preprocessing: O(HW*C)
- Feature extraction: tergantung arsitektur VGG-19 (konvolusi) - catat asumsi
- Gram Matrix: O(C^2 * HW) per layer
- Content Loss: O(HW*C)
- Style Loss: O(sum C_l^2 HW_l)
- Optimization (T iter): T * (forward+backward)
- Memori: dominan fitur & tensor (perkirakan)

Catatan: belum ada klaim final - analisis saat implementasi.
