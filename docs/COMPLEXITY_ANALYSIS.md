# COMPLEXITY_ANALYSIS (Kerangka)

Asumsi: HxW gambar, C channel, iterasi T, fitur F dari layer VGG.

## Preprocessing (S02-04, Opsi B CPU)

Analisis implementasi `src/preprocessing.py`; bukan benchmark seluruh pipeline. Ukuran sumber `Hs x Ws`, target `Ht x Wt`, dan kanal RGB `C=3` (batch `N=1`). Untuk `size=None`, target sama dengan sumber. `D` menyatakan biaya baca/decode file, bergantung codec, ukuran berkas terkompresi, dan sumber; tidak dianggap konstan atau hanya bergantung ukuran target. Konversi sumber ke RGB berbiaya `O(C Hs Ws)` dengan asumsi jumlah kanal masukan terbatas.

### Resize Lanczos
Gunakan `Kx` dan `Ky` sebagai jumlah tap sumber efektif per piksel target di sumbu horizontal/vertikal. Dengan radius dukungan Lanczos `a=3`, batas tap efektif berorde `Kx = O(min(Ws, a max(1, Ws/Wt)))` dan `Ky = O(min(Hs, a max(1, Hs/Ht)))`, dengan pembulatan dan batas gambar. Radius kernel dasar tetap tidak berarti jumlah tap selalu konstan: downsampling memperlebar dukungan pada koordinat sumber.

Untuk model resampling separabel horizontal lalu vertikal dengan buffer antara `Hs x Wt`, batas waktu konservatif `R = O(C (Hs Wt Kx + Ht Wt Ky) + Wt Kx + Ht Ky)`. Dua suku terakhir mencakup pembentukan koefisien. Buffer resize `M_R = O(C Hs Wt + Wt Kx + Ht Ky)`, di luar gambar sumber/hasil. Pillow dapat melewati sumbu yang tidak berubah atau memproses lebih sedikit baris; ini batas model, bukan klaim alokasi persis. Jika ukuran sudah sama atau `size=None`, implementasi melewati resize (`R=0`, `M_R=0`). Penyederhanaan ke biaya linear jumlah piksel target saja tidak berlaku umum ketika ukuran sumber/rasio downsampling berubah.

### Waktu per Fungsi
| Fungsi | Batas waktu |
| :--- | :--- |
| `load_image` | `O(D + C Hs Ws + R)` untuk decode, konversi RGB, dan resize opsional |
| `preprocess_image` dari path | `O(D + C Hs Ws + R + C Ht Wt)`; hanya satu jalur resize, lalu salinan NumPy float32, skala, dan normalisasi |
| `preprocess_image` dari PIL | `O(C Hs Ws + R + C Ht Wt)` untuk input sudah decoded; tambahkan biaya decode bila PIL masih lazy |
| `deprocess_image` | `O(C Ht Wt)` untuk clone, pemeriksaan finite, inverse normalisasi, clamp, pembulatan, uint8, dan hasil PIL |

`from_numpy`, `permute`, `unsqueeze`, `squeeze`, dan `detach` umumnya berbagi storage/view metadata, bukan salinan seluruh piksel. Operasi aritmetika dan clone tetap memindai/mengalokasikan data. Validasi shape/size berbiaya konstan untuk rank tetap; pemeriksaan finite tidak konstan.

### Memori dan Transfer
Asumsi CPU, float32 (4 byte/elemen), uint8 (1 byte/elemen), jumlah temporer aritmetika tetap, dan tidak ada retensi hasil berulang oleh pemanggil. Preprocess memerlukan memori live berorde `O(C Hs Ws + C Ht Wt + M_R)` ditambah buffer codec/I/O; target float32 sendiri `4 C Ht Wt` byte, bukan seluruh peak RSS. Decode tetap dapat menahan gambar sumber besar meskipun target kecil. Array NumPy float32 dan hasil normalisasi/temporer hidup pada beberapa titik; tidak diklaim zero-copy end-to-end.

Deprocess memiliki tambahan memori `O(C Ht Wt)` untuk clone, temporer floating/boolean, uint8, dan gambar PIL; input pemanggil tetap hidup dan tidak dimutasi. `detach` tidak menyalin tetapi `clone` menyalin. Pada jalur CPU, `.to("cpu")`/`.cpu()` tidak memerlukan transfer antar-device; ini tidak menghapus salinan lain. Untuk input floating selain float32, byte floating mengikuti dtype aktual.

Di luar scope verifikasi opsi B, device accelerator dapat menambah transfer `O(C Ht Wt)` elemen serta storage host/device dan sinkronisasi; biaya waktu juga bergantung bandwidth. Tidak ada klaim performa, peak memory terukur, atau kelulusan CUDA. Bukti CPU dan batas acceptance A1 dicatat pada `logs/A2/S02-04-LOG.md`.

- Feature extraction: tergantung arsitektur VGG-19 (konvolusi) - catat asumsi
- Gram Matrix: O(C^2 * HW) per layer
- Content Loss: O(HW*C)
- Style Loss: O(sum C_l^2 HW_l)
- Optimization (T iter): T * (forward+backward)
- Memori: dominan fitur & tensor (perkirakan)

Catatan: belum ada klaim final - analisis saat implementasi.
