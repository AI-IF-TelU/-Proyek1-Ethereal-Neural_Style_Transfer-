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

---

## Feature Extraction VGG-19 (`src/feature_extractor.py` — Stage 3: S03-01 & S03-02)

Analisis implementasi `src/feature_extractor.py` berdasarkan keputusan arsitektur `D-006`. Misalkan input berupa tensor 4D `(N, 3, H, W)` bertipe `float32`, dan model `vgg19.features[:29]` terdiri atas $L \le 29$ layer aktif (berhenti pada layer terakhir yang diminta, default indeks `'28'` / `conv5_1`).

### 1. Kompleksitas Waktu (*Time Complexity*)
* **`get_vgg19_model(device, weights)`:**
  * Pembuatan arsitektur, pemotongan 29 sub-modul pertama, penggantian `nn.ReLU(inplace=False)`, dan pembekuan parameter (`requires_grad=False`) memindai sejumlah parameter tetap VGG-19 hingga `conv5_1` ($P_{28} \approx 1.059 \times 10^7$ parameter konvolusi):
    $$\mathcal{O}(P_{28}) = \mathcal{O}(1) \quad \text{(konstan terhadap dimensi spasial gambar } H \times W\text{)}$$
  * Hanya dipanggil satu kali saat inisialisasi pipeline NST.
* **`get_features(image_tensor, model, layers)`:**
  * Pemeriksaan `torch.isfinite(image_tensor).all()` memindai seluruh elemen input: $\mathcal{O}(N \cdot C_0 \cdot H \cdot W)$ dengan $C_0 = 3$.
  * Untuk setiap layer konvolusi ke-$l \in \mathcal{L}_{\text{conv}}$ dengan ukuran kernel $k_l \times k_l$ ($k_l = 3$ pada seluruh layer VGG-19), channel masukan $C_{l-1}$, channel keluaran $C_l$, dan resolusi spasial $H_l \times W_l = (H / 2^{p_l}) \times (W / 2^{p_l})$ setelah $p_l \in \{0, 1, 2, 3, 4\}$ tahap `MaxPool2d`, biaya operasi *multiply-accumulate* adalah:
    $$T_{\text{conv}}(l) = \mathcal{O}\left(N \cdot C_{l-1} \cdot C_l \cdot k_l^2 \cdot H_l \cdot W_l\right)$$
  * Layer aktivasi `ReLU(inplace=False)` dan `MaxPool2d` berbiaya linear terhadap jumlah elemen aktivasi layer tersebut: $\mathcal{O}(N \cdot C_l \cdot H_l \cdot W_l)$.
  * Total kompleksitas waktu satu kali *forward pass* `get_features`:
    $$T_{\text{features}} = \mathcal{O}\left(N \sum_{l \in \mathcal{L}_{\text{conv}}} C_{l-1} C_l k_l^2 H_l W_l\right) = \mathcal{O}(N \cdot H \cdot W)$$
    karena arsitektur $(C_{l-1}, C_l, k_l, 2^{-p_l})$ bersifat tetap terhadap resolusi input $(H, W)$.
  * Pemotongan model `features[:29]` dan *early exit* menghemat 3 layer konvolusi $512 \times 512$ pada blok ke-5 (`conv5_2`, `conv5_3`, `conv5_4` di indeks `30, 32, 34`), memangkas sekitar $\approx 47\%$ parameter konvolusi VGG-19 penuh ($10.59\text{M}$ dari $20.02\text{M}$ parameter fitur).

### 2. Kompleksitas Ruang / Memori (*Space Complexity*)
* **Memori Parameter Model ($M_{\text{param}}$):**
  * Bobot dan bias `vgg19.features[:29]` bertipe `float32` (4 byte/parameter) membutuhkan memori tetap:
    $$M_{\text{param}} = \mathcal{O}(P_{28}) \approx 10.59 \times 10^6 \times 4\text{ byte} \approx 42.4\text{ MB}$$
  * Karena seluruh parameter dibekukan (`requires_grad=False`), PyTorch **tidak mengalokasikan buffer `.grad`** untuk bobot VGG-19 (menghemat $\approx 42.4\text{ MB}$ tambahan).
* **Memori Peta Fitur & Aktivasi ($M_{\text{act}}$):**
  * Dictionary keluaran menyimpan 6 tensor fitur (`conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv4_2`, `conv5_1`):
    $$M_{\text{dict}} = 4N \cdot H \cdot W \left(64 + \frac{128}{4} + \frac{256}{16} + \frac{512}{64} + \frac{512}{64} + \frac{512}{256}\right) = 4N \cdot 130 \cdot H \cdot W \text{ byte} = \mathcal{O}(N \cdot H \cdot W)$$
    Untuk $N=1, H=W=512$, ukuran ke-6 fitur keluaran adalah $4 \times 130 \times 512^2 \approx 136.3\text{ MB}$.
  * Saat `image_tensor.requires_grad == True` (selama optimasi gambar target), PyTorch menyimpan aktivasi antara pada seluruh layer `0..28` untuk keperluan *backward pass* dengan orde total $\mathcal{O}(N \sum_{l=0}^{28} C_l H_l W_l) = \mathcal{O}(N \cdot H \cdot W)$.

---

## Ringkasan Tahap Selanjutnya (Kerangka Stage 4 — A3)
- Gram Matrix: $\mathcal{O}(C_l^2 \cdot H_l W_l)$ per layer style
- Content Loss: $\mathcal{O}(C_l \cdot H_l W_l)$
- Style Loss: $\mathcal{O}(\sum_l (C_l^2 \cdot H_l W_l + C_l^2))$
- Optimization ($T$ iterasi): $\mathcal{O}(T \cdot (T_{\text{forward}} + T_{\text{backward}}))$
