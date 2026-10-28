# ARCHITECTURE — Neural Style Transfer (NST)

Dokumen ini mendefinisikan arsitektur sistem, aliran data tensor, dan kontrak antarmuka (*API contracts*) antar-modul untuk proyek Neural Style Transfer berbasis VGG-19 beku.

---

## 1. Alur Kerja Sistem (Mermaid Flowchart)

```mermaid
graph TD
    A[Input Content & Style Image] --> B[Module: preprocessing.py]
    B -->|Tensor 1, 3, H, W Normalisasi| C[Module: feature_extractor.py - VGG-19 Beku]
    C -->|conv4_2 Feature Maps| D[Content Features]
    C -->|conv1_1 s.d. conv5_1 Maps| E[Style Features]
    D --> F[Content Loss - MSE]
    E --> G[Gram Matrix - Korelasi Channel]
    G --> H[Style Loss - MSE Gram]
    F --> I[Total Loss = α · Content + β · Style]
    H --> I
    I -->|Autograd Gradient| J[Module: optimization.py - L-BFGS/Adam]
    J -->|Update Piksel Target| K[Piksel Generated Image]
    K -->|Deprocess Image| L[Module: visualization.py - Simpan & Plot]
```

---

## 2. Kontrak Modul & Spesifikasi Fungsi

### A. Modul Preprocessing (`src/preprocessing.py`) — PIC: A2

> **Kebijakan Tunggal Parameter `size` (Keputusan D-005):**
> 1. Seluruh fungsi pada `src/preprocessing.py` menggunakan nama parameter **`size`** (bukan `target_size`).
> 2. Tuple `size` **selalu didefinisikan sebagai `(height, width)` = `(H, W)`** mengikuti konvensi PyTorch `[N, C, H, W]`.
> 3. Saat memanggil `PIL.Image.resize` secara internal, tuple `(height, width)` wajib dibalik menjadi `(width, height)`: `image.resize((width, height), Image.Resampling.LANCZOS)`.
> 4. Metode resize menggunakan *direct exact resize* ke `(H, W)` dengan filter `Image.Resampling.LANCZOS` (tanpa *cropping*).
> 5. Validasi `size`: Harus `None` atau `tuple[int, int]` berisi tepat 2 integer murni (`not isinstance(v, bool)`) dengan `H > 0` dan `W > 0`. Selain itu wajib melempar `ValueError`.

* **`load_image(image_path: str | Path, size: tuple[int, int] | None = None) -> Image.Image`**
  * *Deskripsi:* Membaca file JPEG/PNG, memvalidasi format isi file, dan mengonversi ke mode `"RGB"` mandiri (file sumber ditutup).
  * *Perilaku Resize:*
    * Jika `size=None`: mengembalikan `PIL.Image` RGB dengan ukuran asli.
    * Jika `size=(H, W)`: memvalidasi `size` dan melakukan resize `LANCZOS` sehingga `result.height == H` dan `result.width == W` (yaitu `result.size == (W, H)`).
  * *Exceptions:* `FileNotFoundError`, `ValueError` (format bukan JPEG/PNG, file tidak dikenali, atau `size` tidak valid), `OSError` (akses file atau decoding gagal pada format yang dikenali, termasuk gambar truncated).
  * *Klarifikasi S02-04:* Pembedaan file tidak dikenali versus kegagalan decode di atas mengikuti implementasi yang sudah ada; bukan perubahan runtime/kontrak atau persetujuan manusia baru.
* **`preprocess_image(image: Image.Image | str | Path, size: tuple[int, int] | None = (512, 512), device: torch.device | str = "cpu") -> torch.Tensor`**
  * *Deskripsi:* Mengubah gambar menjadi Tensor PyTorch 4D `(1, 3, H, W)` bertipe `torch.float32` yang siap dimasukkan ke VGG-19.
  * *Perilaku Alur & Resize (Tanpa Double Resize):*
    * Jika `image` berupa `str | Path`: memanggil `load_image(image, size=size)`.
    * Jika `image` berupa `PIL.Image.Image`: memastikan mode `"RGB"`. Jika `size is not None`, memvalidasi `size` dan hanya melakukan resize `LANCZOS` ke `(H, W)` apabila `(image.height, image.width) != size`. Jika `size is None`, mempertahankan ukuran gambar saat itu.
  * *Normalisasi:* Skala awal piksel `[0.0, 1.0]` dinormalisasi dengan standar ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  * *Output Shape:* `(1, 3, H, W)` bertipe `torch.float32` pada `device` yang ditentukan.
  * *Exceptions:* `TypeError` / `ValueError` jika tipe `image` atau nilai `size` tidak valid.
* **`deprocess_image(tensor: torch.Tensor) -> Image.Image`**
  * *Deskripsi:* Mengembalikan tensor hasil olahan menjadi `PIL.Image.Image` mode `"RGB"`.
  * *Operasi:* Menerima tensor 4D `(1, 3, H, W)` atau 3D `(3, H, W)` (selain itu lempar `ValueError`), menyalin secara aman (`detach().cpu().clone()`), menghapus dimensi batch menjadi `(3, H, W)`, denormalisasi ($x \times \sigma + \mu$), *clamping* ke rentang $[0.0, 1.0]$, konversi ke `uint8` `[0, 255]`, dan mengembalikan `PIL.Image` RGB dengan `height == H` dan `width == W`.
  * *Klarifikasi Validasi S02-03:* Input bukan tensor ditolak dengan `TypeError`; dtype non-floating, nilai NaN/Inf, dan dimensi spasial kosong ditolak dengan `ValueError`. Konversi ke `uint8` membulatkan nilai piksel terdekat setelah clamp untuk menghindari bias truncation pada round-trip. Gambar PIL berdimensi kosong pada `preprocess_image` juga ditolak dengan `ValueError`.

### B. Modul Feature Extractor (`src/feature_extractor.py`) — PIC: A1

> **Kebijakan Ekstraksi Fitur VGG-19 Beku (Keputusan D-006):**
> 1. Pemetaan layer standar Gatys et al. (`DEFAULT_LAYERS`):
>    * `'0': 'conv1_1'` (*Style*, $64$ ch, $H \times W$)
>    * `'5': 'conv2_1'` (*Style*, $128$ ch, $\lfloor H/2 \rfloor \times \lfloor W/2 \rfloor$)
>    * `'10': 'conv3_1'` (*Style*, $256$ ch, $\lfloor H/4 \rfloor \times \lfloor W/4 \rfloor$)
>    * `'19': 'conv4_1'` (*Style*, $512$ ch, $\lfloor H/8 \rfloor \times \lfloor W/8 \rfloor$)
>    * `'21': 'conv4_2'` (*Content*, $512$ ch, $\lfloor H/8 \rfloor \times \lfloor W/8 \rfloor$)
>    * `'28': 'conv5_1'` (*Style*, $512$ ch, $\lfloor H/16 \rfloor \times \lfloor W/16 \rfloor$)
> 2. Sekuens `vgg19.features` dipotong hingga indeks `'28'` (`features[:29]`) dan seluruh `nn.ReLU(inplace=True)` diganti menjadi `nn.ReLU(inplace=False)` agar tensor fitur `Conv2d` tidak termodifikasi *in-place* oleh layer ReLU berikutnya serta menjaga keutuhan grafik *autograd*.
> 3. Seluruh parameter model dibekukan (`requires_grad = False`) dan model diset ke `.eval()`.

* **`get_vgg19_model(device: torch.device | str = "cpu", weights: VGG19_Weights | None = VGG19_Weights.DEFAULT) -> torch.nn.Sequential`**
  * *Deskripsi:* Memuat lapisan konvolusi `vgg19(weights=weights).features[:29]`, mengganti `ReLU(inplace=True)` menjadi `ReLU(inplace=False)`, membekukan seluruh parameter (`requires_grad = False`), mengunci mode `.eval()`, dan memindahkan model ke `device`.
  * *Catatan Pengujian:* Default `weights=VGG19_Weights.DEFAULT` memuat bobot ImageNet-1K; pengiriman `weights=None` didukung untuk pengujian struktur dan *autograd* secara cepat/offline.
* **`get_features(image_tensor: torch.Tensor, model: torch.nn.Module, layers: dict[str, str] | None = None) -> dict[str, torch.Tensor]`**
  * *Deskripsi:* Melakukan *forward pass* sekuensial melalui `model` (tanpa `torch.no_grad()` agar gradien tetap mengalir ke `image_tensor` jika `requires_grad=True`) dan mengekstrak dictionary tensor fitur sesuai pemetaan `layers` (default `DEFAULT_LAYERS`).
  * *Validasi & Exceptions:*
    * `TypeError` jika `image_tensor` bukan `torch.Tensor` atau `model` bukan `torch.nn.Module`.
    * `ValueError` jika `image_tensor` bukan 4D `(N, 3, H, W)` dengan `N >= 1`, `C == 3`, `H >= 16`, `W >= 16`, bukan floating-point, mengandung `NaN`/`Inf`, atau jika `layers` kosong / meminta indeks layer yang tidak ada di dalam `model`.

### C. Modul Loss Functions (`src/losses.py`) — PIC: A3

> **Kebijakan Fungsi Kerugian NST (Keputusan D-007):**
> 1. `DEFAULT_STYLE_WEIGHTS: dict[str, float] = {"conv1_1": 0.2, "conv2_1": 0.2, "conv3_1": 0.2, "conv4_1": 0.2, "conv5_1": 0.2}`.
> 2. Target referensi (`content_features` dan `style_gram_dict`) wajib di-`.detach()` di dalam perhitungan loss agar gradien hanya mengalir menuju fitur gambar hasil (`gen_features`).

* **`gram_matrix(tensor: torch.Tensor) -> torch.Tensor`**
  * *Deskripsi:* Menerima tensor fitur 4D `(B, C, H, W)` atau 3D `(C, H, W)` floating-point *finite*, merata-ratakan dimensi spasial dengan `.reshape(B, C, H * W)`, menghitung korelasi channel $G = (F \cdot F^T) / (C \cdot H \cdot W)$ via `torch.bmm`, dan mengembalikan tensor `(B, C, C)`.
* **`compute_content_loss(gen_features: torch.Tensor, content_features: torch.Tensor) -> torch.Tensor`**
  * *Deskripsi:* Memvalidasi kesamaan `shape` 4D `(B, C, H, W)`, `device`, tipe floating-point, dan nilai *finite*, lalu menghitung `F.mse_loss(gen_features, content_features.detach())`.
* **`compute_style_loss(gen_features_dict: dict[str, torch.Tensor], style_gram_dict: dict[str, torch.Tensor], layer_weights: dict[str, float] | None = None) -> torch.Tensor`**
  * *Deskripsi:* Menghitung penjumlahan berbobot $\sum_l w_l \cdot \text{MSE}(G(F_{\text{gen}}^l), G_{\text{style}}^l.\text{detach}())$ pada multi-layer style (default `layer_weights = DEFAULT_STYLE_WEIGHTS`). Mendukung `style_gram_dict[l]` yang sudah berupa Gram Matrix 3D `(B, C, C)` maupun peta fitur 4D `(B, C, H, W)`.
* **`compute_total_loss(content_loss: torch.Tensor, style_loss: torch.Tensor, alpha: float = 1.0, beta: float = 1e4) -> torch.Tensor`**
  * *Deskripsi:* Memvalidasi `content_loss` dan `style_loss` sebagai tensor skalar floating-point *finite* serta $\alpha, \beta \ge 0$ (bukan `bool`), lalu menghitung $L_{total} = \alpha \cdot L_{content} + \beta \cdot L_{style}$.

### D. Modul Optimasi (`src/optimization.py`) — PIC: A3

> **Kebijakan Loop Optimasi Piksel & Clamping ImageNet (Keputusan D-008):**
> 1. Clamping setiap iterasi wajib menggunakan batas per-channel ruang ternormalisasi ImageNet $\left[\frac{0 - \mu_c}{\sigma_c}, \frac{1 - \mu_c}{\sigma_c}\right]$ di dalam `with torch.no_grad():` (`clamp_imagenet_tensor_`), bukan clamping mentah `[0.0, 1.0]`.
> 2. Fitur konten target dan Gram Matrix gaya target wajib dihitung satu kali di awal sebelum loop optimasi dimulai.

* **`clamp_imagenet_tensor_(tensor: torch.Tensor) -> torch.Tensor`**
  * *Deskripsi:* Melakukan *clamping* *in-place* (di dalam `torch.no_grad()`) pada tensor 4D `(B, 3, H, W)` ke rentang valid ruang ternormalisasi ImageNet agar ekuivalen dengan rentang piksel RGB $[0.0, 1.0]$.
* **`run_style_transfer(content_tensor: torch.Tensor, style_tensor: torch.Tensor, model: torch.nn.Module, num_steps: int = 300, alpha: float = 1.0, beta: float = 1e4, optimizer_type: str = "lbfgs", learning_rate: float = 1.0, init_mode: str = "content", style_weights: Mapping[str, float] | None = None, content_layer: str = "conv4_2", seed: int = 42) -> tuple[torch.Tensor, list[float]]`**
  * *Deskripsi:* Menjalankan loop optimasi piksel gambar target menggunakan optimizer `"lbfgs"` (dengan `closure()`) atau `"adam"`, mendukung inisialisasi `"content"`, `"style"`, atau `"random"`, menerapkan `clamp_imagenet_tensor_` tiap langkah, dan mengembalikan `(generated_tensor.detach(), loss_history)`.

### E. Modul Visualisasi & Integrasi Pipeline (`src/visualization.py` & `main.py`) — PIC: A1

> **Kebijakan Visualisasi & Integrasi End-to-End (Keputusan D-009):**
> 1. Menggunakan backend non-interaktif Matplotlib (`matplotlib.use("Agg")`) agar pembuatan grafik berjalan stabil di lingkungan CLI/headless.
> 2. Seluruh hyperparameter dikelola melalui `config/config.yaml` dan dapat di-*override* lewat argumen CLI `main.py`.

* **`save_image(tensor: torch.Tensor | Image.Image, output_path: str | Path) -> None`**
  * *Deskripsi:* Memvalidasi ekstensi berkas tujuan (`.png`, `.jpg`, `.jpeg`), mengonversi tensor terstandarisasi ImageNet melalui `deprocess_image(tensor)` (atau menerima `PIL.Image.Image` langsung), membuat folder induk secara otomatis, dan menyimpan gambar RGB ke `output_path`.
* **`plot_loss_history(loss_history: Sequence[float], output_path: str | Path, title: str = ...) -> None`**
  * *Deskripsi:* Memvalidasi urutan nilai loss numerik *finite* non-negatif dan memplot kurva konvergensi *Total Loss* terhadap langkah iterasi (`1..N`) beserta anotasi *Initial Loss*, *Final Loss*, dan persentase penurunan loss ke `output_path`.
* **`save_comparison_grid(content_image: torch.Tensor | Image.Image, style_image: torch.Tensor | Image.Image, generated_image: torch.Tensor | Image.Image, output_path: str | Path, title: str = ...) -> None`**
  * *Deskripsi:* Menyusun dan menyimpan gambar panel perbandingan 3 kolom (*Content Image* | *Style Image* | *Stylized Output*) untuk keperluan evaluasi visual `S06-02` dan dokumentasi akhir Stage 7.
* **`main(argv: Sequence[str] | None = None) -> dict[str, Any]` (`main.py`)**
  * *Deskripsi:* Membaca `config/config.yaml` (`load_config`), menerapkan override argumen CLI (`build_arg_parser`), dan mengeksekusi rantai penuh `run_pipeline(config)`: `preprocess_image` $\rightarrow$ `get_vgg19_model` $\rightarrow$ `run_style_transfer` $\rightarrow$ `save_image`, `plot_loss_history`, dan `save_comparison_grid`.

---

## 3. Ketentuan Dimensi & Tipe Tensor Global

1. **Format Tensor & Ukuran:** Seluruh tensor gambar dalam pipeline wajib berbentuk 4D: `(N, C, H, W)` di mana $N=1$ (batch size), $C=3$ (RGB), $H$ = tinggi (*height*), $W$ = lebar (*width*). Seluruh parameter `size` menggunakan urutan `(H, W)`.
2. **Tipe Data:** `torch.float32`.
3. **Reproducibility:** Seluruh random initialization wajib dikunci dengan random seed di `config/config.yaml`.
4. **Pemisahan Modul:** Setiap modul hanya berinteraksi melalui fungsi antarmuka resmi di atas tanpa mengakses variabel internal modul lain.
