# DECISIONS (Architecture Decision Records)

| ID | Keputusan | Alasan | Alternatif | Disetujui |
|---|---|---|---|---|
| D-001 | VGG-19 pretrained frozen | Standar NST Gatys et al. | Train from scratch | Seluruh Tim (A1 Lead) |
| D-002 | Optimasi pixel generated image | Mempertahankan bobot model tetap beku | Fine-tune weights | Seluruh Tim (A1 Lead) |
| D-003 | Modular struktur (5 modul src) | Kolaborasi multi-agent terisolasi | Monolitik satu skrip | Seluruh Tim (A1 Lead) |
| D-004 | Batas scope image loader S02-02 | Penolakan parameter size hingga S02-03 | Resize di loader | A1 (Samuel) & A2 (Marthin) |
| D-005 | Kebijakan Tunggal Parameter `size=(H, W)` & Preprocessing Tensor S02-03 | Mencegah tertukarnya `(W, H)` Pillow vs `(H, W)` PyTorch serta menyamakan nama parameter `size` | Memakai `target_size` terpisah atau urutan `(W, H)` | A1 (Samuel) & A2 (Marthin) |
| D-006 | Spesifikasi Kontrak & Arsitektur VGG-19 Feature Extractor (`S03-01` & `S03-02`) | Menjamin ekstraksi fitur `conv1_1`..`conv5_1` & `conv4_2` bebas mutasi `ReLU(inplace=True)`, hemat memori lewat pemotongan `features[:29]`, dan aman bagi aliran gradien ke `generated_img` | Menggunakan `vgg19.features` utuh 37 layer dengan `ReLU(inplace=True)` | A1 (Samuel Aaron - Lead/Arsitek) |

---

## D-004: Batas loader S02-02 (Superseded pada bagian `size` oleh D-005 di S02-03)

- **Konteks:** Signature `load_image(image_path, size=None)` sudah direncanakan, tetapi resize termasuk S02-03 dan perilaku parameter belum dijelaskan.
- **Keputusan (Saat S02-02):** Hasil berupa `PIL.Image` RGB mandiri, ukuran asli saat `size=None`. Nilai `size` selain None ditolak dengan `NotImplementedError` sampai S02-03.
- **Alternatif:** Mengimplementasikan resize pada loader pada tahap S02-02; tidak dipilih untuk mematuhi *Stage-Gate Sequential*.
- **Persetujuan:** Disetujui oleh A2 dan diverifikasi/diterima oleh A1 (Samuel Aaron).
- **Dampak:** `src/preprocessing.py`, `tests/test_preprocessing.py`; tidak mengubah modul model/loss.
- **Pencatatan Sistem:** 2026-10-25T01:57:30+07:00 (WIB).

---

## D-005: Kebijakan Tunggal Parameter `size=(H, W)`, Resize & Normalisasi Tensor S02-03

- **Konteks:**
  1. Pada S02-02 fungsi `load_image` menggunakan nama parameter `size`, sedangkan draf awal S02-03 sempat menyebut `target_size` pada `preprocess_image`. Dua nama berbeda untuk konsep yang sama berisiko menimbulkan kebingungan.
  2. Terdapat perbedaan konvensi urutan dimensi antara pustaka Pillow (`PIL.Image.resize` mengharapkan `(width, height) = (W, H)`) dan PyTorch Tensor (`[N, C, H, W]` menggunakan `(height, width) = (H, W)`). Jika tidak diatur dalam satu kebijakan tegas, dimensi tinggi dan lebar akan tertukar pada gambar non-persegi.
- **Keputusan Teknis (Satu Kebijakan Tunggal):**
  1. **Penyeragaman Nama Parameter (`size`):**
     - Baik `load_image` maupun `preprocess_image` wajib menggunakan nama parameter yang sama: **`size: tuple[int, int] | None`**. Istilah `target_size` dihapus dari kontrak.
  2. **Konvensi Urutan Dimensi `size = (height, width)` = `(H, W)`:**
     - Di seluruh fungsi publik `src/preprocessing.py`, tuple `size` **selalu berarti `(height, width)` = `(H, W)`** selaras dengan dimensi spasial tensor PyTorch `(1, 3, H, W)`.
     - Saat memanggil `PIL.Image.resize(...)` secara internal, fungsi wajib membalik tuple `(height, width)` menjadi `(width, height)`:
       `image.resize((width, height), Image.Resampling.LANCZOS)`.
     - Dengan aturan ini, untuk `size=(256, 512)`:
       - `load_image(path, size=(256, 512))` mengembalikan `PIL.Image` dengan `img.height == 256` dan `img.width == 512` (`img.size == (512, 256)`).
       - `preprocess_image(img, size=(256, 512))` mengembalikan Tensor dengan `tensor.shape == (1, 3, 256, 512)`.
  3. **Validasi Ketat Parameter `size`:**
     - `size` harus bernilai `None` atau `tuple` berisi tepat 2 integer murni (`int`, bukan `bool`) dengan `height > 0` dan `width > 0`.
     - Input tidak valid (misal nilai $\le 0$, panjang tuple $\neq 2$, tipe `list`/`int` tunggal/`float`/`bool`) wajib ditolak dengan `ValueError`.
  4. **Kebijakan Alur Resize (Mencegah Double Resize):**
     - Metode interpolasi: *Direct exact resize* ke `(H, W)` menggunakan `Image.Resampling.LANCZOS` (tanpa *cropping*), sehingga gambar konten dan gaya memiliki ukuran tensor `(1, 3, H, W)` yang identik jika diberi `size` yang sama.
     - Pada `load_image(image_path, size=None)`:
       - `size=None`: ukuran asli dipertahankan.
       - `size=(H, W)`: gambar di-resize ke tinggi `H` dan lebar `W`.
     - Pada `preprocess_image(image, size=(512, 512), device="cpu") -> torch.Tensor`:
       - Default `size=(512, 512)`.
       - Jika `image` berupa `str | Path`: memanggil `load_image(image, size=size)` (resize cukup terjadi sekali di `load_image`).
       - Jika `image` berupa `PIL.Image.Image`: memastikan mode `"RGB"`, lalu jika `size is not None` dan `(image.height, image.width) != size`, melakukan resize `LANCZOS` ke `(H, W)`. Jika `size is None`, mempertahankan ukuran gambar apa adanya.
       - Mengonversi ke tensor `torch.float32` skala `[0.0, 1.0]`, menormalisasi dengan konstanta ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$), menambahkan dimensi batch `(1, 3, H, W)`, dan memindahkan ke `device`.
  5. **Kontrak `deprocess_image(tensor: torch.Tensor) -> Image.Image`:**
     - Menerima tensor 4D `(1, 3, H, W)` atau 3D `(3, H, W)` (selain itu lempar `ValueError`).
     - Tidak mengubah tensor input secara *in-place* (`detach().cpu().clone()`).
     - Denormalisasi ($x \times \sigma + \mu$), *clamping* ke `[0.0, 1.0]`, konversi ke `uint8` `[0, 255]`, dan mengembalikan `PIL.Image.Image` RGB dengan `height == H` dan `width == W`.
  6. **Transisi Unit Test S02-02 ke S02-03:**
     - Tes `test_resize_not_silently_ignored` dari S02-02 yang menguji `NotImplementedError` diperbarui pada S02-03 menjadi pengujian resize ukuran non-persegi `(H, W)` dan validasi `ValueError` untuk `size` tidak valid.
- **Alternatif:** Menggunakan urutan `(W, H)` milik Pillow pada `load_image` dan `(H, W)` pada `preprocess_image`; ditolak keras karena pasti menyebabkan bug tertukarnya tinggi dan lebar.
- **Persetujuan:** Disetujui oleh Samuel Aaron (A1 - Ketua / Arsitek) dan Marthin M. Tambunan (A2).
- **Dampak:** `docs/ARCHITECTURE.md`, `src/preprocessing.py`, `tests/test_preprocessing.py`.
- **Pencatatan Sistem:** 2026-10-25T03:52:00+07:00 (WIB).

---

## D-006: Spesifikasi Kontrak & Arsitektur VGG-19 Feature Extractor (`S03-01` & `S03-02`)

- **Konteks:**
  1. Setelah Stage 2 (`src/preprocessing.py`) resmi selesai, Stage 3 (`src/feature_extractor.py`) memerlukan kontrak teknis yang presisi agar tensor representasi fitur yang dihasilkan dapat langsung dikonsumsi oleh modul Loss (`src/losses.py`) dan Optimasi (`src/optimization.py`) milik A3 di Stage 4.
  2. Arsitektur `torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features` terdiri dari 37 sub-modul (`0` s.d. `36`) dan secara bawaan menggunakan `nn.ReLU(inplace=True)` tepat setelah setiap layer `nn.Conv2d`. Apabila `inplace=True` tidak diganti menjadi `inplace=False`, maka tensor keluaran `Conv2d` yang disimpan ke dalam dictionary `features` akan ikut termodifikasi secara *in-place* (nilai negatifnya terpotong menjadi nol) saat *forward pass* berlanjut ke layer ReLU berikutnya, serta berisiko merusak grafik komputasi *autograd* PyTorch.
  3. Layer terdalam yang digunakan oleh algoritma NST Gatys et al. adalah `conv5_1` pada indeks `'28'`. Mengeksekusi layer `'29'` hingga `'36'` pada setiap iterasi optimasi hanya membuang komputasi dan memori.
- **Keputusan Teknis (`D-006`):**
  1. **Pemetaan Layer Default (`DEFAULT_LAYERS`):**
     - Indeks string pada `vgg19.features` dipetakan ke nama kanonik Gatys et al.:
       ```python
       DEFAULT_LAYERS: dict[str, str] = {
           "0": "conv1_1",   # Style layer 1 (64 ch, H x W)
           "5": "conv2_1",   # Style layer 2 (128 ch, H/2 x W/2)
           "10": "conv3_1",  # Style layer 3 (256 ch, H/4 x W/4)
           "19": "conv4_1",  # Style layer 4 (512 ch, H/8 x W/8)
           "21": "conv4_2",  # Content layer (512 ch, H/8 x W/8)
           "28": "conv5_1",  # Style layer 5 (512 ch, H/16 x W/16)
       }
       ```
     - Konstanta tambahan yang diekspos untuk kemudahan Stage 4 & Stage 6:
       - `CONTENT_LAYERS: tuple[str, ...] = ("conv4_2",)`
       - `STYLE_LAYERS: tuple[str, ...] = ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")`
  2. **Kontrak `get_vgg19_model(device: torch.device | str = "cpu", weights: VGG19_Weights | None = VGG19_Weights.DEFAULT) -> nn.Sequential`:**
     - Memuat `torchvision.models.vgg19(weights=weights).features`. Parameter `weights` bernilai default `VGG19_Weights.DEFAULT` (ImageNet-1K pretrained), namun menerima `weights=None` agar unit test dapat menguji struktur, bentuk tensor, dan aliran gradien secara instan/offline tanpa mengunduh bobot ~548 MB berulang kali.
     - Memotong (*truncate*) sekuens menjadi 29 layer pertama (`features[:29]`, indeks `0` s.d. `28`), sehingga *forward pass* berhenti tepat setelah `conv5_1`.
     - Mengganti seluruh `nn.ReLU(inplace=True)` di dalam sekuens dengan `nn.ReLU(inplace=False)` agar tensor aktivasi `Conv2d` tidak termutasi *in-place* oleh layer ReLU berikutnya dan aman bagi *backward pass*.
     - Membekukan seluruh parameter (`param.requires_grad_(False)`), mengunci model ke mode evaluasi (`.eval()`), dan memindahkan model ke `device`.
  3. **Kontrak `get_features(image_tensor: torch.Tensor, model: nn.Module, layers: dict[str, str] | None = None) -> dict[str, torch.Tensor]`:**
     - Validasi Input:
       - `image_tensor` harus berupa `torch.Tensor` (selain itu lempar `TypeError`), berdimensi 4D `(N, 3, H, W)` dengan `N >= 1`, `C == 3`, `H >= 16, W >= 16` (minimal 16 piksel karena terdapat 4 layer `MaxPool2d` sebelum `conv5_1`), bertipe floating-point (`is_floating_point()`), dan seluruh elemennya bernilai hingga (`torch.isfinite(image_tensor).all()`). Jika melanggar dimensi/dtype/finite, lempar `ValueError`.
       - `model` harus berupa `torch.nn.Module` (selain itu lempar `TypeError`).
       - Jika `layers is None`, gunakan `DEFAULT_LAYERS`. Jika `layers` diberikan, harus berupa `dict[str, str]` non-kosong dan seluruh indeks kuncinya harus ditemukan di dalam `model` (jika tidak valid/tidak lengkap, lempar `ValueError`).
     - Perilaku Forward Pass & Aliran Gradien:
       - Melakukan iterasi sekuensial `for name, layer in model._modules.items(): x = layer(x)` tanpa membungkus dengan `torch.no_grad()`, sehingga grafik komputasi *autograd* tetap terhubung dari fitur keluaran menuju `image_tensor` ketika `image_tensor.requires_grad == True`.
       - Berhenti lebih awal (*early exit*) segera setelah seluruh layer yang diminta pada `layers` telah terkumpul.
       - Mengembalikan `dict[str, torch.Tensor]` yang memetakan nama kanonik (misal `'conv4_2'`) ke tensor fitur 4D `(N, C_l, H_l, W_l)`.
- **Alternatif:** Membiarkan `vgg19.features` utuh dengan `ReLU(inplace=True)`; ditolak karena mutasi *in-place* mengubah nilai tensor fitur `Conv2d` yang disimpan di dictionary dan berisiko menyebabkan kegagalan `autograd` saat optimasi piksel di Stage 4.
- **Persetujuan:** Disetujui oleh Samuel Aaron (A1 - Ketua / Arsitek).
- **Dampak:** `docs/ARCHITECTURE.md`, `src/feature_extractor.py`, `tests/test_feature_extractor.py`, `docs/planning/A1_PLAN.md`.
- **Pencatatan Sistem:** 2026-10-25T16:25:31+07:00 (WIB).
