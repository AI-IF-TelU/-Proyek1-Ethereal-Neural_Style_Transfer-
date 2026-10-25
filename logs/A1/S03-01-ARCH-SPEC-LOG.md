# S03-01-ARCH-SPEC: Perencanaan Arsitektur & Kontrak Modul VGG-19 Feature Extractor (D-006)

## Metadata
- **Nama Kelompok:** Proyek 1 - Ethereal Neural Style Transfer
- **Metodologi:** SDLC + Stage-Gate Sequential + Vibe Engineering
- **Tools:** VS Code, Git, Python, PyTorch, Torchvision
- **Tanggal Sesi:** 2026-10-25
- **Anggota:** A1 — Samuel Aaron (Ketua, Prompt Engineer & Software Architect)
- **Target Modul:** Spesifikasi Kontrak `src/feature_extractor.py` & `tests/test_feature_extractor.py` (`docs/DECISIONS.md`, `docs/ARCHITECTURE.md`)
- **AI Model:** Gemini Assistant
- **Tautan Sesi Chat:** Sesi aktif VS Code AI Assistant
- **Step ID:** S03-01 Planning & Architecture Specification (`D-006`)
- **Phase:** Architecture Planning & Contract Definition

---

## 0. TAHAP PLANNING (Perumusan Keputusan Arsitektur D-006)

### Prompt Kelompok
> "Setelah Stage 2 ditutup, susun kesimpulan reasoning dan spesifikasi kontrak arsitektur untuk tahap planning selanjutnya yaitu Stage 3 (`S03-01` & `S03-02`: VGG-19 Feature Extractor) pada `docs/DECISIONS.md` dan `docs/ARCHITECTURE.md`."

### Alasan Khusus
Dalam algoritma Neural Style Transfer (Gatys et al., 2016), modul `src/feature_extractor.py` berfungsi sebagai tulang punggung (*backbone*) tetap yang mengekstrak representasi konten (`conv4_2`) dan representasi gaya (`conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`). Sebelum menulis kode implementasi, diperlukan keputusan arsitektur formal (**D-006**) untuk mencegah tiga jebakan teknis klasik pada implementasi VGG-19 di PyTorch:
1. **Mutasi *In-Place* oleh `nn.ReLU(inplace=True)`:** Bawaan `torchvision.models.vgg19` menggunakan `ReLU(inplace=True)` pada indeks ganjil (`1, 6, 11, 20, 22, 29`). Jika tidak diganti menjadi `inplace=False`, tensor keluaran `Conv2d` yang disimpan di dictionary fitur akan tertimpa secara *in-place* ketika *forward pass* melewati layer ReLU sesudahnya, serta berpotensi memicu galat *autograd* saat *backward pass*.
2. **Komputasi Layer Sia-sia di Atas `conv5_1`:** `vgg19.features` memiliki 37 layer (`0..36`), padahal layer terdalam NST adalah `conv5_1` di indeks `'28'`. Memotong model menjadi `features[:29]` menghemat memori dan mempercepat setiap iterasi optimasi.
3. **Keseimbangan Parameter Beku vs Aliran Gradien Input:** Seluruh bobot VGG-19 wajib dibekukan (`requires_grad=False`) dan diset `.eval()`, tetapi `get_features()` tidak boleh dibungkus `torch.no_grad()` agar gradien dari *content/style loss* tetap mengalir balik ke `generated_img` (`requires_grad=True`).

---

## 1. SPESIFIKASI TEKNIS YANG DITETAPKAN (D-006)

### A. Pemetaan Layer Kanonik (`DEFAULT_LAYERS`)
| Indeks `vgg19.features` | Nama Kanonik | Peran NST | Jumlah Channel ($C_l$) | Resolusi Spasial ($H_l \times W_l$) |
| :---: | :---: | :---: | :---: | :---: |
| `'0'` | `'conv1_1'` | Style Layer 1 | $64$ | $H \times W$ |
| `'5'` | `'conv2_1'` | Style Layer 2 | $128$ | $\lfloor H/2 \rfloor \times \lfloor W/2 \rfloor$ |
| `'10'` | `'conv3_1'` | Style Layer 3 | $256$ | $\lfloor H/4 \rfloor \times \lfloor W/4 \rfloor$ |
| `'19'` | `'conv4_1'` | Style Layer 4 | $512$ | $\lfloor H/8 \rfloor \times \lfloor W/8 \rfloor$ |
| `'21'` | `'conv4_2'` | Content Layer | $512$ | $\lfloor H/8 \rfloor \times \lfloor W/8 \rfloor$ |
| `'28'` | `'conv5_1'` | Style Layer 5 | $512$ | $\lfloor H/16 \rfloor \times \lfloor W/16 \rfloor$ |

### B. Kontrak Fungsi Publik (`src/feature_extractor.py`)
1. `get_vgg19_model(device: torch.device | str = "cpu", weights: VGG19_Weights | None = VGG19_Weights.DEFAULT) -> torch.nn.Sequential`
   * Memuat `vgg19(weights=weights).features[:29]`.
   * Mengganti setiap sub-modul `nn.ReLU(inplace=True)` dengan `nn.ReLU(inplace=False)`.
   * Membekukan seluruh parameter (`requires_grad = False`), memanggil `.eval()`, dan memindahkan model ke `device`.
2. `get_features(image_tensor: torch.Tensor, model: torch.nn.Module, layers: dict[str, str] | None = None) -> dict[str, torch.Tensor]`
   * Memvalidasi `image_tensor` sebagai tensor 4D `(N, 3, H, W)` bertipe floating-point dengan nilai *finite* dan ukuran spasial minimal $16 \times 16$.
   * Mengeksekusi *forward pass* sekuensial hingga seluruh layer pada `layers` (default `DEFAULT_LAYERS`) terkumpul, lalu mengembalikan `dict[str, torch.Tensor]`.

---

## 2. DAMPAK DOKUMENTASI & KESIAPAN IMPLEMENTASI
- **Status Keputusan D-006:** **APPROVED (Disetujui oleh A1 — Samuel Aaron)** ✅
- **File Diperbarui:** `docs/DECISIONS.md`, `docs/ARCHITECTURE.md`, `docs/planning/A1_PLAN.md`, `docs/planning/PROGRESS_TRACKER.md`, `logs/A1/S03-01-ARCH-SPEC-LOG.md`.
- **Langkah Selanjutnya:** Melanjutkan ke implementasi kode `src/feature_extractor.py` (`S03-01`) dan pengujian unit `tests/test_feature_extractor.py` (`S03-02`).
