# DEMO_CHECKLIST — Kesiapan Demo & Presentasi Akhir

Checklist ini dipakai seluruh tim sebelum dan saat presentasi Proyek 1 —
Ethereal Neural Style Transfer. Disusun A4 (S07-02, 28/10/2026). Semua
perintah dan output di bawah sudah dijalankan dan diverifikasi.

Rujukan: [FINAL_REPORT.md](FINAL_REPORT.md) ·
[README.md](../README.md) · [HANDOFF.md](HANDOFF.md)

---

## 1. Checklist Teknis Pra-Demo

Kerjakan berurutan di laptop yang dipakai presentasi, paling lambat H-1.

- [ ] Remote terkini: `git pull origin main`
- [ ] Repository bersih: `git status` menampilkan `nothing to commit, working tree clean`
- [ ] Virtual environment `.venv` aktif (`source .venv/bin/activate` atau `.venv\Scripts\Activate.ps1`) dan `pip install -r requirements.txt` sudah dijalankan
- [ ] `pytest tests/ -v -rs` → **90 passed, 2 skipped (CUDA), 60 subtests passed**
- [ ] `python main.py --help` berjalan tanpa error dan menampilkan 16 opsi
- [ ] File output tersedia dan bisa dibuka:
  - [ ] `outputs/images/stylized_output.png`
  - [ ] `outputs/images/comparison_grid.png`
  - [ ] `outputs/graphs/loss_curve.png`
- [ ] `docs/FINAL_REPORT.md` terbuka di browser/GitHub dan ketiga gambar tampil
- [ ] Demo 1 (bagian 3) sudah dicoba sekali; catat waktu eksekusinya di laptop presentasi
- [ ] **Bobot VGG-19:** tentukan mode demo (lihat catatan di bawah)
- [ ] Laptop terhubung charger; mode hemat daya dimatikan agar waktu CPU stabil

> **Catatan mode bobot.** Artefak resmi di `outputs/` dihasilkan dengan VGG-19
> *fallback offline* (`--no-pretrained`), bukan bobot ImageNet
> ([FINAL_REPORT.md §4.4](FINAL_REPORT.md#44-verifikasi-ulang-oleh-a4-dan-keterbatasan)).
> - Untuk mereproduksi angka resmi (9.8481 → ±3.81), jalankan Demo 1 dengan
>   `--no-pretrained`.
> - Untuk demo dengan bobot pretrained ImageNet (sesuai D-001), unduh bobot
>   (±548 MB) **sebelum** hari H agar tidak terunduh saat demo:
>   `python -c "from torchvision.models import vgg19, VGG19_Weights; vgg19(weights=VGG19_Weights.DEFAULT)"`.
>   Angka loss akan berbeda dari angka resmi; siapkan penjelasannya.

---

## 2. Alur Presentasi per Anggota

Total ±25 menit (5–7 menit per anggota) + tanya-jawab.

| Urutan | Anggota | Durasi | Materi | Bahan Tayang |
|---|---|---|---|---|
| 1 | **A1 — Samuel Aaron** | 6–7 mnt | Tujuan proyek, arsitektur 5 modul + `main.py`, keputusan D-001–D-009, **Demo 1** (`main.py`) | `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, FINAL_REPORT §3 |
| 2 | **A2 — Marthin M. Tambunan** | 5 mnt | Pipeline preprocessing: load JPEG/PNG → RGB → resize Lanczos `size=(H, W)` → normalisasi ImageNet → tensor `(1, 3, H, W)`; kebalikannya di `deprocess_image`; kebijakan D-005 | `src/preprocessing.py`, FINAL_REPORT §3.1 |
| 3 | **A3 — Hafiz Fajar Ramadhan** | 6–7 mnt | Gram Matrix, content/style/total loss (rumus + implementasi `F.mse_loss`), loop optimasi L-BFGS/Adam, `clamp_imagenet_tensor_`, tuning `lr` 1.0 vs 0.5 | `src/losses.py`, `src/optimization.py`, FINAL_REPORT §2 & §4.3 |
| 4 | **A4 — Muhammad Rafiq Abdurrasyid** | 5–6 mnt | Metodologi Vibe Coding + Stage-Gate (plan → approval → eksekusi → log), audit PEP 8 (10 temuan), integration test & Temuan #11, konsolidasi 30 log / 31 entri JSONL, laporan akhir, **Demo 2 & 3** | `docs/LOG_INTERAKSI.md`, `tests/test_integration.py`, FINAL_REPORT §6 |

Transisi: A1 menutup dengan gambar hasil Demo 1 → A2 memulai dari "bagaimana
gambar itu menjadi tensor" → A3 dari "apa yang dioptimasi" → A4 menutup
dengan "bagaimana kami memastikan semuanya benar".

---

## 3. Perintah Live Demo

Output demo ditulis ke berkas `demo_*` yang di-*ignore* Git, sehingga artefak
resmi tidak tertimpa dan `git status` tetap bersih.

### Demo 1 — Pipeline NST penuh (A1)

```bash
python main.py --content assets/content/content_sample.png \
               --style assets/style/style_sample.png \
               --output-image outputs/images/demo_output.png \
               --comparison-image outputs/images/demo_grid.png \
               --loss-graph outputs/graphs/demo_loss.png \
               --steps 50 --optimizer lbfgs --lr 0.5 --no-pretrained
```

Output yang diharapkan (terverifikasi di macOS, CPU, Python 3.12):

```text
=== Ringkasan Eksekusi Neural Style Transfer ===
Perangkat       : cpu (Offline Calibrated VGG-19 (weights=None, fan_in))
Ukuran (H, W)   : (256, 256)
Optimizer       : LBFGS (steps=50, lr=0.5, init=content)
Bobot Loss      : alpha=1.0, beta=10000.0
Loss Awal       : 9.8481
Loss Akhir      : 3.8104 (-61.31%)
Waktu Eksekusi  : 16.41 detik
Gambar Hasil    : .../outputs/images/demo_output.png
Panel Komparasi : .../outputs/images/demo_grid.png
Grafik Loss     : .../outputs/graphs/demo_loss.png
```

Loss awal selalu 9.8481. Loss akhir bisa sedikit berbeda antar-mesin
(A1 di Windows: 3.8076, −61.34%). Waktu bergantung CPU (16–37 detik).
Setelah selesai, buka `outputs/images/demo_grid.png`.

> Opsi yang benar adalah `--steps` dan `--output-image`. Versi lama
> (`--num-steps`, `--output`) gagal dengan
> `main.py: error: unrecognized arguments`.

### Demo 2 — Seluruh test suite (A4)

```bash
pytest tests/ -v -rs
```

Output yang diharapkan (baris akhir):

```text
SKIPPED [1] tests/test_feature_extractor.py:237: CUDA tidak tersedia
SKIPPED [1] tests/test_preprocessing.py:200: CUDA tidak tersedia
============== 90 passed, 2 skipped, 60 subtests passed in ~10s ===============
```

Sorot `tests/test_integration.py` (16 test) sebagai bukti rantai lintas-modul.

### Demo 3 — Kepatuhan PEP 8 berkas A4 (A4)

```bash
pycodestyle scripts/log_interaction.py tests/test_log_interaction.py tests/test_integration.py
```

Output yang diharapkan: **tidak ada output** (exit code 0 = 0 pelanggaran).

### Rencana Cadangan

| Masalah saat demo | Tindakan |
|---|---|
| Demo 1 terlalu lama / laptop lambat | Ganti ke `--steps 10` (±5 detik; loss 9.8481 → 5.3141, −46.04%), atau langsung tampilkan artefak resmi di `outputs/` |
| Error impor / `.venv` rusak | Tampilkan hasil dari FINAL_REPORT §4 dan rekaman layar yang disiapkan H-1 |
| Proyektor tidak menampilkan gambar | Buka `README.md` di GitHub dari HP/laptop lain |

---

## 4. FAQ Teknis Tanya-Jawab

| # | Pertanyaan | Jawaban Ringkas | Penjawab |
|---|---|---|---|
| 1 | Mengapa VGG-19 dan bukan ResNet? | Gatys et al. memakai VGG-19 dan hasilnya menjadi acuan. VGG berupa tumpukan konvolusi sederhana tanpa *skip connection* atau *batch norm*, sehingga fitur tiap layer bersih dan multi-skala; fitur ResNet umumnya menghasilkan stilisasi yang kurang baik tanpa penyesuaian tambahan. | A1 |
| 2 | Mengapa bobot VGG-19 dibekukan? | Yang dioptimasi adalah piksel gambar, bukan model (D-001, D-002). VGG hanya "penggaris" konten dan gaya. Membekukan (`requires_grad=False`) juga menghemat ±42 MB buffer gradien dan menjamin model tidak berubah (diuji di `test_optimization` & `test_integration`). | A1 |
| 3 | Apa fungsi Gram Matrix? | Mengukur korelasi antar-kanal fitur di seluruh posisi: $G = F F^\top / (C H W)$. Informasi "di mana" hilang, yang tersisa "pola apa yang muncul bersama" — itulah tekstur/gaya. | A3 |
| 4 | Mengapa optimizer L-BFGS, bukan Adam saja? | Masalahnya deterministik (satu gambar, tanpa *mini-batch*), cocok untuk L-BFGS yang memakai informasi kelengkungan sehingga turun lebih cepat per langkah. Adam tetap didukung (`--optimizer adam`) dan diuji. | A3 |
| 5 | Mengapa `learning_rate=0.5`, bukan `1.0`? | Dengan `lr=1.0` loss melonjak di Step 5 (7.35 → 10.90) karena langkah kuasi-Newton awal terlalu agresif (`max_iter=1`, tanpa *line search*). `lr=0.5` turun mulus; selisih penurunan akhir hanya ±2 poin persen (−61.34% vs −63.23%). | A3 / A1 |
| 6 | Apa itu `clamp_imagenet_tensor_`? | Setelah tiap langkah, piksel dibatasi per kanal ke $[(0-\mu_c)/\sigma_c,\ (1-\mu_c)/\sigma_c]$, yaitu rentang RGB [0, 1] di ruang ternormalisasi ImageNet (D-008). *Clamp* mentah ke [0, 1] akan merusak warna karena nilai negatif di ruang ini sah. | A3 |
| 7 | Bagaimana Stage-Gate Sequential diterapkan? | 8 stage (0–7), tiap stage punya PIC dan syarat lolos. Setiap langkah: rencana → disetujui A1 → eksekusi + test → log di `logs/` → handoff tercatat (Handoff #1–#11). Stage berikutnya baru dibuka setelah gate disetujui. | A4 |
| 8 | Mengapa $\alpha=1.0$ dan $\beta=10^4$? | Karena Gram Matrix dinormalisasi $1/(CHW)$ dan loss memakai rata-rata, nilai style loss jauh lebih kecil dari content loss. $\beta=10^4$ menyeimbangkan keduanya sehingga struktur tetap terlihat tetapi gaya terbawa. Rasio dapat diubah lewat `--alpha`/`--beta`. | A3 |
| 9 | Apa itu `init_mode='content'`? | Gambar hasil dimulai dari salinan gambar konten, sehingga struktur sudah benar sejak awal dan optimasi tinggal menambahkan gaya. Alternatif: `style` atau `random` (ber-*seed*). | A3 |
| 10 | Berapa kompleksitas waktu algoritma ini? | Per iterasi didominasi forward + backward VGG-19: $\mathcal{O}(\sum_l C_{l-1} C_l k^2 H_l W_l)$ plus Gram $\mathcal{O}(\sum_l C_l^2 H_l W_l)$; untuk arsitektur tetap setara $\mathcal{O}(H \times W)$. Total $\mathcal{O}(T \times H \times W)$ untuk $T$ langkah; loop optimasi ≥95% waktu. | A4 / A3 |
| 11 | Apakah hasil di laporan memakai bobot ImageNet? | Tidak. Lingkungan A1 tidak bisa mengunduh bobot, sehingga `main.py` memakai fallback VGG-19 acak terkalibrasi `fan_in`. Ini diverifikasi A4 (loss awal identik) dan dicatat di FINAL_REPORT §4.4. Pipeline tetap terbukti benar; kualitas dengan bobot ImageNet umumnya lebih baik. | A1 / A4 |
| 12 | Mengapa kurva loss datar di Step 1–4? | L-BFGS masih membangun histori aproksimasi Hessian dengan langkah yang sangat kecil; setelah itu loss turun tajam lalu melandai. | A3 |
| 13 | Bug apa yang hanya ketahuan lewat integration test? | Temuan #11: `preprocess_image` menghasilkan tensor tidak kontigu (akibat `permute`), dan L-BFGS gagal di `grad.view(-1)`. Unit test lolos karena memakai `torch.randn` yang kontigu. Diperbaiki dengan `clone(memory_format=torch.contiguous_format)` (D-009). | A4 |
| 14 | Bagaimana interaksi AI didokumentasikan? | Tiap langkah punya log Markdown di `logs/A1`–`A4` sesuai `LOG_TEMPLATE.md` (prompt utuh, alasan, output, error nyata, test, persetujuan), plus entri JSONL lewat `scripts/log_interaction.py`. Indeks lengkap di `docs/LOG_INTERAKSI.md` (30 log, 31 entri). | A4 |
| 15 | Apa yang dioptimasi, berapa parameternya? | Hanya piksel gambar hasil: $3 \times 256 \times 256 = 196{,}608$ nilai. Sebanyak ±10.59 juta parameter VGG tetap beku. | A1 |

---

## 5. Checklist Penutupan (Setelah Presentasi)

- [ ] Hapus berkas demo: `outputs/images/demo_*.png`, `outputs/graphs/demo_*.png`
- [ ] `git status` bersih
- [ ] A1 memberi persetujuan akhir Handoff #11 dan menandai Stage 7 `COMPLETED`
