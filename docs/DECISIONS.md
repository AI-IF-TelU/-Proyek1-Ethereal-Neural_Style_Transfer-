# DECISIONS

| ID | Keputusan | Alasan | Alternatif | Disetujui |
|---|---|---|---|---|
| D-001 | VGG-19 pretrained frozen | Standar NST | Train from scratch | A1 (proposed) |
| D-002 | Optimasi pixel | Sesuai prinsip | Fine-tune | A1 (proposed) |
| D-003 | Modular struktur | Kolaborasi multi-agent | Monolitik | A1 (proposed) |

## D-004: Batas loader S02-02

- Konteks: signature `load_image(image_path, size=None)` sudah direncanakan,
  tetapi resize termasuk S02-03 dan perilaku parameter belum dijelaskan.
- Keputusan: hasil berupa `PIL.Image` RGB mandiri, ukuran asli saat `size=None`.
  Nilai `size` selain None ditolak dengan `NotImplementedError` sampai S02-03.
- Alternatif: mengimplementasikan resize pada loader sekarang; tidak dipilih.
- Persetujuan: A2 memilih jawaban "Setujui batas S02-02" pada sesi ini.
  Ini persetujuan batas implementasi, bukan persetujuan hasil atau seluruh tim.
- Dampak: `src/preprocessing.py`, `tests/test_preprocessing.py`; tidak mengubah
  modul model/loss. Kontrak resize berikutnya masih perlu ditentukan pada S02-03.
- Pencatatan sistem: 2026-10-25T01:57:30.051711+07:00 (WIB).
- Commit: belum dibuat; tidak melakukan commit tanpa permintaan pengguna.
