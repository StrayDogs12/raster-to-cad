# Raster Mask Cleaning — Architectural Drawing Layers

Pipeline sederhana untuk mengubah hasil render layer gambar teknik (latar hitam, garis terang) menjadi mask biner PNG yang bersih dan lossless.

## Struktur

```text
mask_cleaning_project/
├── vision/
│   ├── __init__.py
│   └── mask_cleaning.py
├── tests/
│   └── test_mask_cleaning.py
├── scripts/
│   └── run_mask_cleaning.py
├── requirements.txt
└── README.md
```

## 1. Setup di Windows PowerShell

Buka terminal pada folder `mask_cleaning_project`:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika PowerShell memblokir aktivasi environment, jalankan perintah ini hanya untuk sesi terminal saat ini:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Jalankan untuk folder layer

Simpan kelima JPG layer ke satu folder, misalnya `input_layers`:

```powershell
python scripts/run_mask_cleaning.py --input input_layers --output output_masks
```

Hasilnya berupa `*_clean.png` dan `cleaning_report.json` di `output_masks`.

Untuk satu file:

```powershell
python scripts/run_mask_cleaning.py --input "input_layers\Mask Layer Putih & Cyan — Struktur Utama (0,50 – 0,70 mm).jpg" --output output_masks
```

## 3. Jalankan test

```powershell
python -m pytest -q
```

## Parameter awal yang disarankan

- `--threshold-mode fixed --threshold 42`: cocok sebagai titik awal untuk file contoh yang berlatar hitam dengan garis putih/abu-abu.
- `--close-kernel 3`: menutup putus kecil akibat kompresi. Jika detail rapat menyatu, gunakan `--close-kernel 1` untuk menonaktifkan.
- `--min-component-area 1`: jangan buang komponen kecil. Ini penting untuk teks, titik, panah, simbol, dan garis putus-putus.

Contoh lebih konservatif tanpa morphological closing:

```powershell
python scripts/run_mask_cleaning.py --input input_layers --output output_masks_conservative --threshold 42 --close-kernel 1
```


## 4. Validasi mask sebelum vectorization

Setelah proses cleaning, jalankan pemeriksaan kualitas raster:

```powershell
python scripts/run_mask_validation.py --source input_layers --masks output_masks --report validation/validation_report.json
```

Validator mencocokkan nama file sumber dengan nama mask (`nama_asli_clean.png`) dan memeriksa:

- kecocokan resolusi sumber dan mask;
- nilai mask biner (0 dan 255);
- jumlah connected components dan proporsi komponen sangat kecil;
- persentase foreground dan bounding box;
- piksel yang menyentuh tepi gambar;
- rasio piksel mask yang didukung oleh piksel terang pada sumber;
- perkiraan jumlah endpoint dan junction pada skeleton.

Status:

- `PASS`: seluruh pemeriksaan guardrail yang aktif lolos.
- `REVIEW`: tidak ada error fatal, tetapi ada indikator yang perlu ditinjau.
- `FAIL`: ada masalah fatal seperti mask kosong atau dimensi tidak cocok.

Threshold source-support dapat disesuaikan:

```powershell
python scripts/run_mask_validation.py --min-support 0.80 --source-threshold 20
```

Hasil lengkap disimpan sebagai JSON. Nilai guardrail bersifat heuristik dan harus disesuaikan dengan jenis gambar. `PASS` tidak membuktikan bahwa pemisahan objek oleh Gemini benar secara semantik, bahwa garis tidak hilang, atau bahwa geometri CAD sudah akurat. Selalu bandingkan overlay mask dengan gambar sumber sebelum vectorization.

## 5. Urutan kerja yang disarankan

1. Jalankan `run_mask_cleaning.py`.
2. Jalankan `run_mask_validation.py`.
3. Tinjau semua item `REVIEW` dan bandingkan mask dengan sumber.
4. Baru lanjutkan ke vectorization jika layer sudah disetujui secara visual.

## Catatan akurasi

File input yang diberikan berupa JPG 2048×1143, bukan mask warna murni. Karena itu thresholding akan menghilangkan sebagian informasi tingkat abu-abu dan ketebalan garis. Simpan JPG asli sebagai sumber; jangan menimpa file sumber. Untuk redraw CAD presisi, hasil mask ini tetap perlu tahap vectorization, fitting garis/arc, pengelompokan layer, dan pemeriksaan skala. Mask cleaning tidak dapat memulihkan detail yang memang tidak ada pada raster atau memperbaiki geometri yang salah dari model vision.
