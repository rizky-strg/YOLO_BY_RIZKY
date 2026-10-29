# Panduan Tahap 2: Transfer Learning Objek PBL (P2)

Folder ini digunakan untuk mengambil data citra objek PBL baru Anda dan melatih model klasifikasi berbasis **Transfer Learning (ResNet-18 & MobileNetV3)**.

---

## 🛠️ Langkah Pelaksanaan Step-by-Step

### 1. Cek Index Kamera Anda
Pastikan kamera robot/webcam terhubung, lalu cek indexnya:
```bash
python ../01_P1_Camera_Calibration/check_cameras.py
```
*(Catat index kamera yang aktif, misalnya index `1` atau `2`).*

---

### 2. Mengambil Foto Objek PBL Baru (`capture.py`)
Gunakan skrip `capture.py` untuk mengambil data citra objek PBL. Skrip ini otomatis menerapkan koreksi distorsi lensa P1 (`realtime_calibration.npz`):

```bash
# Format: python capture.py <nama_objek> <kondisi_lingkungan> --cam <nomor_kamera>
# Contoh jika objek Anda adalah "komponen_A":
python capture.py komponen_A terang  --cam 2
python capture.py komponen_A redup   --cam 2
python capture.py komponen_A jendela --cam 2

# Ulangi untuk objek kedua (misalnya "komponen_B"):
python capture.py komponen_B terang  --cam 2
python capture.py komponen_B redup   --cam 2
python capture.py komponen_B jendela --cam 2

# Ambil gambar latar (background kosong tanpa objek):
python capture.py latar ruang-lab --cam 2
```

**Kontrol Saat Mengambil Foto:**
* Tekan tombol **`SPACE`** : Menyimpan 1 frame foto.
* Tekan tombol **`Q`** : Selesai / keluar.
* **Target Data**: Minimal $\ge 50$ foto per kelas objek dengan variasi sudut pandang, jarak, dan 2–3 kondisi pencahayaan berbeda.
* *Foto otomatis tersimpan di `dataset_raw/<nama_objek>/` dan dicatat ke `dataset_raw/metadata.csv`.*

---

### 3. Membagi Dataset Train & Val (`split.py`)
Jalankan skrip pembagian data yang dirancang untuk mencegah kebocoran data (*anti-data leakage*):
```bash
python split.py
```
*Hasil pembagian akan otomatis masuk ke folder `dataset/train/` dan `dataset/val/`.*

---

### 4. Melatih 3 Mode Transfer Learning (`train.py`)
Latih model ResNet-18 pada ketiga strategi transfer learning:
```bash
# Mode 1: Feature Extraction (bekukan backbone konvolusi, latih lapisan classifier)
python train.py feature

# Mode 2: Partial Fine-tuning (buka beberapa layer atas untuk adaptasi domain)
python train.py partial

# Mode 3: Train from Scratch (bobot diacak dari nol)
python train.py scratch
```

---

### 5. Membuat Laporan Grafik & Tabel Hasil (`report.py`)
```bash
python report.py
```
*Skrip ini akan membandingkan performa 3 mode dan menghasilkan:*
* Tabel akurasi validasi terbaik, waktu latih, dan epoch konvergensi.
* Grafik kurva pembelajaran di: `results/akurasi_per_epoch.png`.
* Dokumen ringkasan di: `results/hasil_3_mode.md`.

---

### 6. Benchmark Latensi Pipeline Robot (`latency.py`)
Ukur kecepatan komputasi model pada perangkat robot Anda:
```bash
# 1. Latensi model saja (ms/frame dan FPS)
python latency.py

# 2. Latensi pipeline penuh kamera (akuisisi + undistort + prep + inferensi)
python latency.py --cam 2 --ckpt models/resnet18_partial.pt
```

---

### 7. Uji Coba Klasifikasi Live Kamera (`realtime_classify.py`)
Uji model terbaik Anda secara langsung pada objek PBL:
```bash
python realtime_classify.py --cam 2 --ckpt models/resnet18_partial.pt
```
*Arahkan kamera ke objek PBL Anda untuk melihat label prediksi dan FPS live di layar!*
