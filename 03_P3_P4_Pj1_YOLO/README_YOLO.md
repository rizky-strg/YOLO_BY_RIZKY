# Panduan Tahap 3: Deteksi Objek YOLO PBL (P3, P4, Pj1)

Folder ini digunakan untuk melatih model **YOLO (Ultralytics)** menggunakan data objek PBL yang telah Anda kumpulkan pada Tahap 2 (`02_P2_Transfer_Learning`).

---

## 🛠️ Langkah Pelaksanaan Step-by-Step

### 1. Buat Dataset YOLO dari Data PBL Anda
Setelah Anda selesai mengambil foto objek PBL di Tahap 2, jalankan skrip berikut:
```bash
python 00_buat_dataset_yolo_dari_pbl.py
```
*Skrip ini otomatis:*
* Mendeteksi nama-nama folder objek PBL di `02_P2_Transfer_Learning/dataset_raw/`.
* Membuat file `classes.txt`.
* Menghasilkan anotasi bounding box untuk objek PBL dan label kosong untuk latar.
* Menyimpan hasilnya ke folder `dataset_yolo/`.

---

### 2. Validasi Kualitas Anotasi & Uji Konversi Format
Sesuai tuntutan kurikulum PPT Minggu 4 (Asesmen P3):
```bash
# 1. Cek koordinat & gambar kotak visual
python 01_cek_anotasi.py dataset_yolo

# 2. Uji konversi format 2 arah (YOLO <-> COCO)
python 02_yolo_ke_coco.py dataset_yolo dataset_yolo/annotations_coco.json
python 03_coco_ke_yolo.py dataset_yolo/annotations_coco.json hasil_balik
```

---

### 3. Membagi Dataset (Train, Val, Test)
Bagi dataset ke rasio standar 70% Train, 20% Val, 10% Test:
```bash
python 05_split_dataset_yolo.py dataset_yolo dataset_v1
```
*Hasilnya akan membuat struktur folder `dataset_v1/` dan file konfigurasi `dataset_v1/data.yaml`.*

---

### 4. Melatih 3 Eksperimen YOLO (Asesmen P4 & Pj1)
Jalankan benchmark komparasi 3 mode (Transfer Learning E1 vs Scratch E2 vs Tuning E3):
```bash
python 06_latih_bandingkan.py --nim <NIM_ANDA> --nama "<NAMA_ANDA>" --model yolov8n --epochs 30 --ubah imgsz=640
```
*Program ini otomatis:*
* Melatih E1 (Pretrained), E2 (Scratch), dan E3 (Tuning).
* Memilih model dengan `mAP50-95` terbaik di data Val.
* Menguji model terbaik **hanya sekali** pada data Test.
* Menghasilkan laporan resmi: `hasil_praktik_<NIM>.txt` dan `hasil_praktik_<NIM>.csv`.

---

### 5. Menjalankan Sistem Persepsi Real-time di Robot PBL
Jalankan inferensi live pada kamera robot:
```bash
python realtime_yolo_pbl.py --cam 2 --weights hasil_praktik/E3_tuning/weights/best.pt
```
*Otomatis mengoreksi distorsi lensa dengan `realtime_calibration.npz` dan mendeteksi objek PBL Anda secara real-time!*
