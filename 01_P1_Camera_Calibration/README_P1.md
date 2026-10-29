# Panduan Tahap 1: Kalibrasi Kamera Robot PBL (P1)

Folder ini digunakan untuk mengestimasi matriks kamera intrinsik ($K$) dan koefisien distorsi lensa ($D$) agar sudut pandang kamera robot lurus dan bebas distorsi lengkung (*barrel distortion*).

---

## 📌 Kapan Tahap Ini Perlu Dijalankan?
* **Jika kamera yang digunakan SAMA dengan webcam kemarin**:  
  $\rightarrow$ **Tidak perlu diulang**. File `realtime_calibration.npz` di folder ini sudah siap pakai dan sudah disalin ke folder Tahap 2 dan Tahap 3.
* **Jika menggunakan kamera baru / kamera robot berbeda**:  
  $\rightarrow$ Ikuti langkah di bawah ini untuk kalibrasi ulang.

---

## 🛠️ Langkah Kalibrasi Kamera Baru:

1. **Cek index kamera:**
   ```bash
   python check_cameras.py
   ```
2. **Jalankan kalibrasi interaktif papan catur (Checkerboard):**
   ```bash
   python realtimeCameraCalibration.py
   ```
   * **Kontrol:**
     * Arahkan papan catur di depan kamera robot pada berbagai sudut dan jarak.
     * Tekan **`SPACE`** saat sudut terdeteksi untuk mengambil frame ($\ge 15-30$ frame).
     * Tekan **`C`** untuk memproses kalkulasi kalibrasi.
     * Tekan **`S`** untuk menyimpan file `realtime_calibration.npz`.
     * Tekan **`Q`** untuk keluar.
3. **Uji hasil koreksi lensa live (Side-by-side):**
   ```bash
   python save_realtime_result.py
   ```
