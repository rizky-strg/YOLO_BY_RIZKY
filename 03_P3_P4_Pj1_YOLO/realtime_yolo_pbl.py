#!/usr/bin/env python3
"""
Integrasi P1 (Real-time Undistortion) dan P2 (YOLO Object Detection - Botol & Kaleng).

Alur:
1. Menangkap streaming video dari kamera eksternal.
2. Melakukan koreksi distorsi lensa secara real-time menggunakan parameter kalibrasi P1 (realtime_calibration.npz).
3. Melakukan deteksi objek (botol & kaleng) menggunakan model YOLOv8 yang dilatih dengan data P2.
4. Menampilkan visualisasi kotak deteksi, probabilitas keyakinan (confidence), dan FPS live.

Kontrol:
    S     : Simpan screenshot hasil deteksi
    Q/ESC : Keluar
"""
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Real-time YOLO Detection dengan Kalibrasi P1")
    parser.add_argument("--cam", type=int, default=2, help="Index kamera eksternal (default: 2)")
    parser.add_argument("--calib", default=r"D:\Downloads\CameraCalibration\CameraCalibration\realtime_calibration.npz",
                        help="Path ke file matriks kalibrasi P1 (.npz)")
    parser.add_argument("--weights", default="hasil_praktik/E1_pretrained/weights/best.pt",
                        help="Path ke bobot model YOLO hasil pelatihan")
    parser.add_argument("--conf", type=float, default=0.28, help="Confidence threshold (default: 0.28 untuk stabilitas balok putih)")
    parser.add_argument("--no-undistort", action="store_true", help="Nonaktifkan koreksi distorsi P1")
    args = parser.parse_args()

    # 1. Load parameter kalibrasi P1
    undistort_enabled = not args.no_undistort
    K, D = None, None
    calib_path = Path(args.calib)
    if not calib_path.exists() and Path("realtime_calibration.npz").exists():
        calib_path = Path("realtime_calibration.npz")
    if undistort_enabled:
        if calib_path.exists():
            data = np.load(str(calib_path))
            def pick(names):
                for n in names:
                    if n in data.files: return data[n]
                return None
            K = pick(["camera_matrix", "mtx", "K", "cameraMatrix"])
            D = pick(["dist_coeffs", "distortion_coefficients", "dist", "D", "distCoeffs"])
            if K is not None and D is not None:
                print(f"[P1] Berhasil memuat kalibrasi kamera dari: {calib_path}")
            else:
                print(f"[PERINGATAN] Format key kalibrasi tidak cocok. Berjalan tanpa undistort.")
                undistort_enabled = False
        else:
            print(f"[PERINGATAN] File kalibrasi {calib_path} tidak ditemukan. Berjalan tanpa undistortion.")
            undistort_enabled = False

    # 2. Load model YOLO
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[INFO] Bobot {weights_path} belum ditemukan, mencoba E1_pretrained...")
        weights_path = Path("hasil_praktik/E1_pretrained/weights/best.pt")
        if not weights_path.exists():
            weights_path = "yolov8n.pt"
    
    print(f"[YOLO] Memuat model deteksi dari: {weights_path}")
    model = YOLO(str(weights_path))

    # 3. Buka kamera eksternal (coba index yang diberikan atau fallback otomatis)
    cap = None
    cams_to_try = [args.cam, 0, 1, 2, 3]
    for c_idx in cams_to_try:
        temp_cap = cv2.VideoCapture(c_idx)
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()
            if ret:
                cap = temp_cap
                actual_cam = c_idx
                print(f"[KAMERA] Berhasil terhubung ke kamera index: {actual_cam}")
                break
            temp_cap.release()

    if cap is None:
        raise RuntimeError(f"Tidak dapat membuka kamera mana pun dari index {cams_to_try}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    out_dir = Path("snapshots_yolo")
    out_dir.mkdir(exist_ok=True)

    fps = 0.0
    last_time = time.perf_counter()
    current_conf = float(args.conf)
    status_und = "Undistort: ON (P1)" if undistort_enabled else "Undistort: OFF"
    model_e1_path = Path("hasil_praktik/E1_pretrained/weights/best.pt")
    model_e3_path = Path("hasil_praktik/E3_tuning/weights/best.pt")
    current_model_name = "E1_pretrained (Rekomendasi Balok Putih)" if "E1" in str(weights_path) else "E3_tuning"

    print("\n=======================================================")
    print("  DETEKSI REAL-TIME YOLO (P1 + P2)")
    print("=======================================================")
    print("  Kamera          :", actual_cam)
    print("  Undistort (P1)  :", "AKTIF" if undistort_enabled else "NONAKTIF")
    print("  Model Aktif     :", current_model_name)
    print("  Confidence Thr  :", current_conf)
    print("  KONTROL KEYBOARD:")
    print("    [+] / [-]     : Naik/Turunkan Confidence (Sensitivitas)")
    print("    [M]           : Beralih Model (E1_pretrained <-> E3_tuning)")
    print("    [C]           : Ganti Index Kamera")
    print("    [S]           : Simpan Snapshot Hasil")
    print("    [Q] / [ESC]   : Keluar")
    print("=======================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame kamera.")
            break

        # [Pipeline P1] Koreksi distorsi lensa
        if undistort_enabled and K is not None and D is not None:
            processed_frame = cv2.undistort(frame, K, D)
        else:
            processed_frame = frame

        # [Pipeline YOLO] Inferensi deteksi objek
        results = model.predict(processed_frame, conf=current_conf, imgsz=416 if "E1" in current_model_name else 640, verbose=False)
        annotated_frame = results[0].plot()

        # Hitung FPS real-time
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(1e-5, now - last_time)
        last_time = now

        # Overlay status di pojok atas (Desain Bersih & Informatif)
        cv2.rectangle(annotated_frame, (10, 10), (460, 95), (20, 20, 20), -1)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f} | Cam: {actual_cam} | Conf: {current_conf:.2f}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Model: {current_model_name.split()[0]} | {status_und}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 100), 1)
        cv2.putText(annotated_frame, "[+/-] Sensitivitas | [M] Ganti Model | [S] Simpan",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        cv2.imshow("Real-time YOLO Detection (P1 + P2)", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # Q atau ESC
            break
        elif key in (ord('+'), ord('='), ord(']')):  # Naikkan confidence
            current_conf = min(0.95, current_conf + 0.03)
            print(f"[SET] Confidence dinaikkan ke: {current_conf:.2f}")
        elif key in (ord('-'), ord('_'), ord('[')):  # Turunkan confidence (lebih peka)
            current_conf = max(0.15, current_conf - 0.03)
            print(f"[SET] Confidence diturunkan ke: {current_conf:.2f} (Lebih peka)")
        elif key in (ord('m'), ord('M')):  # Switch model E1 <-> E3
            if "E1" in current_model_name and model_e3_path.exists():
                model = YOLO(str(model_e3_path))
                current_model_name = "E3_tuning (imgsz=640)"
                print(f"[MODEL] Beralih ke: {current_model_name}")
            elif model_e1_path.exists():
                model = YOLO(str(model_e1_path))
                current_model_name = "E1_pretrained (Rekomendasi Balok Putih)"
                print(f"[MODEL] Beralih ke: {current_model_name}")
        elif key in (ord('s'), ord('S')):  # S untuk simpan snapshot
            ts = time.strftime("%Y%m%d_%H%M%S")
            snap_path = out_dir / f"yolo_deteksi_{ts}.jpg"
            cv2.imwrite(str(snap_path), annotated_frame)
            print(f"[SNAPSHOT] Tersimpan ke: {snap_path}")
        elif key in (ord('c'), ord('C'), ord('0'), ord('1'), ord('2'), ord('3')):
            # Ganti kamera secara interaktif
            target_cam = (actual_cam + 1) % 4 if key in (ord('c'), ord('C')) else int(chr(key))
            print(f"\n[GANTI KAMERA] Mencoba beralih ke kamera index {target_cam}...")
            new_cap = cv2.VideoCapture(target_cam)
            if new_cap.isOpened():
                ret_test, _ = new_cap.read()
                if ret_test:
                    cap.release()
                    cap = new_cap
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    actual_cam = target_cam
                    print(f"[SUKSES] Kamera beralih ke index: {actual_cam}\n")
                else:
                    new_cap.release()
                    print(f"[GAGAL] Kamera index {target_cam} tidak menghasilkan frame.")
            else:
                print(f"[GAGAL] Tidak dapat membuka kamera index {target_cam}.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
