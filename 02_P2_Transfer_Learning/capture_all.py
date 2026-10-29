"""
Program All-in-One Akuisisi Citra Balok PBL (1 Kali Jalan untuk Semua 6 Balok).

Fitur:
- Sekali jalan untuk semua 6 balok (tidak perlu keluar-masuk program).
- Beralih antar balok cukup tekan tombol [TAB] / [N] atau angka [1] sampai [6].
- 100% Pencahayaan Alami Realtime Lingkungan (Sensor kamera murni).
- Bisa atur kecerahan langsung secara live dengan tombol [+] dan [-] jika ruangan agak gelap/terang.
- Tekan [C] untuk ganti kamera jika salah buka kamera.
- Tekan [SPACE] untuk mengambil foto.
- Tekan [Q] atau [ESC] jika sudah selesai semua.
"""
import argparse
import csv
import datetime
from pathlib import Path
import cv2
import numpy as np
from common import Undistorter

CLASSES = [
    "Balok-Hijau",
    "Balok-Hijau-Tua",
    "Balok-Hitam",
    "Balok-Merah",
    "Balok-Merah-Marun",
    "Balok-Putih"
]

def main():
    parser = argparse.ArgumentParser(description="All-in-One Capture Balok PBL")
    parser.add_argument("--cam", type=int, default=0, help="Index kamera (default: 0)")
    parser.add_argument("--root", default="dataset_raw", help="Folder tujuan dataset")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-undistort", action="store_true")
    args = parser.parse_args()

    und = Undistorter(not args.no_undistort)

    # 1. Buka kamera dengan fallback
    cap = None
    cams_to_try = [args.cam, 0, 1, 2, 3]
    actual_cam = args.cam
    for c_idx in cams_to_try:
        temp = cv2.VideoCapture(c_idx)
        if temp.isOpened():
            ret, _ = temp.read()
            if ret:
                cap = temp
                actual_cam = c_idx
                print(f"[KAMERA] Berhasil terhubung ke kamera index: {actual_cam}")
                break
            temp.release()

    if cap is None:
        raise RuntimeError("Kamera tidak dapat dibuka pada index mana pun.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    root_dir = Path(args.root)
    root_dir.mkdir(parents=True, exist_ok=True)
    meta_path = root_dir / "metadata.csv"
    new_meta = not meta_path.exists()

    tanggal = datetime.date.today().strftime("%Y%m%d")

    # Hitung foto yang sudah ada per kelas
    counts = {}
    for c in CLASSES:
        c_dir = root_dir / c
        c_dir.mkdir(parents=True, exist_ok=True)
        counts[c] = len(list(c_dir.glob("*.png")))

    current_idx = 0
    alpha = 1.00  # 100% murni alami sensor kamera
    beta = 0
    flash_timer = 0

    print("\n" + "="*65)
    print("  ALL-IN-ONE CAPTURE BALOK PBL (1 KALI RUNNING UNTUK SEMUA BALOK)")
    print("="*65)
    print("  KONTROL KEYBOARD:")
    print("    [SPACE]       : Ambil 1 foto balok aktif")
    print("    [TAB] / [N]   : Pindah ke Balok Berikutnya")
    print("    [P]           : Pindah ke Balok Sebelumnya")
    print("    [1] s.d. [6]  : Lompat langsung ke Balok 1-6")
    print("    [C]           : Ganti kamera live (0, 1, 2)")
    print("    [+] / [-]     : Naikkan / Turunkan kecerahan (live realtime)")
    print("    [0]           : Reset kecerahan ke 1.0x (Alami Ruangan)")
    print("    [Q] / [ESC]   : Selesai & Keluar")
    print("="*65 + "\n")

    with open(meta_path, "a", newline="", encoding="utf-8") as f_meta:
        writer = csv.writer(f_meta)
        if new_meta:
            writer.writerow(["nama_file", "kelas", "tanggal", "kondisi_cahaya"])

        while True:
            ok, raw_frame = cap.read()
            if not ok:
                print("Gagal membaca frame kamera.")
                break

            # Koreksi distorsi lensa P1
            frame = und(raw_frame)

            # Pengaturan kecerahan realtime jika pengguna ingin menyesuaikan
            if alpha != 1.00 or beta != 0:
                adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            else:
                adjusted_frame = frame

            view = adjusted_frame.copy()
            cur_cls = CLASSES[current_idx]
            cur_count = counts[cur_cls]

            # Efek flash putih sekejap saat memfoto
            if flash_timer > 0:
                view = cv2.addWeighted(view, 0.4, np.full_like(view, 255), 0.6, 0)
                flash_timer -= 1

            # Header info kelas & jumlah foto
            status_cahaya = "Alami" if (alpha == 1.0 and beta == 0) else f"{alpha:.2f}x"
            color_header = (0, 255, 0) if cur_count >= 30 else (0, 220, 255)
            
            cv2.rectangle(view, (0, 0), (args.width, 70), (20, 20, 20), -1)
            cv2.putText(view, f"BALOK [{current_idx+1}/6]: {cur_cls.upper()}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_header, 2)
            cv2.putText(view, f"Foto: {cur_count} (Target 30-35) | Cahaya: {status_cahaya} | Cam: {actual_cam}",
                        (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Footer petunjuk kontrol
            cv2.rectangle(view, (0, args.height - 35), (args.width, args.height), (20, 20, 20), -1)
            cv2.putText(view, "[SPACE] Foto | [TAB/N] Ganti Balok | [-/+] Cahaya | [C] Cam | [Q] Keluar",
                        (10, args.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Panel mini status 6 balok di samping kanan
            y_offset = 90
            for i, c_name in enumerate(CLASSES):
                c_cnt = counts[c_name]
                prefix = "-> " if i == current_idx else "   "
                color_txt = (0, 255, 0) if i == current_idx else (180, 180, 180)
                cv2.putText(view, f"{prefix}[{i+1}] {c_name[:12]}: {c_cnt}", (args.width - 200, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_txt, 1)
                y_offset += 20

            cv2.imshow("All-in-One Capture Balok PBL", view)
            k = cv2.waitKey(1) & 0xFF

            if k in (ord("q"), ord("Q"), 27):  # Q atau ESC
                break
            elif k == 32:  # SPACE (Ambil foto)
                counts[cur_cls] += 1
                cnt_num = counts[cur_cls]
                c_dir = root_dir / cur_cls
                # Simpan foto asli kamera
                name = f"{cur_cls}_{tanggal}_normal_{cnt_num:03d}.png"
                cv2.imwrite(str(c_dir / name), adjusted_frame)
                writer.writerow([name, cur_cls, tanggal, "normal"])
                f_meta.flush()
                flash_timer = 2
                print(f"[FOTO {cnt_num}] {name} berhasil disimpan!")
            elif k in (9, ord("n"), ord("N")):  # TAB atau N (Balok berikutnya)
                current_idx = (current_idx + 1) % len(CLASSES)
                print(f"\n[GANTI BALOK] Sekarang beralih ke: {CLASSES[current_idx]}")
            elif k in (ord("p"), ord("P")):  # P (Balok sebelumnya)
                current_idx = (current_idx - 1) % len(CLASSES)
                print(f"\n[GANTI BALOK] Sekarang beralih ke: {CLASSES[current_idx]}")
            elif ord("1") <= k <= ord("6"):  # Angka 1-6
                current_idx = k - ord("1")
                print(f"\n[GANTI BALOK] Langsung memilih: {CLASSES[current_idx]}")
            elif k in (ord("c"), ord("C")):  # Ganti kamera
                target_cam = (actual_cam + 1) % 4
                new_cap = cv2.VideoCapture(target_cam)
                if new_cap.isOpened():
                    ret_test, _ = new_cap.read()
                    if ret_test:
                        cap.release()
                        cap = new_cap
                        actual_cam = target_cam
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                        print(f"[KAMERA] Beralih ke kamera index: {actual_cam}")
                    else:
                        new_cap.release()
            elif k in (ord("+"), ord("="), ord("]")):  # Terangkan
                alpha = min(2.00, alpha + 0.05)
                beta = min(50, beta + 5)
                print(f"[CAHAYA] Kecerahan dinaikkan -> {alpha:.2f}x")
            elif k in (ord("-"), ord("_"), ord("[")):  # Gelapkan
                alpha = max(0.30, alpha - 0.05)
                beta = max(-50, beta - 5)
                print(f"[CAHAYA] Kecerahan diturunkan -> {alpha:.2f}x")
            elif k == ord("0"):  # Reset ke murni alami
                alpha = 1.00
                beta = 0
                print("[CAHAYA] Kecerahan di-reset ke 1.0x (Murni Alami Lingkungan)")

    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "="*60)
    print("  PENGAMBILAN DATA SELESAI!")
    print("  Rekap Foto per Balok:")
    for c_name in CLASSES:
        print(f"    - {c_name:18s}: {counts[c_name]} foto")
    print("="*60)

if __name__ == "__main__":
    main()
