"""Langkah 1: ambil citra terkoreksi (undistort) dari kamera eksternal dengan simulasi pencahayaan.
Contoh:  python capture.py objek-A terang --cam 1
         python capture.py objek-A redup  --cam 1
KONTROL:
    SPACE : Simpan 1 foto
    - / [ : Gelapkan kamera (simulasi redup/bayangan)
    + / ] : Terangkan kamera (simulasi terang/silau)
    1     : Preset Terang Normal (1.0x)
    2     : Preset Redup (0.55x)
    3     : Preset Bayangan/Gelap (0.35x)
    4     : Preset Silau/Jendela (1.35x)
    Q/ESC : Keluar
"""
import argparse
import csv
import datetime
import sys
from pathlib import Path
import cv2
import numpy as np
from common import open_camera, Undistorter

p = argparse.ArgumentParser()
p.add_argument("kelas")
p.add_argument("kondisi", help="mis. terang, redup, jendela, bayangan")
p.add_argument("--cam", type=int, default=1)
p.add_argument("--width", type=int, default=640)
p.add_argument("--height", type=int, default=480)
p.add_argument("--root", default="dataset_raw")
p.add_argument("--no-undistort", action="store_true")
a = p.parse_args()

if "_" in a.kelas or "_" in a.kondisi:
    sys.exit("Jangan pakai '_' di kelas/kondisi, gunakan tanda hubung '-'.")

und = Undistorter(not a.no_undistort)

# Kecerahan default: 100% murni warna sensor kamera asli (alpha=1.00, beta=0)
# Tidak ada penggelapan digital otomatis agar warna asli balok terjaga tajam
alpha, beta = 1.00, 0

# Buka kamera dengan fallback
cap = None
for c_idx in [a.cam, 0, 1, 2, 3]:
    temp = cv2.VideoCapture(c_idx)
    if temp.isOpened():
        ret_ok, _ = temp.read()
        if ret_ok:
            cap = temp
            a.cam = c_idx
            print(f"[KAMERA] Berhasil membuka kamera pada index: {a.cam}")
            break
        temp.release()

if cap is None:
    raise RuntimeError("Kamera tidak dapat dibuka pada index mana pun.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)

out_dir = Path(a.root) / a.kelas
out_dir.mkdir(parents=True, exist_ok=True)
meta = Path(a.root) / "metadata.csv"
new_meta = not meta.exists()
tanggal = datetime.date.today().strftime("%Y%m%d")
n = len(list(out_dir.glob(f"{a.kelas}_{tanggal}_{a.kondisi}_*.png")))

print("\n" + "="*60)
print(f"  AKUISISI CITRA OBJEK: {a.kelas.upper()} | KONDISI: {a.kondisi.upper()}")
print("="*60)
print(f"  Kamera          : Index {a.cam}")
print(f"  Kecerahan Awal  : {alpha:.2f}x (Otomatis disesuaikan untuk '{a.kondisi}')")
print("  Kontrol Keyboard:")
print("    [SPACE]       : Simpan foto")
print("    [-] atau [[]  : Gelapkan gambar kamera")
print("    [+] atau []]  : Terangkan gambar kamera")
print("    [1, 2, 3, 4]  : Preset (1=Terang, 2=Redup, 3=Gelap, 4=Silau)")
print("    [Q] / [ESC]   : Selesai / Keluar")
print("="*60 + "\n")

with open(meta, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if new_meta:
        w.writerow(["nama_file", "kelas", "tanggal", "kondisi_cahaya"])

    while True:
        ok, raw_frame = cap.read()
        if not ok:
            print("Gagal membaca frame kamera.")
            break

        # 1. Koreksi distorsi P1
        frame = und(raw_frame)

        # 2. Terapkan simulasi pencahayaan (brightness & contrast scaling)
        if alpha != 1.0 or beta != 0:
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        view = frame.copy()

        # Overlay informasi pada layar preview
        cv2.putText(view, f"{a.kelas} | {a.kondisi} | Kecerahan: {alpha:.2f}x | Foto: {n}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(view, "[SPACE] Foto | [-/+] Atur Gelap/Terang | [Q] Keluar",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Capture Objek PBL", view)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord("q"), 27):  # Q atau ESC
            break
        elif k in (ord("c"), ord("C")):  # C untuk ganti kamera
            target_cam = (a.cam + 1) % 4
            new_cap = cv2.VideoCapture(target_cam)
            if new_cap.isOpened():
                ret_ok, _ = new_cap.read()
                if ret_ok:
                    cap.release()
                    cap = new_cap
                    a.cam = target_cam
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
                    print(f"[KAMERA] Beralih ke kamera index: {a.cam}")
                else:
                    new_cap.release()
            else:
                print(f"[KAMERA] Kamera index {target_cam} tidak tersedia.")
        elif k == 32:  # SPACE
            n += 1
            name = f"{a.kelas}_{tanggal}_{a.kondisi}_{n:03d}.png"
            cv2.imwrite(str(out_dir / name), frame)
            w.writerow([name, a.kelas, tanggal, a.kondisi])
            f.flush()
            print(f"[SIMPAN] {name} (Kecerahan: {alpha:.2f}x)")
        elif k in (ord("-"), ord("[")):  # Gelapkan
            alpha = max(0.20, alpha - 0.10)
            beta = max(-60, beta - 10)
            print(f"[KONTROL] Gambar digelapkan -> Kecerahan: {alpha:.2f}x")
        elif k in (ord("+"), ord("="), ord("]")):  # Terangkan
            alpha = min(2.00, alpha + 0.10)
            beta = min(60, beta + 10)
            print(f"[KONTROL] Gambar diterangkan -> Kecerahan: {alpha:.2f}x")
        elif k == ord("1"):  # Preset Terang
            alpha, beta = 1.00, 0
            print("[PRESET] Mode 1: Terang Normal (1.0x)")
        elif k == ord("2"):  # Preset Redup
            alpha, beta = 0.55, -20
            print("[PRESET] Mode 2: Redup (0.55x)")
        elif k == ord("3"):  # Preset Bayangan/Gelap
            alpha, beta = 0.35, -35
            print("[PRESET] Mode 3: Bayangan / Gelap (0.35x)")
        elif k == ord("4"):  # Preset Silau
            alpha, beta = 1.35, 25
            print("[PRESET] Mode 4: Silau / Cahaya Jendela (1.35x)")

cap.release()
cv2.destroyAllWindows()
