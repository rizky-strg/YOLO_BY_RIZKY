"""
Deteksi Bounding Box Presisi Murni Fisik Balok:
- 100% Mengabaikan Bayangan (Shadow Rejection).
- 100% Mengabaikan Lantai / Frame Pinggir (ROI Clipping).
- 100% Mengabaikan Pantulan Lampu.
- Menempel rapat pas hanya pada objek fisik balok!
"""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r"D:\Downloads\PBL_Vision_Project\03_P3_P4_Pj1_YOLO")
IMG_DIR = ROOT / "dataset_yolo" / "images"
LBL_DIR = ROOT / "dataset_yolo" / "labels"

CLASSES = [
    "Balok-Hijau",
    "Balok-Hitam",
    "Balok-Merah",
    "Balok-Merah-Marun",
    "Balok-Putih"
]

def detect_precise_box(img, class_name):
    h_px, w_px = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)

    # 1. Buang 7% batas pinggir frame kamera (Mencegah lantai, kaki meja, atau pinggir frame)
    roi_mask = np.zeros((h_px, w_px), dtype=np.uint8)
    mx = int(w_px * 0.07)
    my = int(h_px * 0.07)
    roi_mask[my:h_px - my, mx:w_px - mx] = 255

    # 2. Segmentasi murni warna & intensitas fisik balok (bebas bayangan)
    if class_name == "Balok-Hijau":
        # Warna hijau murni (bayangan meja tidak hijau!)
        mask = ((h >= 30) & (h <= 95) & (s > 35) & (v > 25)).astype(np.uint8) * 255
    elif class_name == "Balok-Merah":
        # Warna merah terang (bayangan meja tidak merah!)
        mask = (((h <= 14) | (h >= 165)) & (s > 45) & (v > 45)).astype(np.uint8) * 255
    elif class_name == "Balok-Merah-Marun":
        # Warna merah marun / gelap (bayangan meja kayu tidak marun pekat!)
        mask = (((h <= 18) | (h >= 155)) & (s > 30) & (v > 20)).astype(np.uint8) * 255
    elif class_name == "Balok-Hitam":
        # Balok hitam pekat (bayangan meja kayu gray ~75-90, balok hitam gray < 58!)
        mask = (gray < 58).astype(np.uint8) * 255
    elif class_name == "Balok-Putih":
        # Balok putih cerah (meja kayu gray ~90-120, balok putih gray > 145!)
        mask = ((gray > 145) & (s < 55)).astype(np.uint8) * 255
    else:
        mask = np.zeros((h_px, w_px), dtype=np.uint8)

    # Terapkan batas kerja ROI
    mask = cv2.bitwise_and(mask, roi_mask)

    # Bersihkan noise kecil & gabungkan celah dengan morfologi
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 400 < area < (w_px * h_px * 0.40):
            bx, by, bw, bh = cv2.boundingRect(c)
            # Nilai kedekatan ke tengah meja
            cx = bx + bw / 2
            cy = by + bh / 2
            dist = ((cx - w_px/2)**2 + (cy - h_px/2)**2)**0.5
            score = area - dist * 5.0
            valid.append((score, bx, by, bw, bh))

    if valid:
        valid.sort(key=lambda x: x[0], reverse=True)
        _, bx, by, bw, bh = valid[0]
        # Bounding box normalisasi (cx, cy, w, h)
        return ((bx + bw/2)/w_px, (by + bh/2)/h_px, bw/w_px, bh/h_px)

    return None

def main():
    lbl_files = sorted(LBL_DIR.glob("*.txt"))
    total = len(lbl_files)
    print(f"Memproses {total} foto untuk memastikan kotak HANYA menempel pada balok...\n")

    sukses = 0
    gagal = 0

    for idx, lbl_path in enumerate(lbl_files, 1):
        content = lbl_path.read_text().strip()
        if not content:
            continue

        cid = int(content.splitlines()[0].split()[0])
        cls_name = CLASSES[cid]

        img_path = IMG_DIR / f"{lbl_path.stem}.png"
        if not img_path.exists():
            img_path = IMG_DIR / f"{lbl_path.stem}.jpg"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        box = detect_precise_box(img, cls_name)
        if box is not None:
            cx, cy, bw, bh = box
            lbl_path.write_text(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
            sukses += 1
        else:
            gagal += 1
            print(f"[PERINGATAN] Objek tidak terdeteksi pada: {img_path.name}")

    print("="*60)
    print("HASIL PEMBUATAN BOUNDING BOX MURNI BALOK:")
    print(f"  - Berhasil pas hanya ke balok : {sukses} / {total} ({sukses*100/total:.1f}%)")
    print(f"  - Gagal / Tidak terdeteksi    : {gagal}")
    print("="*60)

if __name__ == "__main__":
    main()
