"""
Perbaiki bounding box seluruh dataset YOLO agar rapat dan presisi di sekeliling balok.
Memperbaiki 58% data yang sebelumnya terkena fallback kotak kebesaran meja.
"""
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(r"D:\Downloads\PBL_Vision_Project\03_P3_P4_Pj1_YOLO")
IMG_DIR = ROOT / "dataset_yolo" / "images"
LBL_DIR = ROOT / "dataset_yolo" / "labels"

model = YOLO("yolov8n.pt")

stats = {"yolo": 0, "contour": 0, "fallback": 0}
lbl_files = sorted(LBL_DIR.glob("*.txt"))
total = len(lbl_files)

print(f"Memproses {total} file label...")

for idx, lbl_path in enumerate(lbl_files, 1):
    content = lbl_path.read_text().strip()
    if not content:
        continue
    
    # Ambil class_id asli dari baris pertama
    class_id = int(content.splitlines()[0].split()[0])
    
    img_path = IMG_DIR / f"{lbl_path.stem}.png"
    if not img_path.exists():
        img_path = IMG_DIR / f"{lbl_path.stem}.jpg"
    if not img_path.exists():
        continue
        
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]
    
    # 1. Deteksi objek dengan YOLO (conf sangat rendah agar mendeteksi balok yang agak gelap)
    res = model.predict(str(img_path), conf=0.01, verbose=False)[0]
    best_box = None
    best_score = -1.0
    for b in res.boxes:
        cx, cy, bw, bh = [float(v) for v in b.xywhn[0]]
        # Balok berada di area tengah kerja meja
        if 0.15 < cx < 0.85 and 0.15 < cy < 0.85:
            # Ukuran balok w: 3%-35%, h: 5%-40%
            if 0.03 < bw < 0.35 and 0.05 < bh < 0.40:
                dist = ((cx - 0.5)**2 + (cy - 0.55)**2)**0.5
                score = float(b.conf[0]) * 3.0 - dist
                if score > best_score:
                    best_score = score
                    best_box = (cx, cy, bw, bh)
                    
    src = "yolo"
    if best_box is None:
        # 2. Coba kontur di crop tengah meja
        crop = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ch, cw = crop.shape[:2]
        for c in cnts:
            area = cv2.contourArea(c)
            if 400 < area < (cw * ch * 0.4):
                x, y, cbw, cbh = cv2.boundingRect(c)
                cx = (w*0.2 + x + cbw/2) / w
                cy = (h*0.2 + y + cbh/2) / h
                if 0.03 < cbw/w < 0.30 and 0.05 < cbh/h < 0.35:
                    best_box = (cx, cy, cbw/w, cbh/h)
                    src = "contour"
                    break
                    
    if best_box is None:
        # 3. Fallback proporsional pas ukuran balok (bukan kebesaran meja)
        best_box = (0.50, 0.55, 0.10, 0.16)
        src = "fallback"
        
    stats[src] += 1
    cx, cy, bw, bh = best_box
    lbl_path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
    
    if idx % 50 == 0 or idx == total:
        print(f"[{idx}/{total}] Selesai... (YOLO: {stats['yolo']}, Kontur: {stats['contour']}, Fallback: {stats['fallback']})")

print("\n" + "="*50)
print("SELESAI MEMPERBAIKI SELURUH BOUNDING BOX!")
print(f"Total diproses: {total}")
for k, v in stats.items():
    print(f"  - Sumber {k:10s}: {v} ({v*100/total:.1f}%)")
print("="*50)
