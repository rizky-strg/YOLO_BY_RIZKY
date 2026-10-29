"""
Mengonversi dataset objek PBL dari Tahap 2 (02_P2_Transfer_Learning/dataset_raw)
menjadi format YOLO siap latih untuk P3, P4, dan Pj1.
"""
import shutil
from pathlib import Path
import cv2
from ultralytics import YOLO

SRC_DIR = Path(r"D:\Downloads\PBL_Vision_Project\02_P2_Transfer_Learning\dataset_raw")
OUT_DIR = Path(r"D:\Downloads\PBL_Vision_Project\03_P3_P4_Pj1_YOLO\dataset_yolo")

OUT_IMAGES = OUT_DIR / "images"
OUT_LABELS = OUT_DIR / "labels"

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)

# 1. Deteksi otomatis kelas objek PBL dari folder di dataset_raw
subdirs = [d.name for d in sorted(SRC_DIR.iterdir()) if d.is_dir() and not d.name.startswith(".")]
object_classes = [c for c in subdirs if c.lower() != "latar"]

if not object_classes:
    raise SystemExit(
        f"[PERINGATAN] Belum ada folder objek di {SRC_DIR}.\n"
        "Silakan ambil data objek PBL terlebih dahulu menggunakan capture.py di folder 02_P2_Transfer_Learning!"
    )

print(f"Kelas objek PBL terdeteksi: {object_classes}")
(OUT_DIR / "classes.txt").write_text("\n".join(object_classes) + "\n", encoding="utf-8")

# 2. Model dasar untuk auto-annotation awal
model = YOLO("yolov8n.pt")

total_saved = 0
stats = {c: 0 for c in object_classes}
stats["latar"] = 0

for cls_idx, cls_name in enumerate(object_classes):
    img_files = sorted((SRC_DIR / cls_name).glob("*.png")) + sorted((SRC_DIR / cls_name).glob("*.jpg"))
    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_px, w_px = img.shape[:2]
        
        # Prediksi awal dengan YOLOv8 (conf rendah agar menangkap objek balok)
        res = model.predict(str(img_path), conf=0.01, verbose=False)[0]
        
        best_box = None
        best_score = -1.0
        
        for b in res.boxes:
            conf = float(b.conf[0])
            cx, cy, w, h = [float(v) for v in b.xywhn[0]]
            area = w * h
            if 0.15 < cx < 0.85 and 0.15 < cy < 0.85:
                if 0.03 < w < 0.40 and 0.05 < h < 0.45:
                    dist_to_center = ((cx - 0.5) ** 2 + (cy - 0.55) ** 2) ** 0.5
                    score = conf * 3.0 - dist_to_center
                    if score > best_score:
                        best_score = score
                        best_box = (cx, cy, w, h)
        
        # 2. Fallback kontur jika YOLO belum menangkap
        if best_box is None:
            crop = img[int(h_px*0.15):int(h_px*0.85), int(w_px*0.15):int(w_px*0.85)]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ch, cw = crop.shape[:2]
            for c in cnts:
                c_area = cv2.contourArea(c)
                if 400 < c_area < (cw * ch * 0.45):
                    bx, by, cbw, cbh = cv2.boundingRect(c)
                    cx = (w_px * 0.15 + bx + cbw / 2) / w_px
                    cy = (h_px * 0.15 + by + cbh / 2) / h_px
                    if 0.03 < cbw / w_px < 0.40 and 0.05 < cbh / h_px < 0.45:
                        best_box = (cx, cy, cbw / w_px, cbh / h_px)
                        break
                        
        if best_box is None:
            best_box = (0.50, 0.55, 0.12, 0.18)
            
        cx, cy, w, h = best_box
        cx = max(0.01, min(0.99, cx))
        cy = max(0.01, min(0.99, cy))
        w = max(0.05, min(0.95, w))
        h = max(0.05, min(0.95, h))
        
        # Salin gambar & simpan label
        shutil.copy(img_path, OUT_IMAGES / img_path.name)
        label_text = f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
        (OUT_LABELS / f"{img_path.stem}.txt").write_text(label_text, encoding="utf-8")
        
        stats[cls_name] += 1
        total_saved += 1

# 3. Masukkan gambar latar sebagai negative samples
latar_dir = SRC_DIR / "latar"
if latar_dir.exists():
    latar_files = sorted(latar_dir.glob("*.png")) + sorted(latar_dir.glob("*.jpg"))
    for img_path in latar_files[:40]:
        shutil.copy(img_path, OUT_IMAGES / img_path.name)
        (OUT_LABELS / f"{img_path.stem}.txt").write_text("", encoding="utf-8")
        stats["latar"] += 1
        total_saved += 1

print("\n" + "="*50)
print(f"Dataset YOLO PBL berhasil dibuat di: {OUT_DIR}")
print(f"Total gambar tersimpan : {total_saved}")
for k, v in stats.items():
    print(f"  - {k:15s}: {v}")
print("="*50)
print("Lanjutkan dengan: python 01_cek_anotasi.py dataset_yolo")
