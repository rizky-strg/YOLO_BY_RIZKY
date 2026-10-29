"""
Tool Labeling Kilat PBL - Mode Presisi 4 Sudut Balok (4 Corners).

Fitur Utama:
- Klik 4 Sudut Balok:
    1. Klik Sudut 1
    2. Klik Sudut 2
    3. Klik Sudut 3
    4. Klik Sudut 4 -> Balok langsung terkotakkan sempurna sesuai orientasi & kemiringannya!
- Kotak & Poligon otomatis dihitung dan disimpan ke format standar YOLO (cx cy w h).
- Garis Pandu Crosshair mengikuti kursor mouse untuk membidik sudut balok secara akurat.
- Klik Kanan / Z : Batalkan titik sudut terakhir (Undo).
- R              : Reset semua sudut pada foto saat ini.
- SPACE / ENTER  : Simpan & lanjut ke foto berikutnya.
- D / DEL        : Hapus foto ini (jika ada tangan/sepatu/sandal).
- B              : Kembali ke foto sebelumnya.
- Q / ESC        : Selesai / Keluar (Semua progres tersimpan aman).
"""
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r"D:\Downloads\PBL_Vision_Project\03_P3_P4_Pj1_YOLO")
IMG_DIR = ROOT / "dataset_yolo" / "images"
LBL_DIR = ROOT / "dataset_yolo" / "labels"
CLASSES_FILE = ROOT / "dataset_yolo" / "classes.txt"

if not CLASSES_FILE.exists():
    CLASSES = ["Balok-Hijau", "Balok-Hitam", "Balok-Merah", "Balok-Merah-Marun", "Balok-Putih"]
    CLASSES_FILE.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
else:
    CLASSES = [c.strip() for c in CLASSES_FILE.read_text(encoding="utf-8").splitlines() if c.strip()]

class_to_id = {c: i for i, c in enumerate(CLASSES)}

# State mouse & 4 sudut
pts = []
mouse_pos = (0, 0)
current_box = None  # (x1, y1, x2, y2)
current_polygon = None  # list of 4 points

def mouse_callback(event, x, y, flags, param):
    global pts, mouse_pos, current_box, current_polygon
    mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) < 4:
            pts.append((x, y))
            if len(pts) == 4:
                # 4 sudut selesai! Hitung bounding box luar
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                current_box = (min(xs), min(ys), max(xs), max(ys))
                current_polygon = list(pts)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Klik kanan: Undo titik terakhir
        if pts:
            pts.pop()
            current_box = None
            current_polygon = None

def get_class_from_name(filename):
    for c in CLASSES:
        if filename.startswith(c):
            return c, class_to_id[c]
    return CLASSES[0], 0

def load_existing_box(lbl_path, w, h):
    if lbl_path.exists():
        content = lbl_path.read_text().strip()
        if content:
            parts = content.splitlines()[0].split()
            if len(parts) == 5:
                cid, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                return cid, (x1, y1, x2, y2)
    return None, None

def save_box(lbl_path, cid, box, w, h):
    x1, y1, x2, y2 = box
    bw = max(2, x2 - x1)
    bh = max(2, y2 - y1)
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    nw = bw / float(w)
    nh = bh / float(h)
    lbl_path.write_text(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n", encoding="utf-8")

def main():
    global pts, current_box, current_polygon
    images = sorted(list(IMG_DIR.glob("*.png")) + list(IMG_DIR.glob("*.jpg")))
    total = len(images)
    if total == 0:
        print("[INFO] Tidak ada foto di dataset_yolo/images!")
        return

    win_title = "Labeling Kilat PBL (Mode 4 Sudut Balok)"
    cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_title, mouse_callback)

    STATE_FILE = ROOT / "dataset_yolo" / ".label_state.txt"
    idx = 0
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        for i, p in enumerate(images):
            if arg in p.name.lower():
                idx = i
                break
    elif STATE_FILE.exists():
        saved_name = STATE_FILE.read_text(encoding="utf-8").strip()
        for i, p in enumerate(images):
            if p.name == saved_name:
                idx = i
                break

    print("\n" + "="*70)
    print("  TOOL LABELING KILAT BALOK PBL (MODE 4 SUDUT BALOK)")
    print("="*70)
    print(f"  Melanjutkan dari foto #{idx+1}/{len(images)}: {images[idx].name if images else ''}")
    print("  CARA KOTAKKAN:")
    print("    Klik 4 SUDUT BALOK secara berurutan:")
    print("      Sudut 1 -> Sudut 2 -> Sudut 3 -> Sudut 4")
    print("    Seketika kotak presisi langsung membungkus kemiringan balok!")
    print("  KONTROL KEYBOARD:")
    print("    [SPACE] / [ENTER]: Simpan kotak & lanjut foto berikutnya.")
    print("    [1] s/d [5]      : Langsung LOMPAT KELAS (1:Hijau, 2:Hitam, 3:Merah, 4:Marun, 5:Putih)")
    print("    [N]              : Lewati ke foto berikutnya.")
    print("    [D] / [DEL]      : HAPUS FOTO (jika ada tangan/sandal/rusak).")
    print("    [B]              : Kembali ke foto sebelumnya.")
    print("    [Klik Kanan / Z] : Batalkan sudut terakhir (Undo).")
    print("    [R]              : Reset semua sudut foto ini.")
    print("    [Q] / [ESC]      : Selesai / Keluar (Semua tersimpan aman).")
    print("="*70 + "\n")

    while 0 <= idx < len(images):
        img_path = images[idx]
        lbl_path = LBL_DIR / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue

        h, w = img.shape[:2]
        cname, cid = get_class_from_name(img_path.name)

        # Muat kotak yang sudah ada (jika ada)
        ex_cid, ex_box = load_existing_box(lbl_path, w, h)
        if ex_cid is not None:
            cid = ex_cid
            cname = CLASSES[cid] if cid < len(CLASSES) else cname

        current_box = ex_box
        current_polygon = None
        pts = []

        while True:
            display = img.copy()
            mx, my = mouse_pos

            # 1. Garis bidik Crosshair tipis mengikuti kursor mouse (Laser Guide)
            cv2.line(display, (0, my), (w, my), (100, 100, 100), 1, cv2.LINE_AA)
            cv2.line(display, (mx, 0), (mx, h), (100, 100, 100), 1, cv2.LINE_AA)

            # 2. Gambar titik-titik sudut yang sudah diklik
            for i, p in enumerate(pts):
                cv2.circle(display, p, 5, (0, 0, 255), -1)
                cv2.circle(display, p, 7, (255, 255, 255), 1)
                cv2.putText(display, f"S{i+1}", (p[0] + 6, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            # Gambar garis penghubung antar sudut
            if len(pts) > 1:
                for i in range(len(pts) - 1):
                    cv2.line(display, pts[i], pts[i+1], (0, 220, 255), 2, cv2.LINE_AA)

            # Garis sementara ke posisi mouse saat ini jika belum 4 sudut
            if 0 < len(pts) < 4:
                cv2.line(display, pts[-1], (mx, my), (0, 160, 255), 1, cv2.LINE_AA)

            # Jika 4 sudut sudah lengkap: gambar poligon tertutup & bounding box
            if len(pts) == 4 and current_polygon is not None:
                poly_arr = np.array(current_polygon, np.int32).reshape((-1, 1, 2))
                # Poligon fisik balok (Cyan)
                cv2.polylines(display, [poly_arr], True, (255, 255, 0), 2, cv2.LINE_AA)
                
                # Bounding box YOLO luar (Hijau)
                if current_box is not None:
                    x1, y1, x2, y2 = current_box
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    tag = f"{cname} [4 Sudut Selesai]"
                    cv2.rectangle(display, (x1, max(0, y1 - 22)), (x1 + len(tag)*9, max(22, y1)), (0, 255, 0), -1)
                    cv2.putText(display, tag, (x1 + 3, max(16, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            # Jika memakai kotak yang sudah ada (sebelum diklik ulang)
            elif current_box is not None and len(pts) == 0:
                x1, y1, x2, y2 = current_box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                tag = f"{cname} [Kotak Ada - Klik 4 Sudut untuk Ubah]"
                cv2.rectangle(display, (x1, max(0, y1 - 22)), (x1 + len(tag)*8, max(22, y1)), (0, 255, 0), -1)
                cv2.putText(display, tag, (x1 + 3, max(16, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

            # Header Info
            cv2.rectangle(display, (0, 0), (w, 65), (25, 25, 25), -1)
            cv2.putText(display, f"[{idx+1}/{len(images)}] {cname.upper()} ({img_path.name})",
                        (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(display, f"Sudut: {len(pts)}/4 | [1]Hijau [2]Hitam [3]Merah [4]Marun [5]Putih",
                        (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
            panduan = f"Klik Sudut {len(pts)+1}" if len(pts) < 4 else "[SPACE] Simpan 4 Sudut"
            cv2.putText(display, f"{panduan} | [D] HAPUS FOTO Tangan/Sandal | [Z] Undo | [B] Balik | [Q] Selesai",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            cv2.imshow(win_title, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('q'), ord('Q'), 27):  # Q atau ESC (Keluar)
                cv2.destroyAllWindows()
                print("\n[INFO] Labeling dihentikan. Semua data yang sudah diproses tersimpan aman!")
                return
            elif ord('1') <= key <= ord('5'):
                # Lompat langsung ke kelas terpilih!
                target_cls = CLASSES[key - ord('1')]
                for target_i, p in enumerate(images):
                    if p.name.startswith(target_cls):
                        idx = target_i
                        STATE_FILE.write_text(images[idx].name, encoding="utf-8")
                        print(f"\n[LOMPAT] Langsung ke kelas: {target_cls} (Foto #{idx+1})")
                        break
                break
            elif key in (32, 13):  # SPACE atau ENTER (Lanjut)
                if current_box is not None:
                    x1, y1, x2, y2 = current_box
                    if (x2 - x1) > 5 and (y2 - y1) > 5:
                        save_box(lbl_path, cid, current_box, w, h)
                        # Otomatis simpan preview visual ke cek_visual
                        cek_v_dir = ROOT / "dataset_yolo" / "cek_visual"
                        cek_v_dir.mkdir(exist_ok=True)
                        cv2.imwrite(str(cek_v_dir / img_path.name), display)
                        print(f"[{idx+1}/{len(images)}] OK -> {img_path.name}")
                idx += 1
                if idx < len(images):
                    STATE_FILE.write_text(images[idx].name, encoding="utf-8")
                break
            elif key in (ord('n'), ord('N')):  # N (Lewati ke foto berikutnya tanpa ubah)
                idx += 1
                if idx < len(images):
                    STATE_FILE.write_text(images[idx].name, encoding="utf-8")
                break
            elif key in (ord('d'), ord('D'), ord('x'), ord('X'), 8):  # D atau DELETE (Hapus foto)
                print(f"[HAPUS] Menghapus foto: {img_path.name}")
                if img_path.exists():
                    img_path.unlink()
                if lbl_path.exists():
                    lbl_path.unlink()
                cek_v = ROOT / "dataset_yolo" / "cek_visual" / img_path.name
                if cek_v.exists():
                    cek_v.unlink()
                images.pop(idx)
                if idx < len(images):
                    STATE_FILE.write_text(images[idx].name, encoding="utf-8")
                break
            elif key in (ord('b'), ord('B')):  # B (Kembali)
                idx = max(0, idx - 1)
                if idx < len(images):
                    STATE_FILE.write_text(images[idx].name, encoding="utf-8")
                break
            elif key in (ord('z'), ord('Z')):  # Z (Undo sudut terakhir)
                if pts:
                    pts.pop()
                    current_box = None
                    current_polygon = None
            elif key in (ord('r'), ord('R')):  # R (Reset semua sudut)
                pts = []
                current_box = None
                current_polygon = None
                if lbl_path.exists():
                    lbl_path.unlink()

    cv2.destroyAllWindows()
    print("\n" + "="*60)
    print("  SEMUA FOTO SELESAI DILABELI 4 SUDUT SECARA PRESISI!")
    print(f"  Total foto tersisa: {len(images)}")
    print("="*60)

if __name__ == "__main__":
    main()
