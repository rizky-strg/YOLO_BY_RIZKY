"""Langkah 2: bagi train/val BERDASARKAN KONDISI/SESI (cegah data leakage).
Contoh:  python split.py                      (otomatis ~25% val)
         python split.py --val-kondisi jendela"""
import argparse, csv, random, shutil
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", default="dataset_raw")
p.add_argument("--out", default="dataset")
p.add_argument("--val-kondisi", nargs="*", default=None)
p.add_argument("--val-frac", type=float, default=0.25)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

rows = list(csv.DictReader(open(Path(a.root) / "metadata.csv", encoding="utf-8")))
groups = defaultdict(list)
for r in rows:
    groups[(r["tanggal"], r["kondisi_cahaya"])].append(r)
if len(groups) < 2:
    # Jika hanya 1 kondisi/sesi, gunakan pembagian acak terstratifikasi per kelas (75% train, 25% val)
    if Path(a.out).exists():
        shutil.rmtree(a.out)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["kelas"]].append(r)
    count = defaultdict(int)
    for cls_name, items in by_class.items():
        random.Random(a.seed).shuffle(items)
        n_val = max(1, int(len(items) * a.val_frac))
        val_items = items[:n_val]
        train_items = items[n_val:]
        for split, subset in [("train", train_items), ("val", val_items)]:
            for r in subset:
                dst = Path(a.out) / split / r["kelas"]
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(Path(a.root) / r["kelas"] / r["nama_file"], dst / r["nama_file"])
                count[(split, r["kelas"])] += 1
    print("Pembagian stratified per kelas selesai (75% train, 25% val):")
    for key in sorted(count): print(key, count[key])
    raise SystemExit(0)

if a.val_kondisi:
    val_keys = [k for k in groups if k[1] in a.val_kondisi]
else:
    keys = list(groups); random.Random(a.seed).shuffle(keys)
    val_keys, cnt = [], 0
    for k in keys[:-1]:                      # sisakan minimal 1 grup untuk train
        if cnt >= a.val_frac * len(rows): break
        val_keys.append(k); cnt += len(groups[k])

if Path(a.out).exists():
    shutil.rmtree(a.out)
count = defaultdict(int)
for k, items in groups.items():
    split = "val" if k in val_keys else "train"
    for r in items:
        dst = Path(a.out) / split / r["kelas"]; dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(a.root) / r["kelas"] / r["nama_file"], dst / r["nama_file"])
        count[(split, r["kelas"])] += 1
print("Kondisi val:", [k[1] for k in val_keys])
for key in sorted(count): print(key, count[key])
