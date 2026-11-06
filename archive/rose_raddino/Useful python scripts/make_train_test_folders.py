"""
Sort the frontal CXRs into train/ and test/ folders using the shared split file.

Run:
  python3 make_train_test_folders.py --images "name of file w/ images" --split split_80_20.csv

"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, help="folder containing the frontal CXR .png files")
ap.add_argument("--split", default="split_80_20_seed42.csv", help="split file with image_name and split columns")
ap.add_argument("--out", default=".", help="folder to create train/ and test/ in (default: current folder)")
ap.add_argument("--move", action="store_true", help="move files instead of copying")
args = ap.parse_args()

src, out = Path(args.images), Path(args.out)
split = pd.read_csv(args.split)
assert {"image_name", "split"} <= set(split.columns), "split file needs image_name and split columns"
split["split"] = split["split"].str.strip().str.lower()
assert set(split["split"]) <= {"train", "test"}, f"unexpected split values: {set(split['split'])}"

for name in ("train", "test"):
    (out / name).mkdir(parents=True, exist_ok=True)

action = shutil.move if args.move else shutil.copy2
counts = {"train": 0, "test": 0}
missing = []
for row in split.itertuples():
    f = src / row.image_name
    if not f.exists():
        missing.append(row.image_name)
        continue
    action(str(f), str(out / row.split / row.image_name))
    counts[row.split] += 1

in_split = set(split["image_name"])
extra = sorted(p.name for p in src.glob("*.png") if p.name not in in_split)

print(f"train: {counts['train']} | test: {counts['test']} | total: {sum(counts.values())}")
print(f"expected from split file: train {(split.split == 'train').sum()} | test {(split.split == 'test').sum()}")
if missing:
    print(f"\nWARNING: {len(missing)} images listed in the split file were NOT found in {src}:")
    print("  " + "\n  ".join(missing[:20]) + ("\n  ..." if len(missing) > 20 else ""))
if extra:
    print(f"\nNote: {len(extra)} images in {src} are not in the split file and were left alone "
          f"(e.g. laterals): {extra[:5]}")