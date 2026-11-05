"""
Rank chest X-rays by haziness (low contrast + low sharpness).

Run:
  python haziness_ranking.py --images "Frontal xrays" --split split_80_20.csv --out haziness_ranking.csv

Needs: pip install numpy pandas pillow

How it works (each image is converted to grayscale and resized to 512x512):
  LOW CONTRAST measures
    std      standard deviation of pixel brightness      (low = flat / washed out)
    range99  99th minus 1st percentile of brightness      (low = narrow brightness range)
  BLUR measures
    lapvar   variance of the Laplacian (edge filter)      (low = soft edges)
    grad     mean gradient magnitude                      (low = soft edges)
  Each measure is turned into a percentile rank across all images, and the four ranks
  are averaged into `score`. LOW score = HAZIER. rank 1 = hazi­est image.

Extra column (not in the first version):
  sharp_norm = lapvar / std^2 , sharpness with contrast divided out, so a flat-but-sharp image
  is not called blurry. Sort by sharp_norm (ascending) to look at blur only, and by std or
  range99 (ascending) to look at low contrast only.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, help="folder with the .png images")
ap.add_argument("--split", default=None, help="optional split file (image_name, T2D, split, hazy_flag); "
                                              "if given, only these images are ranked")
ap.add_argument("--out", default="haziness_ranking.csv")
args = ap.parse_args()

folder = Path(args.images)
if args.split:
    meta = pd.read_csv(args.split)
    names = meta["image_name"].tolist()
else:
    meta = None
    names = sorted(p.name for p in folder.glob("*.png"))

rows = []
for i, name in enumerate(names, 1):
    a = np.asarray(Image.open(folder / name).convert("L").resize((512, 512)), dtype=float)
    p1, p99 = np.percentile(a, [1, 99])
    lap = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    gy, gx = np.gradient(a)
    rows.append((name, a.std(), p99 - p1, lap.var(), np.hypot(gx, gy).mean()))
    print(f"{i}/{len(names)}", end="\r")

d = pd.DataFrame(rows, columns=["image_name", "std", "range99", "lapvar", "grad"])
if meta is not None:
    keep = [c for c in ["image_name", "hazy_flag", "T2D", "split"] if c in meta.columns]
    d = d.merge(meta[keep], on="image_name")

for c in ["std", "range99", "lapvar", "grad"]:
    d[c + "_pct"] = d[c].rank(pct=True).round(3)
d["score"] = d[["std_pct", "range99_pct", "lapvar_pct", "grad_pct"]].mean(axis=1)  # lower = hazier
d["sharp_norm"] = d["lapvar"] / d["std"] ** 2
d = d.sort_values("score").reset_index(drop=True)
d["rank"] = d.index + 1
d.to_csv(args.out, index=False)
print(f"\nsaved {args.out}  ({len(d)} images)")
print(d.head(15)[["rank", "image_name", "std", "range99", "lapvar", "score"]].round(2).to_string(index=False))
