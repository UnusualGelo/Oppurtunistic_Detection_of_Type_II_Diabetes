"""
Step 2: extract RAD-DINO embeddings once for all 949 frontal CXRs in your folder.

Run (Colab or your computer):
  pip install torch transformers pandas pillow numpy
  python extract_raddino_embeddings.py \
      --images "Frontal xrays" \
      --clinical Clinical_Data_IncludedMissingSDI_2.csv \
      --split split_with_haziness_exclusion.csv \
      --out raddino_embeddings.npz

The image list comes from your folder (949 .png files), so report.csv is NOT needed.
No images are excluded here: Run 1 (all 949) and Run 2 (lowest 50 excluded) are chosen later,
using the split / exclude_lowest50 columns that get saved into master_table.csv.

Outputs
  raddino_embeddings.npz : ids, cls (N x 768), patch_mean (N x 768)
  master_table.csv       : one row per image: label + clinical columns + split + exclude_lowest50
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, help="folder with the 949 frontal .png images")
ap.add_argument("--clinical", required=True)
ap.add_argument("--split", default=None, help="split_with_haziness_exclusion.csv (adds split + exclude_lowest50)")
ap.add_argument("--out", default="raddino_embeddings.npz")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--table-only", action="store_true", help="only build master_table.csv, skip the model")
args = ap.parse_args()

# ---- 1. master table: images in the folder joined to the clinical file ----
files = sorted(p.name for p in Path(args.images).glob("*.png"))
clin = pd.read_csv(args.clinical).dropna(subset=["image_name", "T2D"]).drop_duplicates("image_name")
m = pd.DataFrame({"image_name": files}).merge(clin, on="image_name", how="left")
assert m["T2D"].notna().all(), f"{m['T2D'].isna().sum()} images have no clinical row, e.g. {m[m['T2D'].isna()].image_name.head(3).tolist()}"
m["T2D"] = m["T2D"].astype(int)
if args.split:
    s = pd.read_csv(args.split)[["image_name", "split", "exclude_lowest50"]]
    m = m.merge(s, on="image_name", how="left")
    assert m["split"].notna().all(), "some images are not in the split file"
print(f"images: {len(m)} | T2D: {m.T2D.sum()} | not T2D: {(m.T2D == 0).sum()}")
if args.split:
    print(m.groupby("split").T2D.agg(["size", "sum"]).rename(columns={"size": "images", "sum": "T2D"}).to_string())
    print("lowest-50 flagged:", int(m.exclude_lowest50.sum()))
assert len(m) == 949, f"expected 949 images, got {len(m)}"
m.to_csv("master_table.csv", index=False)
if args.table_only:
    raise SystemExit

# ---- 2. model ----
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
model = AutoModel.from_pretrained("microsoft/rad-dino").to(device).eval()

# ---- 3. extract (deterministic, so only done once) ----
cls_list, mean_list = [], []
names = m.image_name.tolist()
with torch.no_grad():
    for i in range(0, len(names), args.batch):
        batch = [Image.open(f"{args.images}/{n}").convert("RGB") for n in names[i:i + args.batch]]
        inp = proc(images=batch, return_tensors="pt").to(device)
        out = model(**inp)
        cls_list.append(out.pooler_output.cpu().numpy())                       # CLS token
        mean_list.append(out.last_hidden_state[:, 1:].mean(1).cpu().numpy())    # mean of patch tokens
        print(f"{min(i + args.batch, len(names))}/{len(names)}", end="\r")

np.savez(args.out, ids=np.array(names), cls=np.concatenate(cls_list), patch_mean=np.concatenate(mean_list))
print(f"\nsaved {args.out}")
