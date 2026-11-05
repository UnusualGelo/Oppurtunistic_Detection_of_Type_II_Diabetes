"""
Load a saved model and score the held-out test images (proves the saved file works).

Run:
  python predict_example.py --model results/model_run1_embedding_patchmean.joblib \
                            --table master_table.csv --emb raddino_embeddings.npz
Use the matching feature set for each model (see README_models.md).
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--table", default="master_table.csv")
ap.add_argument("--emb", default="raddino_embeddings.npz")
args = ap.parse_args()

model = joblib.load(args.model)
m = pd.read_csv(args.table)
z = np.load(args.emb)
assert list(z["ids"]) == m.image_name.tolist()
if "sex" in m.columns:  # clinical columns only exist in the full (private) table; embedding models do not need them
    m["male"] = (m.sex == "Male").astype(int)
    m["married_yes"] = (m.married == "Yes").astype(int)
    m["ethnicity"] = m.ethnicity.fillna("unknown").replace({" ": "unknown"})
if "run2" in args.model:
    keep = ~m.exclude_lowest50
else:
    keep = pd.Series(True, index=m.index)
te = ((m.split == "test") & keep).values

name = args.model
if "embedding_cls" in name:
    X = z["cls"][te]
elif "embedding_patchmean" in name:
    X = z["patch_mean"][te]
elif "metadata_extended" in name:
    X = m.loc[te, ["age", "male", "bmi", "married_yes", "SDI", "race", "ethnicity"]]
else:  # metadata_basic
    X = m.loc[te, ["age", "male", "bmi"]].values

p = model.predict_proba(X)[:, 1]
print(f"{te.sum()} test images | predicted T2D probability range {p.min():.2f} to {p.max():.2f}")
print(f"Test AUC: {roc_auc_score(m.T2D[te], p):.3f}")
