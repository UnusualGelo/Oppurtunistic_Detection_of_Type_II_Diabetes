"""
Load a saved model and score the held-out test images (proves the saved file works).
  python predict_example.py --model models/model_run1_raddino_multilayer.joblib --table master_table.csv --emb raddino_multilayer_embeddings.npz
"""
import argparse, joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
ap.add_argument("--table", default="master_table.csv"); ap.add_argument("--emb", default="raddino_multilayer_embeddings.npz")
a = ap.parse_args()
model = joblib.load(a.model); m = pd.read_csv(a.table); z = np.load(a.emb, allow_pickle=True)
assert list(z["ids"]) == m.image_name.tolist()
keys = sorted([k for k in z.files if k.startswith("pm_L")], key=lambda s: int(s[4:]))
X = np.hstack([z[k] for k in keys])                      # 4 layers x 768 = 3072 numbers per image
keep = ~m.exclude_lowest50 if "run2" in a.model else pd.Series(True, index=m.index)
te = ((m.split == "test") & keep).values
p = model.predict_proba(X[te])[:, 1]
print(f"{te.sum()} test images | predicted T2D probability range {p.min():.2f} to {p.max():.2f}")
print(f"Test AUC: {roc_auc_score(m.T2D[te], p):.3f}")
