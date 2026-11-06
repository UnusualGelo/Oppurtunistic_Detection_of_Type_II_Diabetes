"""
Build the full "RAD-DINO all layers" image-only package (models, predictions, results, README, predict example).

Run (Colab or anywhere; needs emb/raddino.npz from extract_multi.py and master_table.csv):
  python build_multilayer_package.py --root /content/root --emb emb/raddino.npz --outdir RADDINO_multilayer_package

Features: the four mean-patch-token vectors (25%, 50%, 75%, 100% of network depth: pm_L3, pm_L6, pm_L9, pm_L12)
concatenated -> 3072 numbers per image. Model: StandardScaler + L2 logistic regression (class_weight="balanced"),
C tuned by 5-fold stratified CV on the training rows only. Same protocol/seeds as step3_baselines.py and
eval_embeddings.py, so the numbers match your earlier table. Run 1 = all 949 images, Run 2 = lowest-50 hazy excluded.
"""
import argparse, shutil, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="Training"); ap.add_argument("--emb", default="emb/raddino.npz")
ap.add_argument("--outdir", default="RADDINO_multilayer_package"); ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
SEED = args.seed; CS = np.logspace(-4, 2, 7)
out = Path(args.outdir); (out / "models").mkdir(parents=True, exist_ok=True); (out / "predictions").mkdir(exist_ok=True)

m = pd.read_csv(Path(args.root) / "master_table.csv")
z = np.load(args.emb, allow_pickle=True)
assert list(z["ids"]) == m.image_name.tolist(), "embedding order does not match master_table.csv"
LAYER_KEYS = sorted([k for k in z.files if k.startswith("pm_L")], key=lambda s: int(s[4:]))
assert len(LAYER_KEYS) == 4, f"expected 4 pm_L* layers, found {LAYER_KEYS}"
X_all = np.hstack([z[k] for k in LAYER_KEYS]); y_all = m.T2D.values
print("layers used:", LAYER_KEYS, "| feature matrix", X_all.shape)
np.savez_compressed(out / "raddino_multilayer_embeddings.npz", ids=z["ids"], **{k: z[k] for k in LAYER_KEYS})

def search():
    pipe = Pipeline([("pre", StandardScaler()), ("lr", LogisticRegression(penalty="l2", class_weight="balanced", max_iter=5000))])
    return GridSearchCV(pipe, {"lr__C": CS}, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)

def boot(yt, p, n=2000):
    rng = np.random.default_rng(SEED); a = []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].min() != yt[i].max(): a.append(roc_auc_score(yt[i], p[i]))
    return np.percentile(a, [2.5, 97.5])

rows = []
for run, drop in (("run1", False), ("run2", True)):
    keep = ~m.exclude_lowest50.values if drop else np.ones(len(m), bool)
    tr = (m.split == "train").values & keep; te = (m.split == "test").values & keep
    Xtr, Xte, ytr, yte = X_all[tr], X_all[te], y_all[tr], y_all[te]
    oof = cross_val_predict(search(), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 1), method="predict_proba")[:, 1]
    gs = search().fit(Xtr, ytr); model = gs.best_estimator_; p = model.predict_proba(Xte)[:, 1]; pred = (p >= 0.5).astype(int)
    lo, hi = boot(yte, p)
    joblib.dump(model, out / "models" / f"model_{run}_raddino_multilayer.joblib")
    pd.DataFrame({"image_name": m.image_name[tr].values, "T2D": ytr, "oof_prob": oof}).to_csv(out / "predictions" / f"oof_{run}_raddino_multilayer.csv", index=False)
    pd.DataFrame({"image_name": m.image_name[te].values, "T2D": yte, "test_prob": p}).to_csv(out / "predictions" / f"testpred_{run}_raddino_multilayer.csv", index=False)
    rows.append(dict(run="Run 1: all 949" if not drop else "Run 2: lowest 50 excluded", model="RAD-DINO all layers (pm_L3+L6+L9+L12)",
                     file=f"model_{run}_raddino_multilayer.joblib", n_train=int(tr.sum()), n_test=int(te.sum()), test_T2D=int(yte.sum()),
                     best_C=gs.best_params_["lr__C"], cv_auc=round(roc_auc_score(ytr, oof), 3), test_auc=round(roc_auc_score(yte, p), 3),
                     test_auc_ci_low=round(lo, 3), test_auc_ci_high=round(hi, 3), accuracy=round(accuracy_score(yte, pred), 3),
                     recall=round(recall_score(yte, pred), 3), precision=round(precision_score(yte, pred), 3), f1=round(f1_score(yte, pred), 3)))
    print(rows[-1], flush=True)

res = pd.DataFrame(rows); res.to_csv(out / "image_only_multilayer_results.csv", index=False)

(out / "predict_example.py").write_text('''"""
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
''')

r1, r2 = rows
(out / "README_multilayer_model.md").write_text(f"""# RAD-DINO "all layers" image-only model (frozen RAD-DINO -> logistic regression)

What it is: logistic regression (L2, class_weight="balanced") on FROZEN RAD-DINO (microsoft/rad-dino) features from frontal chest X-rays.
RAD-DINO was not trained or changed. Each .joblib is a scikit-learn Pipeline (StandardScaler + logistic regression, best C chosen by 5-fold
stratified CV on the training images only). Seed {SEED}, scikit-learn {sklearn.__version__}.

Features: for each image, the mean of the patch tokens taken at four depths of the network (layers 3, 6, 9 and 12, i.e. 25/50/75/100% of
depth; the last-layer one is the same patch-token average used in the earlier baseline), concatenated = 3072 numbers. Whole image used
(squashed to 518x518, no centre crop), ImageNet-style normalisation from the model's own processor. Extracted by extract_multi.py.
Output: predicted probability of T2D (call T2D at >= 0.5).

| File | Training images |
|---|---|
| models/model_run1_raddino_multilayer.joblib | {r1['n_train']} (all) |
| models/model_run2_raddino_multilayer.joblib | {r2['n_train']} (lowest-50 hazy images excluded) |

Scores (image_only_multilayer_results.csv). CV AUC = out-of-fold on training images; test = held-out test set, scored once, 0.5 cutoff for
accuracy/recall/precision/F1:
- Run 1: CV AUC {r1['cv_auc']}, test AUC {r1['test_auc']} [{r1['test_auc_ci_low']}, {r1['test_auc_ci_high']}], accuracy {r1['accuracy']}, recall {r1['recall']}, precision {r1['precision']}, F1 {r1['f1']}
- Run 2: CV AUC {r2['cv_auc']}, test AUC {r2['test_auc']} [{r2['test_auc_ci_low']}, {r2['test_auc_ci_high']}], accuracy {r2['accuracy']}, recall {r2['recall']}, precision {r2['precision']}, F1 {r2['f1']}
Earlier baseline for comparison (patch-token average, last layer only): Run 1 CV 0.667 / test 0.631.

predictions/ holds out-of-fold predictions for training images (oof_*) and test predictions (testpred_*).
raddino_multilayer_embeddings.npz holds the four 768-number layer features per image (same row order as master_table.csv).
To reproduce a score: python predict_example.py --model models/model_run1_raddino_multilayer.joblib --table master_table.csv --emb raddino_multilayer_embeddings.npz
To rebuild everything: python build_multilayer_package.py --root <folder with master_table.csv> --emb emb/raddino.npz

Honest summary: the improvement over the last-layer baseline is about +0.007 CV AUC, which is within noise (fold-to-fold spread is ~0.05 and the
test-set 95% CI is about +/-0.08). Four other pretrained encoders (DINOv2 base/large, BiomedCLIP, MedSigLIP) and partial fine-tuning of RAD-DINO
all landed at the same ~0.65-0.68 CV AUC. Research only; not for clinical use.
""")
shutil.copy(__file__, out / "build_multilayer_package.py")
print("\npackage written to", out)
print(res.to_string(index=False))
