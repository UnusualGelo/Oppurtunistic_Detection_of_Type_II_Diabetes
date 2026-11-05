"""
Compare frozen embeddings from several models with the SAME protocol as step3_baselines.py.

  python eval_embeddings.py --root /content/root --emb emb [--exclude-lowest50] --outdir eval_run1

Per feature set: logistic regression (L2, balanced), C tuned by inner 5-fold CV, nested 5-fold CV on the
TRAIN rows -> "CV AUC (train)"; then refit on train and scored ONCE on test (+ bootstrap 95% CI).
Models are compared and chosen on CV AUC only. The final ensemble rank-averages the top-N feature sets
(picked by CV AUC) and is evaluated the same way.
"""
import argparse, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="Training"); ap.add_argument("--emb", default="emb")
ap.add_argument("--outdir", default="eval_run"); ap.add_argument("--exclude-lowest50", action="store_true")
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--topn", type=int, default=3)
args = ap.parse_args()
out = Path(args.outdir); out.mkdir(exist_ok=True, parents=True); SEED = args.seed
CS = np.logspace(-4, 2, 7)

m = pd.read_csv(Path(args.root) / "master_table.csv")
keep = ~m.exclude_lowest50.values if args.exclude_lowest50 else np.ones(len(m), bool)
tr = ((m.split == "train").values & keep); te = ((m.split == "test").values & keep)
y = m.T2D.values

FS = {}
for f in sorted(glob.glob(f"{args.emb}/*.npz")):
    z = np.load(f, allow_pickle=True); name = Path(f).stem
    assert list(z["ids"]) == m.image_name.tolist(), f"{f}: row order differs from master_table.csv"
    keys = [k for k in z.files if k != "ids"]
    for k in keys: FS[f"{name}:{k}"] = z[k]
    pm = sorted([k for k in keys if k.startswith("pm_L")], key=lambda s: int(s[4:]))
    if len(pm) > 1: FS[f"{name}:pm_multilayer"] = np.hstack([z[k] for k in pm])
print(f"{len(FS)} feature sets | train {tr.sum()} (T2D {y[tr].sum()}) | test {te.sum()} (T2D {y[te].sum()})")

def search():
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(class_weight="balanced", max_iter=5000))])
    return GridSearchCV(pipe, {"lr__C": CS}, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)

def boot(yt, p, n=2000):
    rng = np.random.default_rng(SEED); a = []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].min() != yt[i].max(): a.append(roc_auc_score(yt[i], p[i]))
    return np.percentile(a, [2.5, 97.5])

OOF, TEST, rows = {}, {}, []
for name, X in FS.items():
    Xtr, Xte = X[tr], X[te]
    oof = cross_val_predict(search(), Xtr, y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 1),
                            method="predict_proba")[:, 1]
    gs = search().fit(Xtr, y[tr]); pte = gs.predict_proba(Xte)[:, 1]
    lo, hi = boot(y[te], pte); OOF[name], TEST[name] = oof, pte
    rows.append(dict(features=name, dim=X.shape[1], cv_auc_train=round(roc_auc_score(y[tr], oof), 3),
                     test_auc=round(roc_auc_score(y[te], pte), 3), ci_low=round(lo, 3), ci_high=round(hi, 3)))
    print(f"{name:32s} dim {X.shape[1]:5d}  CV {rows[-1]['cv_auc_train']:.3f} | test {rows[-1]['test_auc']:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)

res = pd.DataFrame(rows).sort_values("cv_auc_train", ascending=False)
top = res.features.head(args.topn).tolist()
rk = lambda d: np.mean([rankdata(d[k]) / len(d[k]) for k in top], axis=0)
eo, et = rk(OOF), rk(TEST); lo, hi = boot(y[te], et)
ens = dict(features=f"ENSEMBLE top{args.topn} (by CV): " + " + ".join(top), dim=0,
           cv_auc_train=round(roc_auc_score(y[tr], eo), 3), test_auc=round(roc_auc_score(y[te], et), 3),
           ci_low=round(lo, 3), ci_high=round(hi, 3))
res = pd.concat([res, pd.DataFrame([ens])], ignore_index=True)
res.to_csv(out / "results.csv", index=False)
pd.DataFrame({"image_name": m.image_name[tr].values, "T2D": y[tr], "oof_prob": eo}).to_csv(out / "oof_ensemble.csv", index=False)
pd.DataFrame({"image_name": m.image_name[te].values, "T2D": y[te], "test_prob": et}).to_csv(out / "testpred_ensemble.csv", index=False)
print("\n", res.to_string(index=False))
print("\nBaseline to beat (frozen RAD-DINO patch-mean + LR): CV 0.667 / test 0.631 [0.549, 0.714]")
