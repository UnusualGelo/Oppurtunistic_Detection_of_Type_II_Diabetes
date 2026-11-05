"""
Step 3: baseline models on RAD-DINO embeddings and clinical metadata.

Run:
  pip install numpy pandas scikit-learn
  python step3_baselines.py --table master_table.csv --emb raddino_embeddings.npz --outdir results

Design (same for every model)
  * Train = rows with split == "train", Test = rows with split == "test" (the shared team split).
  * Run 1 = all images.  Run 2 = drop exclude_lowest50 rows from BOTH train and test.
  * Logistic regression (L2), class_weight="balanced", scaling INSIDE the pipeline.
  * Regularization strength C tuned with an inner 5-fold stratified CV on the training rows only.
  * "CV AUC (train)" = nested CV: out-of-fold predictions on the training rows, with C tuned inside
    each outer fold. Out-of-fold predictions are saved to results/oof_*.csv.
  * "Test AUC" = model refit on all training rows, scored ONCE on the held-out test rows,
    with a 95% bootstrap CI (2000 resamples of the test rows).
  Do not use the test numbers to choose between models. Choose with the CV numbers.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--table", default="master_table.csv")
ap.add_argument("--emb", default="raddino_embeddings.npz")
ap.add_argument("--outdir", default="results")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
out = Path(args.outdir)
out.mkdir(exist_ok=True)
SEED = args.seed
CS = np.logspace(-4, 2, 7)

# ---------- load; align embeddings to the table ----------
m = pd.read_csv(args.table)
z = np.load(args.emb)
assert list(z["ids"]) == m.image_name.tolist(), "embedding order does not match master_table.csv"
EMB = {"cls": z["cls"], "patch_mean": z["patch_mean"]}
m["male"] = (m.sex == "Male").astype(int)
m["married_yes"] = (m.married == "Yes").astype(int)
m["ethnicity"] = m.ethnicity.fillna("unknown").replace({" ": "unknown"})

# ---------- model definitions ----------
def lr_search(pre):
    pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(penalty="l2", class_weight="balanced",
                                                             max_iter=5000))])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return GridSearchCV(pipe, {"lr__C": CS}, scoring="roc_auc", cv=cv, n_jobs=-1)

def meta_pre(cols_num, cols_cat=()):
    t = [("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                           ("sc", StandardScaler())]), list(cols_num))]
    if cols_cat:
        t.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(cols_cat)))
    return ColumnTransformer(t)

MODELS = {
    # name: (feature builder, preprocessor)
    "metadata_basic (age, sex, BMI)":
        (lambda d, i: d[["age", "male", "bmi"]].values, lambda: StandardScaler()),
    "metadata_extended (+race, ethnicity, married, SDI)":
        (lambda d, i: d[["age", "male", "bmi", "married_yes", "SDI", "race", "ethnicity"]],
         lambda: meta_pre(["age", "male", "bmi", "married_yes", "SDI"], ["race", "ethnicity"])),
    "embedding_cls (image only)":
        (lambda d, i: EMB["cls"][i], lambda: StandardScaler()),
    "embedding_patchmean (image only)":
        (lambda d, i: EMB["patch_mean"][i], lambda: StandardScaler()),
}

def boot_ci(y, p, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5])

rows = []
for run, drop in (("Run 1: all 949", False), ("Run 2: lowest 50 excluded", True)):
    d = m[~m.exclude_lowest50] if drop else m
    idx = d.index.values
    tr = (d.split == "train").values
    te = (d.split == "test").values
    y = d.T2D.values
    print(f"\n=== {run}: train {tr.sum()} (T2D {y[tr].sum()}), test {te.sum()} (T2D {y[te].sum()}) ===")
    for name, (feat, pre) in MODELS.items():
        X = feat(d, idx)
        Xtr = X[tr] if isinstance(X, np.ndarray) else X[tr]
        Xte = X[te] if isinstance(X, np.ndarray) else X[te]
        # nested CV on train -> OOF predictions
        outer = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)
        oof = cross_val_predict(lr_search(pre()), Xtr, y[tr], cv=outer, method="predict_proba")[:, 1]
        cv_auc = roc_auc_score(y[tr], oof)
        # final fit on train, score test once
        gs = lr_search(pre()).fit(Xtr, y[tr])
        pte = gs.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y[te], pte)
        lo, hi = boot_ci(y[te], pte)
        rows.append(dict(run=run, model=name, cv_auc_train=round(cv_auc, 3), best_C=gs.best_params_["lr__C"],
                         test_auc=round(auc, 3), test_ci_low=round(lo, 3), test_ci_high=round(hi, 3),
                         n_train=int(tr.sum()), n_test=int(te.sum())))
        tag = ("run2_" if drop else "run1_") + name.split(" ")[0]
        # save the fitted model (the trained pipeline: scaler/encoder + logistic regression with the best C)
        joblib.dump(gs.best_estimator_, out / f"model_{tag}.joblib")
        pd.DataFrame({"image_name": d.image_name[tr].values, "T2D": y[tr], "oof_prob": oof}).to_csv(
            out / f"oof_{tag}.csv", index=False)
        pd.DataFrame({"image_name": d.image_name[te].values, "T2D": y[te], "test_prob": pte}).to_csv(
            out / f"testpred_{tag}.csv", index=False)
        print(f"{name:55s} CV AUC {cv_auc:.3f} | test AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]")

res = pd.DataFrame(rows)
res.to_csv(out / "step3_results.csv", index=False)
print(f"saved {len(rows)} fitted models to {out}/model_*.joblib (scikit-learn {sklearn.__version__})")

# ---------- sanity check: can the embeddings predict age and sex? (Run 1 only) ----------
tr = (m.split == "train").values
te = (m.split == "test").values
sc = StandardScaler().fit(EMB["cls"][tr])
Xtr, Xte = sc.transform(EMB["cls"][tr]), sc.transform(EMB["cls"][te])
rr = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(Xtr, m.age[tr])
age_r = np.corrcoef(rr.predict(Xte), m.age[te])[0, 1]
gs = lr_search(StandardScaler()).fit(EMB["cls"][tr], m.male[tr])
sex_auc = roc_auc_score(m.male[te], gs.predict_proba(EMB["cls"][te])[:, 1])
print(f"\nSanity check (test rows): age from embeddings r = {age_r:.2f} | sex from embeddings AUC = {sex_auc:.3f}")
pd.DataFrame([dict(check="age_from_cls_embedding_pearson_r", value=round(age_r, 3)),
              dict(check="sex_from_cls_embedding_auc", value=round(sex_auc, 3))]).to_csv(
    out / "step3_sanity_checks.csv", index=False)
print(res.to_string(index=False))
