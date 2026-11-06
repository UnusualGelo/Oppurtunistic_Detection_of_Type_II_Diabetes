"""
Training curves + validation graphs + metrics for the RAD-DINO "all layers" image-only model.

Run (Colab; needs emb/raddino.npz, master_table.csv, and the package from build_multilayer_package.py):
  python training_curves.py --root /content/root --emb emb/raddino.npz --pkg RADDINO_multilayer_package --outdir curves_run1
  python training_curves.py ... --run 2 --outdir curves_run2        # Run 2 = lowest-50 hazy excluded

Why this script exists: logistic regression is normally solved in one step, so sklearn gives no "training"
to plot. Here the SAME model (same loss, same class weights, same L2 strength C, same standardisation) is
trained step by step with Adam, so you can watch the loss fall. It converges to the same solution.

Everything for the curves is computed on the TRAIN rows only (5-fold CV: each fold's held-out part is the
"validation" set). The test set is touched once, at the end, for the ROC curve / confusion matrix / final metrics.

Outputs (in --outdir)
  1_loss_curve.png          training loss vs validation loss per epoch (mean +/- sd over 5 folds)
  2_validation_metrics.png  validation AUC, accuracy, recall, precision, F1 per epoch
  3_test_roc_confusion.png  test ROC curve + confusion matrix at the 0.5 cutoff
  4_learning_curve.png      train vs validation AUC as the amount of training data grows
  5_c_validation_curve.png  validation AUC vs regularisation strength C (how the C was chosen)
  epoch_metrics.csv, final_test_metrics.csv
"""
import argparse, warnings
from pathlib import Path
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="Training"); ap.add_argument("--emb", default="emb/raddino.npz")
ap.add_argument("--pkg", default="RADDINO_multilayer_package"); ap.add_argument("--run", type=int, default=1, choices=[1, 2])
ap.add_argument("--outdir", default="curves_run1"); ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--lr", type=float, default=0.002); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--C", type=float, default=0, help="override; default = best C stored in the package results csv")
args = ap.parse_args()
SEED = args.seed; out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)

m = pd.read_csv(Path(args.root) / "master_table.csv"); z = np.load(args.emb, allow_pickle=True)
assert list(z["ids"]) == m.image_name.tolist()
keys = sorted([k for k in z.files if k.startswith("pm_L")], key=lambda s: int(s[4:]))
X = np.hstack([z[k] for k in keys]).astype(np.float32); y = m.T2D.values.astype(np.float32)
keep = ~m.exclude_lowest50.values if args.run == 2 else np.ones(len(m), bool)
tr = (m.split == "train").values & keep; te = (m.split == "test").values & keep
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

C = args.C
if not C:
    res = pd.read_csv(Path(args.pkg) / "image_only_multilayer_results.csv")
    C = float(res[res.run.str.startswith(f"Run {args.run}")].best_C.iloc[0])
print(f"Run {args.run}: train {len(ytr)} (T2D {int(ytr.sum())}), test {len(yte)}, features {X.shape[1]}, C={C}")

def train_curve(Xa, ya, Xb, yb, epochs):
    """Full-batch Adam on the exact sklearn objective:  (1/n) sum_i w_i*BCE_i + ||w||^2 / (2*C*n).
    Returns per-epoch train loss, val loss (both plain class-weighted BCE) and val probabilities."""
    sc = StandardScaler().fit(Xa); A = torch.tensor(sc.transform(Xa), dtype=torch.float32); B = torch.tensor(sc.transform(Xb), dtype=torch.float32)
    ta, tb = torch.tensor(ya), torch.tensor(yb)
    n = len(ya); cw = {0: n / (2 * (ya == 0).sum()), 1: n / (2 * (ya == 1).sum())}   # sklearn "balanced"
    wa = torch.tensor([cw[int(v)] for v in ya]); wb = torch.tensor([cw[int(v)] for v in yb])
    w = torch.zeros(A.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=args.lr)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    tl, vl, vp = [], [], []
    for ep in range(epochs):
        logit = A @ w + b
        data = (wa * bce(logit, ta, reduction="none")).mean()
        loss = data + (w ** 2).sum() / (2 * C * n)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tl.append(float((wa * bce(A @ w + b, ta, reduction="none")).mean()))
            lv = B @ w + b; vl.append(float((wb * bce(lv, tb, reduction="none")).mean())); vp.append(torch.sigmoid(lv).numpy())
    return np.array(tl), np.array(vl), np.array(vp)

def metrics(yt, p, thr=0.5):
    pr = (p >= thr).astype(int)
    return dict(auc=roc_auc_score(yt, p), accuracy=accuracy_score(yt, pr), recall=recall_score(yt, pr),
                precision=precision_score(yt, pr, zero_division=0), f1=f1_score(yt, pr, zero_division=0))

# ---------- 1+2: epoch curves with 5-fold CV on train ----------
E = args.epochs; skf = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)
TL, VL, M = [], [], {k: [] for k in ("auc", "accuracy", "recall", "precision", "f1")}
for k, (a, b) in enumerate(skf.split(Xtr, ytr)):
    tl, vl, vp = train_curve(Xtr[a], ytr[a], Xtr[b], ytr[b], E)
    TL.append(tl); VL.append(vl)
    per = [metrics(ytr[b], vp[e]) for e in range(E)]
    for key in M: M[key].append([d[key] for d in per])
    print(f"  fold {k}: final train loss {tl[-1]:.3f}, val loss {vl[-1]:.3f}, val AUC {per[-1]['auc']:.3f}", flush=True)
TL, VL = np.array(TL), np.array(VL); ep = np.arange(1, E + 1)
best_ep = int(VL.mean(0).argmin()) + 1
pd.DataFrame({"epoch": ep, "train_loss": TL.mean(0), "val_loss": VL.mean(0), **{f"val_{k}": np.mean(v, 0) for k, v in M.items()}}).to_csv(out / "epoch_metrics.csv", index=False)

def band(ax, arr, label, color):
    mu, sd = arr.mean(0), arr.std(0); ax.plot(ep, mu, label=label, color=color); ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=.15)
fig, ax = plt.subplots(figsize=(7, 4.2))
band(ax, TL, "training loss", "#1f77b4"); band(ax, VL, "validation loss", "#d62728")
ax.set(xlabel="epoch (full pass over training data)", ylabel="class-weighted log-loss", title=f"Loss curve, RAD-DINO all layers, Run {args.run} (5-fold CV on training images)")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "1_loss_curve.png", dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 5, figsize=(17, 3.4), sharex=True)
for ax, (k, v), col in zip(axs, M.items(), plt.cm.tab10.colors):
    band(ax, np.array(v), k, col); ax.set_title(f"validation {k}"); ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    if k == "auc": ax.axhline(.5, c="gray", ls=":", lw=1)
fig.suptitle("Validation metrics during training (0.5 cutoff for all but AUC)", y=1.02); fig.tight_layout(); fig.savefig(out / "2_validation_metrics.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# ---------- 3: test ROC + confusion (package model, scored once) ----------
mp = Path(args.pkg) / "models" / f"model_run{args.run}_raddino_multilayer.joblib"
if mp.exists():
    model = joblib.load(mp); ptest = model.predict_proba(Xte)[:, 1]; src = "saved package model"
else:
    model = Pipeline([("pre", StandardScaler()), ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=5000))]).fit(Xtr, ytr)
    ptest = model.predict_proba(Xte)[:, 1]; src = f"refit with C={C}"
fm = metrics(yte, ptest); pd.DataFrame([fm]).round(3).to_csv(out / "final_test_metrics.csv", index=False)
fpr, tpr, _ = roc_curve(yte, ptest); cm = confusion_matrix(yte, (ptest >= .5).astype(int))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
a1.plot(fpr, tpr, c="#d62728", label=f"AUC {fm['auc']:.3f}"); a1.plot([0, 1], [0, 1], "k:", lw=1, label="chance")
a1.set(xlabel="false positive rate (1 - specificity)", ylabel="true positive rate (recall)", title="Test ROC curve"); a1.legend(loc="lower right"); a1.grid(alpha=.3)
a2.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2): a2.text(j, i, cm[i, j], ha="center", va="center", fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")
a2.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no T2D", "T2D"], yticklabels=["no T2D", "T2D"], xlabel="predicted", ylabel="actual",
       title=f"Test confusion matrix (0.5 cutoff)\nacc {fm['accuracy']:.2f}  recall {fm['recall']:.2f}  precision {fm['precision']:.2f}  F1 {fm['f1']:.2f}")
fig.tight_layout(); fig.savefig(out / "3_test_roc_confusion.png", dpi=160); plt.close(fig)

# ---------- 4: learning curve ----------
def lr(c): return Pipeline([("pre", StandardScaler()), ("lr", LogisticRegression(C=c, class_weight="balanced", max_iter=5000))])
fracs = [.2, .4, .6, .8, 1.0]; trA, vaA = [[] for _ in fracs], [[] for _ in fracs]
for a, b in skf.split(Xtr, ytr):
    rng = np.random.default_rng(SEED)
    for i, f in enumerate(fracs):
        sub = rng.permutation(a)[: int(len(a) * f)]; mdl = lr(C).fit(Xtr[sub], ytr[sub])
        trA[i].append(roc_auc_score(ytr[sub], mdl.predict_proba(Xtr[sub])[:, 1])); vaA[i].append(roc_auc_score(ytr[b], mdl.predict_proba(Xtr[b])[:, 1]))
sizes = [int(len(Xtr) * .8 * f) for f in fracs]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for arr, lab, col in ((trA, "training AUC", "#1f77b4"), (vaA, "validation AUC", "#d62728")):
    a = np.array(arr); ax.plot(sizes, a.mean(1), "o-", label=lab, c=col); ax.fill_between(sizes, a.mean(1) - a.std(1), a.mean(1) + a.std(1), color=col, alpha=.15)
ax.set(xlabel="training images used", ylabel="AUC", title="Learning curve (does more data help?)"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(out / "4_learning_curve.png", dpi=160); plt.close(fig)

# ---------- 5: validation curve over C ----------
Cs = np.logspace(-5, 0, 6); tA, vA = [], []
for c in Cs:
    t, v = [], []
    for a, b in skf.split(Xtr, ytr):
        mdl = lr(c).fit(Xtr[a], ytr[a]); t.append(roc_auc_score(ytr[a], mdl.predict_proba(Xtr[a])[:, 1])); v.append(roc_auc_score(ytr[b], mdl.predict_proba(Xtr[b])[:, 1]))
    tA.append(t); vA.append(v)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for arr, lab, col in ((tA, "training AUC", "#1f77b4"), (vA, "validation AUC", "#d62728")):
    a = np.array(arr); ax.semilogx(Cs, a.mean(1), "o-", label=lab, c=col); ax.fill_between(Cs, a.mean(1) - a.std(1), a.mean(1) + a.std(1), color=col, alpha=.15)
ax.axvline(C, ls="--", c="gray", lw=1); ax.set(xlabel="C (larger = weaker regularisation, more overfitting)", ylabel="AUC", title="Validation curve for regularisation strength"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(out / "5_c_validation_curve.png", dpi=160); plt.close(fig)

tn, fp, fn, tp = cm.ravel()
(out / "metrics_explained.md").write_text(f"""# What the metrics mean (RAD-DINO all layers, Run {args.run}, held-out test set)

Test set: {len(yte)} images, {int(yte.sum())} with T2D. The model outputs a probability; at the 0.5 cutoff, probability >= 0.5 counts as "T2D".
Confusion matrix: {tp} true positives (T2D, called T2D), {fn} false negatives (T2D, missed), {fp} false positives (no T2D, called T2D), {tn} true negatives.

| Metric | Plain meaning | Formula | Value |
|---|---|---|---|
| Accuracy | Of all images, what fraction did it get right? | (TP+TN) / all = ({tp}+{tn}) / {len(yte)} | {fm['accuracy']:.3f} |
| Recall (sensitivity) | Of the people who truly have T2D, what fraction did it catch? | TP / (TP+FN) = {tp} / {tp + fn} | {fm['recall']:.3f} |
| Precision | When it says T2D, how often is it right? | TP / (TP+FP) = {tp} / {tp + fp} | {fm['precision']:.3f} |
| F1 score | One number balancing precision and recall (their harmonic mean) | 2*P*R / (P+R) | {fm['f1']:.3f} |
| AUC | If you pick one T2D and one non-T2D image at random, how often does the model give the T2D image the higher score? 0.5 = coin flip, 1.0 = perfect. Does not depend on the 0.5 cutoff. | area under the ROC curve | {fm['auc']:.3f} |

Reading them together: only {fm['precision']*100:.0f}% of T2D calls are correct, and about {fm['recall']*100:.0f}% of true cases are caught. T2D is {yte.mean()*100:.0f}% of this test set, so a model that
guessed "T2D" at random would have precision near {yte.mean():.2f}; compare precision to that base rate, not to 100%. Accuracy can look fine just by favouring the bigger
group, which is why AUC, recall and precision are reported too. Small test set: AUC 95% CI is roughly +/-0.08.
""")
print(f"\nTEST ({src}): " + " | ".join(f"{k} {v:.3f}" for k, v in fm.items()) + f"\nconfusion matrix [[TN FP],[FN TP]]:\n{cm}")
print(f"Validation (CV) at last epoch: " + " | ".join(f"{k} {np.mean(v, 0)[-1]:.3f}" for k, v in M.items()))
print("graphs saved in", out)
