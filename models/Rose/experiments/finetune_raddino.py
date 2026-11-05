"""
Fine-tune RAD-DINO for T2D detection from frontal chest X-rays (image only).

Colab (T4) quick start
  !pip install -q transformers pandas scikit-learn pillow
  !python finetune_raddino.py --root Training --outdir ft_run1 --unfreeze 4 --res 336

  --root must contain master_table.csv and dataset/train, dataset/test (your Training folder).

Protocol (matches your step3_baselines.py so numbers are comparable)
  * Train rows = split == "train", test rows = split == "test". --exclude-lowest50 gives Run 2.
  * 5-fold stratified CV on the TRAIN rows only. Each fold fine-tunes the model and records, per epoch,
    the validation-fold probabilities (out-of-fold, OOF) and the test probabilities.
  * The epoch is chosen from pooled OOF AUC on train ONLY. Test AUC (+ bootstrap 95% CI) is then reported
    for that epoch, averaged over the 5 fold models. Test is never used to pick anything.
  * Test predictions for every epoch are saved, but only the CV-chosen epoch is reported.

What is trained
  * Backbone: RAD-DINO (ViT-B/14). First (12 - K) transformer blocks stay frozen, last K blocks + final
    LayerNorm are trained with layer-wise LR decay. K=0 trains only the head (= your linear probe baseline,
    useful as a sanity check inside this same pipeline).
  * Head: concat(CLS, mean patch tokens) -> dropout -> linear logit  (same two features you used before).
  * Loss: BCE with pos_weight (class imbalance). AdamW, cosine schedule with warmup, mixed precision.
  * Augmentation: mild random-resized-crop, +/-7 deg rotation, brightness/contrast. No horizontal flip
    (heart position is anatomy).
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoImageProcessor, AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="Training")
ap.add_argument("--outdir", default="ft_run1")
ap.add_argument("--model", default="microsoft/rad-dino")
ap.add_argument("--exclude-lowest50", action="store_true", help="Run 2")
ap.add_argument("--unfreeze", type=int, default=4, help="number of last transformer blocks to train")
ap.add_argument("--res", type=int, default=336, help="square input size, multiple of 14 (518 = native, slow)")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-5, help="LR of the last block; earlier blocks decay by --lld")
ap.add_argument("--head-lr", type=float, default=1e-3)
ap.add_argument("--lld", type=float, default=0.8, help="layer-wise LR decay per block")
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--limit", type=int, default=0, help="debug: use only N train + N test rows")
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
root = Path(args.root)

# ---------------- data ----------------
m = pd.read_csv(root / "master_table.csv")
if args.exclude_lowest50:
    m = m[~m.exclude_lowest50].reset_index(drop=True)
if args.limit:
    m = pd.concat([m[m.split == "train"].groupby("T2D").head(args.limit // 2),
                   m[m.split == "test"].groupby("T2D").head(args.limit // 2)]).reset_index(drop=True)
proc = AutoImageProcessor.from_pretrained(args.model)
MEAN = torch.tensor(proc.image_mean).view(3, 1, 1); STD = torch.tensor(proc.image_std).view(3, 1, 1)
R = args.res
# Pre-size slightly larger than R so random crops have room; whole image is kept (no centre crop).
PRE = int(R * 1.1)

def load(split, name):
    im = Image.open(root / "dataset" / split / name).convert("RGB").resize((PRE, PRE), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im)).permute(2, 0, 1).contiguous()  # uint8 3xPRExPRE

print("loading images ...", flush=True)
IMG = torch.stack([load(s, n) for s, n in zip(m.split, m.image_name)])
y_all = m.T2D.values.astype(np.float32)
tr_idx = np.where(m.split == "train")[0]; te_idx = np.where(m.split == "test")[0]
print(f"train {len(tr_idx)} (T2D {int(y_all[tr_idx].sum())}), test {len(te_idx)} (T2D {int(y_all[te_idx].sum())})", flush=True)

def prep(batch_u8, train):
    x = batch_u8.float() / 255.0
    if train:
        B = x.shape[0]
        # random-resized-crop (scale 0.8-1) + small rotation via one affine grid
        s = torch.empty(B).uniform_(0.8, 1.0).sqrt()                   # side fraction
        ang = torch.empty(B).uniform_(-7, 7) * math.pi / 180
        tx = (1 - s) * torch.empty(B).uniform_(-1, 1); ty = (1 - s) * torch.empty(B).uniform_(-1, 1)
        theta = torch.zeros(B, 2, 3)
        theta[:, 0, 0] = s * torch.cos(ang); theta[:, 0, 1] = -s * torch.sin(ang); theta[:, 0, 2] = tx
        theta[:, 1, 0] = s * torch.sin(ang); theta[:, 1, 1] = s * torch.cos(ang); theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, (B, 3, R, R), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        b = torch.empty(B, 1, 1, 1).uniform_(-0.1, 0.1); c = torch.empty(B, 1, 1, 1).uniform_(0.85, 1.15)
        x = ((x - 0.5) * c + 0.5 + b).clamp(0, 1)
    else:
        x = F.interpolate(x, size=(R, R), mode="bilinear", align_corners=False, antialias=True)
    return (x - MEAN) / STD

# ---------------- model ----------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = AutoModel.from_pretrained(args.model)
        d = self.bb.config.hidden_size
        self.drop = nn.Dropout(0.3); self.fc = nn.Linear(2 * d, 1)
    def forward(self, x):
        h = self.bb(pixel_values=x).last_hidden_state
        f = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)   # CLS (post-layernorm) + mean patch
        return self.fc(self.drop(f)).squeeze(1)

def build():
    net = Net().to(dev)
    L = len(net.bb.encoder.layer); K = min(args.unfreeze, L)
    for p in net.bb.parameters(): p.requires_grad = False
    groups = [{"params": list(net.fc.parameters()), "lr": args.head_lr, "weight_decay": 1e-2}]
    for i in range(L - K, L):
        for p in net.bb.encoder.layer[i].parameters(): p.requires_grad = True
        groups.append({"params": list(net.bb.encoder.layer[i].parameters()),
                       "lr": args.lr * args.lld ** (L - 1 - i), "weight_decay": args.wd})
    if K > 0:
        for p in net.bb.layernorm.parameters(): p.requires_grad = True
        groups.append({"params": list(net.bb.layernorm.parameters()), "lr": args.lr, "weight_decay": 0.0})
    return net, groups

@torch.no_grad()
def predict(net, idx):
    net.eval(); ps = []
    for i in range(0, len(idx), 32):
        x = prep(IMG[idx[i:i + 32]], False).to(dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            ps.append(net(x).float().cpu())
    return torch.sigmoid(torch.cat(ps)).numpy()

def run_fold(tri, vai, fold):
    net, groups = build()
    opt = torch.optim.AdamW(groups)
    base = [g["lr"] for g in opt.param_groups]
    steps_per_ep = math.ceil(len(tri) / args.bs); total = steps_per_ep * args.epochs; warm = steps_per_ep
    pos_w = torch.tensor((1 - y_all[tri].mean()) / y_all[tri].mean(), device=dev)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    val_p, test_p, step = [], [], 0
    for ep in range(args.epochs):
        net.train(); rng = np.random.default_rng(args.seed * 100 + fold * 1000 + ep)
        perm = rng.permutation(tri); tl = 0
        for i in range(0, len(perm), args.bs):
            b = perm[i:i + args.bs]
            if len(b) < 2: continue
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))
            for g, bl in zip(opt.param_groups, base): g["lr"] = bl * f
            x = prep(IMG[b], True).to(dev); yy = torch.from_numpy(y_all[b]).to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                logit = net(x)
            loss = F.binary_cross_entropy_with_logits(logit.float(), yy, pos_weight=pos_w)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update(); step += 1; tl += loss.item() * len(b)
        pv = predict(net, vai); val_p.append(pv); test_p.append(predict(net, te_idx))
        print(f"  fold {fold} ep {ep + 1:2d}/{args.epochs} loss {tl / len(perm):.3f} val AUC {roc_auc_score(y_all[vai], pv):.3f}", flush=True)
    del net, opt; torch.cuda.empty_cache()
    return np.array(val_p), np.array(test_p)          # (epochs, n_val), (epochs, n_test)

# ---------------- CV on train ----------------
skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed + 1)
oof = np.zeros((args.epochs, len(tr_idx))); test_sum = np.zeros((args.epochs, len(te_idx)))
t0 = time.time()
for k, (a, b) in enumerate(skf.split(tr_idx, y_all[tr_idx])):
    v, t = run_fold(tr_idx[a], tr_idx[b], k)
    oof[:, b] = v; test_sum += t / args.folds
    print(f"fold {k} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)

y_tr, y_te = y_all[tr_idx], y_all[te_idx]
cv_auc = np.array([roc_auc_score(y_tr, oof[e]) for e in range(args.epochs)])
best = int(cv_auc.argmax())                                   # chosen on TRAIN OOF only
p_te = test_sum[best]; auc_te = roc_auc_score(y_te, p_te)
rng = np.random.default_rng(args.seed); bs = []
for _ in range(2000):
    i = rng.integers(0, len(y_te), len(y_te))
    if y_te[i].min() != y_te[i].max(): bs.append(roc_auc_score(y_te[i], p_te[i]))
lo, hi = np.percentile(bs, [2.5, 97.5])
res = dict(unfreeze=args.unfreeze, res=args.res, lr=args.lr, epochs=args.epochs, exclude_lowest50=args.exclude_lowest50,
           chosen_epoch=best + 1, cv_auc_train=round(float(cv_auc[best]), 3), test_auc=round(float(auc_te), 3),
           test_ci_low=round(float(lo), 3), test_ci_high=round(float(hi), 3),
           cv_auc_by_epoch=[round(float(a), 3) for a in cv_auc])
json.dump(res, open(out / "result.json", "w"), indent=1)
pd.DataFrame({"image_name": m.image_name.values[tr_idx], "T2D": y_tr, "oof_prob": oof[best]}).to_csv(out / "oof.csv", index=False)
pd.DataFrame({"image_name": m.image_name.values[te_idx], "T2D": y_te, "test_prob": p_te}).to_csv(out / "testpred.csv", index=False)
np.savez(out / "all_epochs.npz", oof=oof, test=test_sum)
print("\nCV AUC by epoch:", res["cv_auc_by_epoch"])
print(f"Chosen epoch {best + 1} | CV AUC (train, OOF) {cv_auc[best]:.3f} | TEST AUC {auc_te:.3f} [{lo:.3f}, {hi:.3f}]")
print("Baseline to beat (frozen patch-mean + LR): CV 0.667 / test 0.631 [0.549, 0.714]")
