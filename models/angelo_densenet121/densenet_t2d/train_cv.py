"""K-fold cross-validated training on the DEV split only (the test split is never loaded).

For each outer fold k (and each seed):
  1. outer-val = dev images in fold k        -> scored once, after training
  2. the rest is split again, by patient,
     into inner-train / inner-val            -> inner-val drives early stopping
  3. best-epoch weights predict outer-val    -> out-of-fold (OOF) predictions

Because the images that pick the epoch are never the images that are scored,
the OOF metrics are an honest estimate. Compare experiments on these numbers.

    python -m densenet_t2d.train_cv --config configs/densenet121_xrv_final.yaml
    python -m densenet_t2d.train_cv --config configs/densenet121_xrv_final.yaml --set train.lr=1e-3 --name densenet_lr1e-3
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from .data import CXRDataset, build_cache, load_split_table, make_train_transform
from .metrics import choose_threshold, full_report
from .models import build_model, count_params

ROOT = Path(__file__).resolve().parents[1]  # models/angelo_densenet121


# ----------------------------------------------------------------------------- config
def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, overrides=()):
    """YAML config; a `base:` key (relative path) is loaded first and overridden."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        if "model" in cfg:  # a model block replaces the base one (different models take different args)
            base.pop("model", None)
        cfg = deep_merge(base, cfg)
    for item in overrides:
        key, val = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- training
def inner_split(df, frac, seed):
    n = max(2, round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(df, df["label"], groups=df["patient_id"]))
    return df.iloc[tr], df.iloc[va]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.sigmoid(model(x.to(device))).squeeze(1).float().cpu())
    return torch.cat(out).numpy()


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            total += criterion(model(x), y).item() * len(y)
            n += len(y)
    return total / n


def make_scheduler(opt, epochs, warmup):
    def f(e):
        if e < warmup:
            return (e + 1) / warmup
        prog = (e - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def train_one(cfg, images, tr_df, va_df, device, seed, log_prefix=""):
    """Train on tr_df, early-stop on va_df. Returns best model + per-epoch history."""
    t = cfg["train"]
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)

    tr_ds = CXRDataset(images[tr_df["row"].values], tr_df["label"].values, make_train_transform(cfg))
    va_ds = CXRDataset(images[va_df["row"].values], va_df["label"].values)
    nw = t.get("num_workers", 0)
    tr_dl = DataLoader(tr_ds, batch_size=t["batch_size"], shuffle=True, drop_last=True, generator=g, num_workers=nw)
    va_dl = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=nw)

    model = build_model(cfg).to(device)
    pos = tr_df["label"].sum()
    pw = t.get("pos_weight", "auto")
    pw = (len(tr_df) - pos) / pos if pw == "auto" else float(pw)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pw)], dtype=torch.float32, device=device))
    smooth = float(t.get("label_smoothing", 0.0))

    opt = torch.optim.AdamW(model.param_groups(float(t["lr"]), float(t.get("backbone_lr_mult", 1.0))),
                            weight_decay=float(t.get("weight_decay", 1e-4)))
    sched = make_scheduler(opt, t["epochs"], t.get("warmup_epochs", 3))

    es = t.get("early_stopping", {})
    monitor, patience = es.get("monitor", "val_loss"), es.get("patience", 10)
    best_score, best_state, best_epoch, bad = -np.inf, None, 0, 0
    history = []

    for epoch in range(1, t["epochs"] + 1):
        model.train()
        run, n = 0.0, 0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            y_s = y * (1 - smooth) + 0.5 * smooth
            loss = criterion(model(x), y_s)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if t.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            run += loss.item() * len(y)
            n += len(y)
        sched.step()

        val_loss = evaluate_loss(model, va_dl, criterion, device)
        val_auc = roc_auc_score(va_df["label"].values, predict(model, va_dl, device))
        history.append({"epoch": epoch, "train_loss": run / n, "inner_val_loss": val_loss,
                        "inner_val_auc": val_auc, "lr": opt.param_groups[0]["lr"]})
        score = -val_loss if monitor == "val_loss" else val_auc
        improved = score > best_score + 1e-4
        if improved:
            best_score, best_state, best_epoch, bad = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
        print(f"{log_prefix} ep {epoch:3d} | train {run / n:.4f} | inner-val loss {val_loss:.4f} "
              f"auc {val_auc:.4f}{' *' if improved else ''}", flush=True)
        if bad >= patience:
            break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch, pw


def plot_curves(hist: pd.DataFrame, out_png: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    runs = list(hist.groupby(["fold", "seed"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (k, s), h in runs:
        axes[0].plot(h["epoch"], h["train_loss"], color="tab:blue", alpha=0.5)
        axes[0].plot(h["epoch"], h["inner_val_loss"], color="tab:orange", alpha=0.5)
        axes[1].plot(h["epoch"], h["inner_val_auc"], color="tab:green", alpha=0.5)
    axes[0].set(title="Loss (blue=train, orange=inner-val)", xlabel="epoch")
    axes[1].set(title="Inner-val ROC-AUC", xlabel="epoch")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(cfg, name=None, max_folds=None, overwrite=False):
    name = name or cfg["name"]
    out_dir = resolve(cfg.get("results_dir", "results/experiments")) / name
    if (out_dir / "summary.json").exists() and not overwrite:
        raise SystemExit(f"{out_dir} already has results. Use a new --name, or --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    d = cfg["data"]
    table = load_split_table(resolve(d["splits_csv"]))
    names = table["image_name"].tolist()
    images = build_cache(names, resolve(d["image_dir"]), d["image_size"], d.get("clahe", False),
                         resolve(d.get("cache_dir", "results/cache")))
    table["row"] = np.arange(len(table))
    dev = table[table["split"] == "dev"].reset_index(drop=True)  # test rows are never used here

    device = get_device()
    seeds = cfg["train"].get("seeds", [42])
    folds = sorted(dev["fold"].unique())[: max_folds or None]
    print(f"[cv] {name} | device={device} | dev={len(dev)} | folds={folds} | seeds={seeds} | "
          f"params={count_params(build_model(cfg)):,}")

    oof_rows, histories, fold_info = [], [], []
    t0 = time.time()
    for k in folds:
        outer = dev[dev["fold"] == k]
        pool = dev[dev["fold"] != k]
        for s in seeds:
            tr, va = inner_split(pool, cfg["train"].get("early_stopping", {}).get("inner_val_frac", 0.15), seed=s + k)
            model, hist, best_ep, pw = train_one(cfg, images, tr, va, device, seed=s, log_prefix=f"[fold {k} seed {s}]")
            outer_dl = DataLoader(CXRDataset(images[outer["row"].values], outer["label"].values), batch_size=64)
            p = predict(model, outer_dl, device)
            fold_auc = roc_auc_score(outer["label"], p)
            print(f"[fold {k} seed {s}] best epoch {best_ep} | OUTER-fold AUC {fold_auc:.4f}")
            torch.save({"state_dict": model.state_dict(), "config": cfg, "fold": int(k), "seed": int(s),
                        "best_epoch": best_ep, "pos_weight": pw}, out_dir / f"fold{k}_seed{s}.pt")
            hist["fold"], hist["seed"] = k, s
            histories.append(hist)
            fold_info.append({"fold": int(k), "seed": int(s), "best_epoch": best_ep, "outer_auc": fold_auc,
                              "n_train": len(tr), "n_inner_val": len(va), "n_outer": len(outer)})
            oof_rows.append(pd.DataFrame({"image_name": outer["image_name"].values, "patient_id": outer["patient_id"].values,
                                          "label": outer["label"].values, "fold": k, "seed": s, "prob": p}))

    hist = pd.concat(histories)
    hist.to_csv(out_dir / "history.csv", index=False)
    plot_curves(hist, out_dir / "learning_curves.png")
    oof_all = pd.concat(oof_rows)
    oof_all.to_csv(out_dir / "oof_per_seed.csv", index=False)
    # Average seeds -> one prediction per image (a small seed-ensemble)
    oof = oof_all.groupby(["image_name", "patient_id", "label", "fold"], as_index=False)["prob"].mean()
    oof.to_csv(out_dir / "oof.csv", index=False)

    ev = cfg.get("evaluation", {})
    thr = choose_threshold(oof["label"].values, oof["prob"].values, ev.get("threshold_method", "youden"),
                           ev.get("target_sensitivity", 0.8))
    report = full_report(oof["label"].values, oof["prob"].values, {"fixed_0.50": 0.5, "dev_selected": thr})
    fi = pd.DataFrame(fold_info)
    per_fold = oof.groupby("fold").apply(lambda g: roc_auc_score(g["label"], g["prob"]), include_groups=False)
    summary = {
        "experiment": name, "data_used": "dev split only (out-of-fold)", "params": count_params(build_model(cfg)),
        "oof": report,
        "per_fold_auc": {int(k): float(v) for k, v in per_fold.items()},
        "per_fold_auc_mean": float(per_fold.mean()), "per_fold_auc_sd": float(per_fold.std(ddof=1)) if len(per_fold) > 1 else None,
        "best_epochs": fi["best_epoch"].tolist(), "runs": fold_info,
        "selected_threshold": thr,
        "note": "dev_selected threshold metrics are optimistic (threshold chosen on these same OOF predictions). "
                "Its frozen value is what final_test.py applies to the test set.",
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    r = report
    print(f"\n[cv] {name}: OOF AUC {r['roc_auc']:.4f} (95% CI {r['roc_auc_ci95'][0]:.3f}-{r['roc_auc_ci95'][1]:.3f}) | "
          f"AP {r['average_precision']:.4f} | fold AUC {summary['per_fold_auc_mean']:.4f} ± {summary['per_fold_auc_sd'] or 0:.4f}")
    print(f"[cv] results -> {out_dir}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--name", default=None, help="Experiment name (defaults to config 'name')")
    p.add_argument("--set", nargs="*", default=[], help="Overrides, e.g. train.lr=1e-3 model.dropout=0.5")
    p.add_argument("--max-folds", type=int, default=None, help="Debug only: run the first N folds")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing experiment's results")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    if a.name:
        cfg["name"] = a.name
    run(cfg, a.name, a.max_folds, a.overwrite)


if __name__ == "__main__":
    main()
