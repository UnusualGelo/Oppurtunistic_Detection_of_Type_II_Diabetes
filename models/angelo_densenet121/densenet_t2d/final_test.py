"""One-time evaluation of a FINISHED experiment on the locked test split.

The K fold models from train_cv.py are used as an ensemble (mean probability),
so no retraining and no extra choices are made after looking at the dev data.
The decision threshold is the one frozen from OOF predictions in summary.json.

Every evaluation is appended to results/test_log.csv. Running a second model on
the test set requires --force and is logged: each extra look turns the test set
a little more into a validation set, so do this only for your final 1-2 models.

    python -m densenet_t2d.final_test --experiment densenet121_xrv_final
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .data import CXRDataset, build_cache, load_split_table
from .metrics import full_report
from .models import build_model
from .train_cv import get_device, predict, resolve

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, help="Name under results/experiments/ or a path")
    p.add_argument("--force", action="store_true", help="Allow evaluating after the test set was already used")
    a = p.parse_args()

    exp = Path(a.experiment)
    if not exp.exists():
        exp = ROOT / "results" / "experiments" / a.experiment
    summary = json.loads((exp / "summary.json").read_text())
    log_path = ROOT / "results" / "test_log.csv"
    if log_path.exists() and not a.force:
        prev = pd.read_csv(log_path)
        raise SystemExit(f"Test set already evaluated {len(prev)} time(s): {prev['experiment'].tolist()}.\n"
                         f"Re-run with --force only if this is a pre-planned final comparison.")

    ckpts = sorted(exp.glob("fold*_seed*.pt"))
    if not ckpts:
        raise SystemExit(f"No fold checkpoints in {exp}")
    cfg = torch.load(ckpts[0], map_location="cpu", weights_only=False)["config"]
    d = cfg["data"]
    table = load_split_table(resolve(d["splits_csv"]))
    images = build_cache(table["image_name"].tolist(), resolve(d["image_dir"]), d["image_size"],
                         d.get("clahe", False), resolve(d.get("cache_dir", "results/cache")))
    test_mask = (table["split"] == "test").values
    test = table[test_mask].reset_index(drop=True)
    dl = DataLoader(CXRDataset(images[test_mask], test["label"].values), batch_size=64)

    device = get_device()
    probs = []
    for c in ckpts:
        ck = torch.load(c, map_location=device, weights_only=False)
        model = build_model(ck["config"]).to(device)
        model.load_state_dict(ck["state_dict"])
        probs.append(predict(model, dl, device))
    probs = np.stack(probs)
    p_ens = probs.mean(0)

    thr = summary["selected_threshold"]
    report = full_report(test["label"].values, p_ens, {"fixed_0.50": 0.5, "dev_selected": thr})
    report["experiment"] = summary["experiment"]
    report["n_models_in_ensemble"] = len(ckpts)
    report["single_model_test_auc"] = {c.stem: float(roc_auc_score(test["label"], pr)) for c, pr in zip(ckpts, probs)}
    report["dev_oof_auc"] = summary["oof"]["roc_auc"]
    (exp / "test_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame({"image_name": test["image_name"], "label": test["label"], "prob": p_ens}).to_csv(
        exp / "test_predictions.csv", index=False)

    entry = pd.DataFrame([{"experiment": summary["experiment"], "timestamp": datetime.now().isoformat(timespec="seconds"),
                           "test_auc": report["roc_auc"], "forced": a.force}])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    r, t = report, report["at_threshold"]["dev_selected"]
    print(f"TEST  n={r['n']} pos={r['n_pos']}")
    print(f"  ROC-AUC {r['roc_auc']:.4f} (95% CI {r['roc_auc_ci95'][0]:.3f}-{r['roc_auc_ci95'][1]:.3f})   "
          f"[dev OOF was {r['dev_oof_auc']:.4f}]")
    print(f"  AP      {r['average_precision']:.4f} (95% CI {r['average_precision_ci95'][0]:.3f}-{r['average_precision_ci95'][1]:.3f})"
          f"   [prevalence {r['prevalence']:.3f}]")
    print(f"  @thr {thr:.3f} (frozen from dev): sens {t['sensitivity']:.3f} spec {t['specificity']:.3f} "
          f"F1 {t['f1']:.3f} CM {t['confusion_matrix']}")
    print(f"Saved {exp / 'test_report.json'}")


if __name__ == "__main__":
    main()
