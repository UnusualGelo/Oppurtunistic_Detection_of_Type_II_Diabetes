"""Compare CV experiments on their out-of-fold (dev) predictions.

    python -m densenet_t2d.compare --baseline densenet121_xrv_final

Prints a table sorted by OOF ROC-AUC and, for every experiment, a paired
bootstrap test of AUC difference against the baseline on the SAME images.
If the 95% CI of the difference includes 0, treat the two as tied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .metrics import paired_bootstrap_auc_diff

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=str(ROOT / "results" / "experiments"))
    p.add_argument("--baseline", default=None)
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()

    rows, oofs = [], {}
    for s in sorted(Path(a.dir).glob("*/summary.json")):
        sm = json.loads(s.read_text())
        name = sm["experiment"]
        oofs[name] = pd.read_csv(s.parent / "oof.csv").set_index("image_name")
        r = sm["oof"]
        t05 = r["at_threshold"]["fixed_0.50"]
        rows.append({"experiment": name, "params": sm.get("params"),
                     "oof_auc": r["roc_auc"], "auc_ci_lo": r["roc_auc_ci95"][0], "auc_ci_hi": r["roc_auc_ci95"][1],
                     "oof_ap": r["average_precision"], "fold_auc_mean": sm["per_fold_auc_mean"],
                     "fold_auc_sd": sm["per_fold_auc_sd"], "sens@0.5": t05["sensitivity"],
                     "spec@0.5": t05["specificity"], "f1@0.5": t05["f1"],
                     "median_best_epoch": pd.Series(sm["best_epochs"]).median()})
    if not rows:
        raise SystemExit(f"No summary.json files under {a.dir}")
    df = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)

    if a.baseline:
        if a.baseline not in oofs:
            raise SystemExit(f"Baseline {a.baseline} not found")
        base = oofs[a.baseline]
        diffs = []
        for name in df["experiment"]:
            other = oofs[name]
            common = base.index.intersection(other.index)
            if len(common) != len(base) or len(common) != len(other):
                print(f"[warn] {name}: different image set from baseline ({len(common)} shared) - not comparable")
            y = base.loc[common, "label"].values
            d = paired_bootstrap_auc_diff(y, base.loc[common, "prob"].values, other.loc[common, "prob"].values, a.n_boot)
            diffs.append({"experiment": name, "auc_diff_vs_base": d["auc_diff"],
                          "diff_ci_lo": d["ci95"][0], "diff_ci_hi": d["ci95"][1], "p_value": d["p_value"]})
        df = df.merge(pd.DataFrame(diffs), on="experiment")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(df.round(4).to_string(index=False))
    out = Path(a.dir) / "comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
