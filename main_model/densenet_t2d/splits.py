"""Leakage-safe data splits.

Creates ONE split file that every experiment reads, so all models are compared
on exactly the same images:

    split == "test"  -> locked hold-out set. Touched once, by final_test.py.
    split == "dev"   -> used for model development, with a `fold` column (0..K-1)
                        from StratifiedGroupKFold grouped by patient.

Usage
-----
    # Fresh patient-grouped split (20% test, 5 CV folds on the rest)
    python -m densenet_t2d.splits --csv patient.csv --image-dir "Image Folder" \
        --out splits/splits.csv

    # Re-use the team's existing hold-out test set (malek_and_angelo/malek)
    python -m densenet_t2d.splits --csv patient.csv --image-dir "Image Folder" \
        --out splits/splits.csv \
        --team-split ../../malek_and_angelo/malek/train_val_test_split.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PATIENT_REGEX = r"^(CXR\d+)"


def extract_patient_id(image_name: str, regex: str = PATIENT_REGEX) -> str:
    match = re.match(regex, str(image_name))
    return match.group(1) if match else str(image_name)


def load_labels(csv_file, image_dir, label_col="T2D", patient_col=None, regex=PATIENT_REGEX):
    df = pd.read_csv(csv_file)
    for col in ("image_name", label_col):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df = df.dropna(subset=["image_name", label_col]).copy()
    df["image_name"] = df["image_name"].astype(str)
    if not df[label_col].isin([0, 1]).all():
        raise ValueError(f"{label_col} labels must be 0 or 1")
    df["label"] = df[label_col].astype(int)

    if df["image_name"].duplicated().any():
        raise ValueError("Duplicate image filenames found")

    image_dir = Path(image_dir)
    missing = [n for n in df["image_name"] if not (image_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[:5]}")

    if patient_col and patient_col in df.columns:
        df["patient_id"] = df[patient_col].astype(str)
    else:
        df["patient_id"] = df["image_name"].map(lambda n: extract_patient_id(n, regex))

    # A patient must have one label; otherwise grouping is ambiguous.
    conflicting = df.groupby("patient_id")["label"].nunique()
    if (conflicting > 1).any():
        raise ValueError(f"Patients with conflicting labels: {list(conflicting[conflicting > 1].index[:5])}")

    return df[["image_name", "patient_id", "label"]].reset_index(drop=True)


def assign_folds(dev: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = pd.Series(-1, index=dev.index, dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(dev, dev["label"], groups=dev["patient_id"])):
        folds.iloc[idx] = k
    return folds


def make_splits(df, test_frac=0.2, n_folds=5, seed=42, team_split=None):
    df = df.copy()
    if team_split is not None:
        team = pd.read_csv(team_split)
        test_names = set(team.loc[team["split"].str.upper() == "TEST", "image_name"].astype(str))
        test_patients = set(df.loc[df["image_name"].isin(test_names), "patient_id"])
        df["split"] = np.where(df["patient_id"].isin(test_patients), "test", "dev")
        unmatched = len(test_names - set(df["image_name"]))
        if unmatched:
            print(f"[splits] note: {unmatched} team test images are not in your label CSV")
    else:
        # 1/test_frac folds, take one fold as test -> stratified AND patient-grouped
        n = max(2, round(1 / test_frac))
        sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
        _, test_idx = next(sgkf.split(df, df["label"], groups=df["patient_id"]))
        df["split"] = "dev"
        df.loc[df.index[test_idx], "split"] = "test"

    df["fold"] = -1
    dev_mask = df["split"] == "dev"
    df.loc[dev_mask, "fold"] = assign_folds(df[dev_mask], n_folds, seed).values
    check_splits(df)
    return df


def check_splits(df: pd.DataFrame) -> None:
    """Hard failures for anything that would leak information."""
    test_p = set(df.loc[df["split"] == "test", "patient_id"])
    dev_p = set(df.loc[df["split"] == "dev", "patient_id"])
    assert not (test_p & dev_p), "Patient appears in both dev and test"
    per_patient_folds = df[df["split"] == "dev"].groupby("patient_id")["fold"].nunique()
    assert (per_patient_folds == 1).all(), "Patient appears in more than one CV fold"
    assert (df.loc[df["split"] == "dev", "fold"] >= 0).all()


def summarize(df: pd.DataFrame) -> str:
    lines = [f"Images: {len(df)}  Patients: {df['patient_id'].nunique()}  "
             f"Prevalence: {df['label'].mean():.3f}"]
    multi = (df["patient_id"].value_counts() > 1).sum()
    lines.append(f"Patients with >1 image: {multi}")
    for name, g in df.groupby("split"):
        lines.append(f"  {name:4s}: n={len(g):4d}  pos={g['label'].sum():4d}  prev={g['label'].mean():.3f}")
    for k, g in df[df["split"] == "dev"].groupby("fold"):
        lines.append(f"    fold {k}: n={len(g):4d}  pos={g['label'].sum():4d}  prev={g['label'].mean():.3f}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--out", default="splits/splits.csv")
    p.add_argument("--label-col", default="T2D")
    p.add_argument("--patient-col", default=None, help="Use this column as patient id if present")
    p.add_argument("--patient-regex", default=PATIENT_REGEX)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--team-split", default=None, help="Re-use TEST rows of an existing split CSV")
    p.add_argument("--force", action="store_true", help="Overwrite an existing split file")
    a = p.parse_args()

    out = Path(a.out)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} already exists. Splits should be created once; pass --force to overwrite.")

    df = load_labels(a.csv, a.image_dir, a.label_col, a.patient_col, a.patient_regex)
    df = make_splits(df, a.test_frac, a.n_folds, a.seed, a.team_split)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(summarize(df))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
