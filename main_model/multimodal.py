#!/usr/bin/env python3
"""multimodal.py - Multimodal Late Fusion Pipeline for Type 2 Diabetes Detection.

Fuses continuous chest radiograph predictions (CXR risk score) with clinical metadata
(Age, BMI, Sex, Marital Status, and Area Deprivation Index) using a Gradient Boosted Classifier.
Evaluates using patient-grouped 5-fold cross-validation on the development cohort.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Fusion model (CXR + Clinical) for T2D detection.")
    parser.add_argument("--splits-csv", default=str(ROOT / "splits/splits.csv"), help="Path to splits.csv")
    parser.add_argument("--clinical-csv", default="/home/e92/School/CS396/Clinical_data_with_SDI.csv", help="Path to clinical table")
    parser.add_argument("--cxr-oof-csv", default=str(ROOT / "predictions/densenet121_final/oof.csv"), help="Path to CXR OOF predictions")
    args = parser.parse_args()

    splits = pd.read_csv(args.splits_csv)
    clinical = pd.read_csv(args.clinical_csv)
    oof = pd.read_csv(args.cxr_oof_csv)

    # Use 'prob' if available (DenseNet predictions format) else fallback to 'predicted_prob'
    prob_col = "prob" if "prob" in oof.columns else "predicted_prob"
    df = pd.merge(splits, clinical, on="image_name", how="inner")
    df = pd.merge(df, oof[["image_name", prob_col]], on="image_name", how="inner")
    df["cxr_score"] = df[prob_col]

    dev_df = df[df["split"] == "dev"].copy().reset_index(drop=True)
    y = dev_df["label"].values
    folds = dev_df["fold"].values

    num_cols = ["cxr_score", "age", "bmi", "sdi", "SDI_score"]
    cat_cols = ["sex", "married"]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
    ])

    oof_preds = np.zeros(len(dev_df))
    fold_metrics = {}

    print(f"[multimodal.py] Evaluating Multimodal Fusion on dev set (N = {len(dev_df)})...")
    for f in sorted(np.unique(folds)):
        tr = (folds != f)
        va = (folds == f)
        
        X_tr = pre.fit_transform(dev_df.loc[tr, num_cols + cat_cols])
        X_va = pre.transform(dev_df.loc[va, num_cols + cat_cols])

        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        clf.fit(X_tr, y[tr])
        oof_preds[va] = clf.predict_proba(X_va)[:, 1]

        fold_metrics[f] = evaluate_predictions(y[va], oof_preds[va])
        print(f"  Fold {f}: AUC = {fold_metrics[f]['auroc']:.4f} | Recall = {fold_metrics[f]['recall']*100:.2f}% | F1 = {fold_metrics[f]['f1']:.4f}")

    overall = evaluate_predictions(y, oof_preds)
    print("\n[multimodal.py] Overall Multimodal Fusion (OOF Dev):")
    print(f"  AUROC:     {overall['auroc']:.4f}")
    print(f"  Accuracy:  {overall['accuracy']*100:.2f}%")
    print(f"  Recall:    {overall['recall']*100:.2f}% ({overall['tp']}/{overall['tp']+overall['fn']})")
    print(f"  Precision: {overall['precision']*100:.2f}%")
    print(f"  F1-Score:  {overall['f1']:.4f}")
    print(f"  Confusion: TN={overall['tn']}, FP={overall['fp']}, FN={overall['fn']}, TP={overall['tp']}")


if __name__ == "__main__":
    main()
