#!/usr/bin/env python3
"""Clinical Metadata T2D Prediction & Accuracy Scoring Script.

Reads patient clinical metadata and Social Deprivation Index (SDI) features
to predict Type 2 Diabetes (T2D).
Outputs:
  - 0: False (No T2D)
  - 1: True  (Has T2D)
along with prediction probability/confidence, and prints a comprehensive
accuracy and performance scorecard.

Usage:
  # Run full evaluation on Clinical_data_with_SDI.csv and print predictions + accuracy:
  .venv/bin/python predict_t2d_metadata.py

  # Limit printed patient rows to first 25:
  .venv/bin/python predict_t2d_metadata.py --limit 25

  # Predict T2D for a single patient profile:
  .venv/bin/python predict_t2d_metadata.py --patient-age 58 --patient-sex Female --patient-bmi 34.5 --patient-sdi 65.0

Authors: Malek & Angelo
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)


def build_clinical_preprocessor(df: pd.DataFrame):
    """Build imputation and encoding preprocessor for clinical columns."""
    exclude = {"image_name", "T2D", "tensor_path", "patient_id"}
    candidate_cols = [c for c in df.columns if c not in exclude]

    num_cols = []
    cat_cols = []
    for c in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )
    return preprocessor, num_cols, cat_cols


def run_metadata_evaluation(csv_path: Path, limit: int = None, threshold: float = 0.5):
    """Evaluate clinical metadata for T2D prediction and print accuracy scorecard."""
    if not csv_path.exists():
        print(f"Error: Clinical metadata CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if "T2D" not in df.columns:
        print("Error: Target column 'T2D' not found in CSV.")
        sys.exit(1)

    id_col = "image_name" if "image_name" in df.columns else None

    clean_df = df.dropna(subset=["T2D"]).copy()
    clean_df["T2D"] = clean_df["T2D"].astype(int)

    y = clean_df["T2D"].to_numpy()
    preprocessor, num_cols, cat_cols = build_clinical_preprocessor(clean_df)
    X = preprocessor.fit_transform(clean_df)

    print("=" * 85)
    print("      CLINICAL METADATA TYPE 2 DIABETES (T2D) PREDICTION SCREENER")
    print(f"      Evaluating {len(clean_df)} patient records using Clinical & SDI Covariates")
    print(f"      Covariates: Age, Sex, BMI, SDI score, Race, Ethnicity, Social Determinants")
    print("=" * 85)
    print(f"{'Patient / Image ID':<26} | {'Age':<4} | {'Sex':<6} | {'BMI':<5} | {'SDI':<5} | {'Predicted T2D':<13} | {'Prob':<6} | {'Actual':<6} | {'Result'}")
    print("-" * 85)

    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y))

    for train_idx, val_idx in skf.split(X, y):
        clf = LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X[train_idx], y[train_idx])
        oof_probs[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    oof_preds = (oof_probs >= threshold).astype(int)

    display_count = len(clean_df) if (limit is None or limit <= 0) else min(limit, len(clean_df))
    correct_count = 0

    for i in range(len(clean_df)):
        row = clean_df.iloc[i]
        pid = str(row[id_col]) if id_col else f"Patient_{i+1:04d}"
        age = f"{row.get('age', 'N/A'):.0f}" if pd.notna(row.get('age')) else "N/A"
        sex = str(row.get('sex', 'N/A'))[:6]
        bmi = f"{row.get('bmi', 'N/A'):.1f}" if pd.notna(row.get('bmi')) else "N/A"
        sdi_val = row.get('SDI_score') if pd.notna(row.get('SDI_score')) else row.get('SDI')
        sdi = f"{sdi_val:.1f}" if pd.notna(sdi_val) else "N/A"

        pred = oof_preds[i]
        actual = y[i]
        prob = oof_probs[i]
        is_correct = (pred == actual)
        if is_correct:
            correct_count += 1

        if i < display_count:
            pred_text = "1 (True)" if pred == 1 else "0 (False)"
            match_str = "[CORRECT]  " if is_correct else "[INCORRECT]"
            print(f"{pid:<26} | {age:<4} | {sex:<6} | {bmi:<5} | {sdi:<5} | {pred_text:<13} | {prob:.2f}   | {actual}      | {match_str}")

    if display_count < len(clean_df):
        print(f"... and {len(clean_df) - display_count} more patient records evaluated.")

    acc = accuracy_score(y, oof_preds)
    auc = roc_auc_score(y, oof_probs)
    auprc = average_precision_score(y, oof_probs)
    f1 = f1_score(y, oof_preds)
    tn, fp, fn, tp = confusion_matrix(y, oof_preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("=" * 85)
    print("                 CLINICAL METADATA ACCURACY & PERFORMANCE SCORECARD")
    print("=" * 85)
    print(f"  Total Patients Evaluated:       {len(y)}")
    print(f"  Total Correct Predictions:      {correct_count} / {len(y)}")
    print(f"  ACCURACY SCORE:                 {acc * 100:.2f}%")
    print(f"  AUROC (Discrimination):         {auc:.4f}")
    print(f"  AUPRC (Precision-Recall):       {auprc:.4f}")
    print(f"  Sensitivity (True Positives):   {sensitivity * 100:.2f}%  ({tp}/{tp+fn} T2D patients correctly detected)")
    print(f"  Specificity (True Negatives):   {specificity * 100:.2f}%  ({tn}/{tn+fp} Non-T2D correctly cleared)")
    print(f"  F1-Score:                       {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    - True Negatives  (Actual 0, Predicted 0): {tn}")
    print(f"    - False Positives (Actual 0, Predicted 1): {fp}")
    print(f"    - False Negatives (Actual 1, Predicted 0): {fn}")
    print(f"    - True Positives  (Actual 1, Predicted 1): {tp}")
    print("=" * 85)


def predict_single_patient(age: float, sex: str, bmi: float, sdi: float, csv_path: Path):
    """Predict T2D for an individual patient profile."""
    df = pd.read_csv(csv_path)
    clean_df = df.dropna(subset=["T2D"]).copy()
    y = clean_df["T2D"].astype(int).to_numpy()

    preprocessor, num_cols, cat_cols = build_clinical_preprocessor(clean_df)
    X = preprocessor.fit_transform(clean_df)

    clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X, y)

    patient_row = clean_df.iloc[[0]].copy()
    patient_row["age"] = age
    patient_row["sex"] = sex
    patient_row["bmi"] = bmi
    patient_row["SDI_score"] = sdi

    X_patient = preprocessor.transform(patient_row)
    prob = clf.predict_proba(X_patient)[0, 1]
    pred = 1 if prob >= 0.5 else 0
    pred_str = "1 (True - Likely Type 2 Diabetes)" if pred == 1 else "0 (False - Unlikely Type 2 Diabetes)"

    print("=" * 65)
    print("           INDIVIDUAL PATIENT METADATA T2D SCREENING")
    print("=" * 65)
    print(f"  Input Age:       {age}")
    print(f"  Input Sex:       {sex}")
    print(f"  Input BMI:       {bmi:.1f} kg/m²")
    print(f"  Input SDI Score: {sdi:.1f}")
    print("-" * 65)
    print(f"  PREDICTED T2D:   {pred_str}")
    print(f"  T2D RISK SCORE:  {prob * 100:.1f}% probability")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("Clinical_data_with_SDI.csv"), help="Clinical CSV")
    parser.add_argument("--limit", type=int, default=25, help="Number of patient rows to print (default: 25)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability cutoff")
    parser.add_argument("--patient-age", type=float, default=None, help="Age for single patient test")
    parser.add_argument("--patient-sex", type=str, default="Female", help="Sex for single patient test")
    parser.add_argument("--patient-bmi", type=float, default=None, help="BMI for single patient test")
    parser.add_argument("--patient-sdi", type=float, default=None, help="SDI for single patient test")
    args = parser.parse_args()

    if args.patient_age is not None and args.patient_bmi is not None:
        predict_single_patient(args.patient_age, args.patient_sex, args.patient_bmi, args.patient_sdi or 50.0, args.csv)
    else:
        run_metadata_evaluation(args.csv, limit=args.limit, threshold=args.threshold)


if __name__ == "__main__":
    main()
