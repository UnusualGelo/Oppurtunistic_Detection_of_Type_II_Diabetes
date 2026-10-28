#!/usr/bin/env python3
"""Multimodal Fusion Pipeline for Type 2 Diabetes Detection.

Stage 1: Converts raw chest radiographs into a continuous scalar biomarker
         ('cxr_t2d_score' / 'predicted_prob') via CustomScratchCXRNet (~75k params).
Stage 2: Fuses the X-ray continuous numerical score with clinical metadata
         (Age, Sex, BMI, SDI) using strict 5-fold Stratified Group Cross-Validation.

Author: Malek & Angelo
Course: CS 396 — AI in Healthcare
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def evaluate_binary(y_true, y_prob, threshold=0.5):
    """Compute standard healthcare ML evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def main():
    print("=" * 90)
    print("    STAGE 1: X-RAY BIOMARKER NUMBER  ──►  STAGE 2: MULTIMODAL FUSION MODEL")
    print("=" * 90)

    # 1. Load Stage 1 X-Ray Continuous Risk Scores
    cxr_oof_path = Path("/home/e92/School/CS396/Oppurtunistic_Detection_of_Type_II_Diabetes/malek_and_angelo/malek/oof_all_cxrs.csv")
    clinical_path = Path("/home/e92/School/CS396/Clinical_Data_Included_Missing_SDI.csv")

    print(f"\n[Stage 1] Loading X-Ray continuous prediction scores from {cxr_oof_path.name}...")
    cxr_df = pd.read_csv(cxr_oof_path)
    print(f"          Cohort: {len(cxr_df)} radiographs across 5 folds (Grouped by patient_id)")

    # Evaluate pure X-ray standalone
    cxr_metrics = evaluate_binary(cxr_df["actual_t2d"].values, cxr_df["predicted_prob"].values)
    print(f"          ► Pure X-Ray Vision AUROC:     {cxr_metrics['auroc']:.4f}")
    print(f"          ► Pure X-Ray Vision Accuracy:  {cxr_metrics['accuracy']*100:.2f}%")
    print(f"          ► Pure X-Ray Sensitivity (TP): {cxr_metrics['sensitivity']*100:.2f}% ({cxr_metrics['tp']}/{cxr_metrics['tp']+cxr_metrics['fn']})")
    print(f"          ► Pure X-Ray Specificity (TN): {cxr_metrics['specificity']*100:.2f}% ({cxr_metrics['tn']}/{cxr_metrics['tn']+cxr_metrics['fp']})")

    # 2. Load Clinical Metadata & Merge
    print(f"\n[Stage 2] Merging X-Ray numerical score with Clinical & SDI records...")
    clin_df = pd.read_csv(clinical_path)
    merged = pd.merge(cxr_df, clin_df, on="image_name", how="inner")
    print(f"          Successfully aligned {len(merged)} patients.")

    y = merged["actual_t2d"].values
    folds = merged["fold"].values
    
    # Feature subsets (strictly excluding administrative identifiers/zipcode)
    core_num = ["age", "bmi", "SDI", "SDI_RawScore"]
    core_cat = ["sex", "married", "race", "ethnicity"]

    # 3. Clinical Alone Benchmark
    clin_pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), core_num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), core_cat)
    ])

    clin_lr_oof = np.zeros(len(merged))
    clin_gb_oof = np.zeros(len(merged))

    for f in sorted(np.unique(folds)):
        tr = (folds != f)
        va = (folds == f)
        X_tr = clin_pre.fit_transform(merged.loc[tr, core_num + core_cat])
        X_va = clin_pre.transform(merged.loc[va, core_num + core_cat])
        
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        lr.fit(X_tr, y[tr])
        clin_lr_oof[va] = lr.predict_proba(X_va)[:, 1]

        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        gb.fit(X_tr, y[tr])
        clin_gb_oof[va] = gb.predict_proba(X_va)[:, 1]

    clin_lr_metrics = evaluate_binary(y, clin_lr_oof)
    clin_gb_metrics = evaluate_binary(y, clin_gb_oof)

    # 4. Multimodal Fusion: Fusing X-Ray Score + Clinical Covariates
    multi_num = ["predicted_prob"] + core_num
    multi_pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), multi_num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), core_cat)
    ])

    multi_lr_oof = np.zeros(len(merged))
    multi_gb_oof = np.zeros(len(merged))

    for f in sorted(np.unique(folds)):
        tr = (folds != f)
        va = (folds == f)
        X_tr = multi_pre.fit_transform(merged.loc[tr, multi_num + core_cat])
        X_va = multi_pre.transform(merged.loc[va, multi_num + core_cat])
        
        lr_m = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        lr_m.fit(X_tr, y[tr])
        multi_lr_oof[va] = lr_m.predict_proba(X_va)[:, 1]

        gb_m = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        gb_m.fit(X_tr, y[tr])
        multi_gb_oof[va] = gb_m.predict_proba(X_va)[:, 1]

    multi_lr_metrics = evaluate_binary(y, multi_lr_oof)
    multi_gb_metrics = evaluate_binary(y, multi_gb_oof)
    
    # Calibrated Soft Voting Ensemble
    multi_ensemble_oof = 0.5 * multi_lr_oof + 0.5 * multi_gb_oof
    multi_ens_metrics = evaluate_binary(y, multi_ensemble_oof)

    # 5. Side-by-side Table
    print("\n" + "=" * 98)
    print("                     MULTIMODAL FUSION BENCHMARK COMPARISON TABLE")
    print("=" * 98)

    comp_df = pd.DataFrame([
        {
            "Model Pipeline": "1. Pure X-Ray Vision (CXR Alone)",
            "Input Features": "Grayscale Radiograph (Pixels)",
            "AUROC": f"{cxr_metrics['auroc']:.4f}",
            "AUPRC": f"{cxr_metrics['auprc']:.4f}",
            "Accuracy": f"{cxr_metrics['accuracy']*100:.2f}%",
            "Recall (Sens.)": f"{cxr_metrics['sensitivity']*100:.2f}%",
            "Specificity": f"{cxr_metrics['specificity']*100:.2f}%",
            "Precision": f"{cxr_metrics['precision']*100:.2f}%",
            "F1-Score": f"{cxr_metrics['f1']:.4f}",
        },
        {
            "Model Pipeline": "2. Clinical Alone (Linear / LR)",
            "Input Features": "Age, Sex, BMI, SDI",
            "AUROC": f"{clin_lr_metrics['auroc']:.4f}",
            "AUPRC": f"{clin_lr_metrics['auprc']:.4f}",
            "Accuracy": f"{clin_lr_metrics['accuracy']*100:.2f}%",
            "Recall (Sens.)": f"{clin_lr_metrics['sensitivity']*100:.2f}%",
            "Specificity": f"{clin_lr_metrics['specificity']*100:.2f}%",
            "Precision": f"{clin_lr_metrics['precision']*100:.2f}%",
            "F1-Score": f"{clin_lr_metrics['f1']:.4f}",
        },
        {
            "Model Pipeline": "3. Multimodal Fusion (Linear / LR)",
            "Input Features": "X-Ray Score + Clinical & SDI",
            "AUROC": f"{multi_lr_metrics['auroc']:.4f}",
            "AUPRC": f"{multi_lr_metrics['auprc']:.4f}",
            "Accuracy": f"{multi_lr_metrics['accuracy']*100:.2f}%",
            "Recall (Sens.)": f"{multi_lr_metrics['sensitivity']*100:.2f}%",
            "Specificity": f"{multi_lr_metrics['specificity']*100:.2f}%",
            "Precision": f"{multi_lr_metrics['precision']*100:.2f}%",
            "F1-Score": f"{multi_lr_metrics['f1']:.4f}",
        },
        {
            "Model Pipeline": "4. Multimodal Fusion (Gradient Boosting)",
            "Input Features": "X-Ray Score + Clinical & SDI",
            "AUROC": f"{multi_gb_metrics['auroc']:.4f}",
            "AUPRC": f"{multi_gb_metrics['auprc']:.4f}",
            "Accuracy": f"{multi_gb_metrics['accuracy']*100:.2f}%",
            "Recall (Sens.)": f"{multi_gb_metrics['sensitivity']*100:.2f}%",
            "Specificity": f"{multi_gb_metrics['specificity']*100:.2f}%",
            "Precision": f"{multi_gb_metrics['precision']*100:.2f}%",
            "F1-Score": f"{multi_gb_metrics['f1']:.4f}",
        },
        {
            "Model Pipeline": "5. Multimodal Ensemble (Calibrated Blend)",
            "Input Features": "X-Ray Score + Clinical & SDI",
            "AUROC": f"{multi_ens_metrics['auroc']:.4f}",
            "AUPRC": f"{multi_ens_metrics['auprc']:.4f}",
            "Accuracy": f"{multi_ens_metrics['accuracy']*100:.2f}%",
            "Recall (Sens.)": f"{multi_ens_metrics['sensitivity']*100:.2f}%",
            "Specificity": f"{multi_ens_metrics['specificity']*100:.2f}%",
            "Precision": f"{multi_ens_metrics['precision']*100:.2f}%",
            "F1-Score": f"{multi_ens_metrics['f1']:.4f}",
        },
    ])
    
    print(comp_df.to_string(index=False))

    # 6. Save Artifacts
    out_dir = Path("/home/e92/School/CS396/Oppurtunistic_Detection_of_Type_II_Diabetes/malek_and_angelo/malek")
    
    merged["cxr_score"] = merged["predicted_prob"]
    merged["multimodal_prob"] = multi_gb_oof
    merged["multimodal_pred"] = (multi_gb_oof >= 0.5).astype(int)
    
    save_cols = [
        "image_name", "patient_id", "actual_t2d", "fold",
        "cxr_score", "multimodal_prob", "multimodal_pred"
    ]
    oof_multi_path = out_dir / "oof_multimodal.csv"
    merged[save_cols].to_csv(oof_multi_path, index=False)
    print(f"\n[Save] Multimodal OOF Predictions saved -> {oof_multi_path}")

    comp_csv_path = out_dir / "multimodal_benchmark_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"[Save] Comparison Table saved -> {comp_csv_path}")

    print(f"\n★ AUC JUMP: From pure X-Ray {cxr_metrics['auroc']:.4f} ──► Multimodal {multi_gb_metrics['auroc']:.4f} (+{multi_gb_metrics['auroc'] - cxr_metrics['auroc']:.4f} AUC increase!)")

if __name__ == "__main__":
    main()
