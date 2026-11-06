#!/usr/bin/env python3
"""Train Two From-Scratch CNN Models for Type 2 Diabetes Detection.

Requirement (Due Oct 22nd):
  - Model 1: All frontal CXRs (N=949)
  - Model 2: Hazy-11 CXRs excluded (N=938)
  - Evaluate using 5-Fold Stratified Group Cross-Validation (by patient_id).
  - Save Out-Of-Fold (OOF) prediction CSVs for both runs.

Author: Malek 
Course: CS 396 — AI in Healthcare
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# 1. Dataset & In-Memory Tensor Caching
# ---------------------------------------------------------------------------

def extract_patient_id(image_name: str) -> str:
    match = re.match(r"(CXR\d+)", str(image_name))
    return match.group(1) if match else str(image_name)


class InMemoryCXRDataset(Dataset):
    """Fast in-memory PyTorch dataset for 1x112x112 grayscale tensors."""

    def __init__(self, tensors: torch.Tensor, labels: torch.Tensor, df: pd.DataFrame):
        self.tensors = tensors
        self.labels = labels
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "tensor": self.tensors[idx],
            "label": self.labels[idx],
            "image_name": self.df.iloc[idx]["image_name"],
            "patient_id": self.df.iloc[idx]["patient_id"],
        }


def preload_images_to_memory(df: pd.DataFrame, image_dir: Path, target_size=(112, 112)) -> torch.Tensor:
    """Load, resize, and normalize images once into RAM for fast CPU training."""
    tensor_list = []
    print(f"[DataLoader] Preloading {len(df)} images into RAM at {target_size} resolution...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preloading"):
        img_path = image_dir / row["image_name"]
        with Image.open(img_path) as img:
            # Grayscale 8-bit conversion
            gray = img.convert("L").resize(target_size, Image.Resampling.BILINEAR)
            arr = np.array(gray, dtype=np.float32) / 255.0
            # Standard radiograph normalization: mean ~0.5, std ~0.25
            arr = (arr - 0.5) / 0.25
            tensor = torch.from_numpy(arr).unsqueeze(0)  # Shape: (1, H, W)
            tensor_list.append(tensor)
    return torch.stack(tensor_list)


# ---------------------------------------------------------------------------
# 2. From-Scratch CNN Architecture
# ---------------------------------------------------------------------------

class CustomScratchCXRNet(nn.Module):
    """Lightweight 4-block CNN designed from scratch for thoracic radiograph features.

    Parameters: ~75,000 parameters (fast and resistant to overfitting on ~950 images).
    """

    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 112x112 -> 56x56
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 56x56 -> 28x28
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# 3. Training & Evaluation Engine
# ---------------------------------------------------------------------------

def evaluate_metrics(y_true, y_prob):
    """Compute standard healthcare ML evaluation metrics."""
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def train_scratch_model(
    cohort_df: pd.DataFrame,
    preloaded_tensors: torch.Tensor,
    experiment_name: str,
    output_dir: Path,
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 8e-4,
    device: str = "cpu",
) -> tuple:
    """Train custom CNN from scratch across 5-fold Stratified Group Cross-Validation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y = cohort_df["T2D"].to_numpy().astype(int)
    groups = cohort_df["patient_id"].to_numpy()

    # Class weights for imbalanced data (~32% T2D)
    pos_weight = (y == 0).sum() / (y == 1).sum()
    class_weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32).to(device)

    print("\n" + "=" * 70)
    print(f"  STARTING RUN: {experiment_name}")
    print(f"  Cohort Size: {len(cohort_df)} images ({len(np.unique(groups))} unique patients)")
    print(f"  T2D Cases: {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%) | Non-T2D: {(y == 0).sum()}")
    print(f"  Cross-Validation: 5-Fold Stratified Group K-Fold (grouped by patient_id)")
    print("=" * 70)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y))
    oof_preds = np.zeros(len(y), dtype=int)
    fold_assignments = np.zeros(len(y), dtype=int)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups=groups), 1):
        fold_assignments[val_idx] = fold

        train_tensors = preloaded_tensors[train_idx]
        train_labels = torch.from_numpy(y[train_idx]).long()
        val_tensors = preloaded_tensors[val_idx]
        val_labels = torch.from_numpy(y[val_idx]).long()

        train_ds = InMemoryCXRDataset(train_tensors, train_labels, cohort_df.iloc[train_idx])
        val_ds = InMemoryCXRDataset(val_tensors, val_labels, cohort_df.iloc[val_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = CustomScratchCXRNet(in_channels=1, num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_auc = -1.0
        best_val_probs = None

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                x = batch["tensor"].to(device)
                target = batch["label"].to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)

            scheduler.step()

            # Validation
            model.eval()
            val_probs_epoch = []
            val_y_epoch = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["tensor"].to(device)
                    target = batch["label"].to(device)
                    logits = model(x)
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    val_probs_epoch.extend(probs)
                    val_y_epoch.extend(target.cpu().numpy())

            val_probs_epoch = np.array(val_probs_epoch)
            val_y_epoch = np.array(val_y_epoch)

            try:
                val_auc = roc_auc_score(val_y_epoch, val_probs_epoch)
            except ValueError:
                val_auc = 0.5

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_probs = val_probs_epoch

        oof_probs[val_idx] = best_val_probs
        fold_metrics.append(evaluate_metrics(y[val_idx], best_val_probs))
        print(f"  Fold {fold}/5 Complete -> Best Val AUROC: {best_val_auc:.4f}")

    # Out-of-fold metrics across entire cohort
    overall_metrics = evaluate_metrics(y, oof_probs)

    # Save OOF Predictions CSV
    oof_df = pd.DataFrame({
        "image_name": cohort_df["image_name"],
        "patient_id": cohort_df["patient_id"],
        "actual_t2d": y,
        "predicted_prob": np.round(oof_probs, 4),
        "predicted_t2d": (oof_probs >= 0.5).astype(int),
        "fold": fold_assignments,
    })

    oof_csv_path = output_dir / f"oof_{experiment_name.lower().replace(' ', '_')}.csv"
    oof_df.to_csv(oof_csv_path, index=False)
    print(f"\n[Save] OOF Predictions saved -> {oof_csv_path}")

    return overall_metrics, oof_df


# ---------------------------------------------------------------------------
# 4. Main Experiment Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("Image Folder"))
    parser.add_argument("--clinical-csv", type=Path, default=Path("Clinical_data_with_SDI.csv"))
    parser.add_argument("--view-manifest", type=Path, default=Path("X-rays by View/view_classification_manifest.csv"))
    parser.add_argument("--qc-manifest", type=Path, default=Path("image_qc_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("Oppurtunistic_Detection_of_Type_II_Diabetes/malek_and_angelo"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # 1. Build Base Frontal Cohort (949 images)
    view_df = pd.read_csv(args.view_manifest)
    clin_df = pd.read_csv(args.clinical_csv)
    qc_df = pd.read_csv(args.qc_manifest)

    frontal_df = view_df[view_df["view"] == "Frontal"].copy()
    cohort_all = pd.merge(frontal_df, clin_df, left_on="filename", right_on="image_name", how="inner")
    cohort_all["patient_id"] = cohort_all["image_name"].apply(extract_patient_id)
    print(f"[Setup] Loaded {len(cohort_all)} frontal radiographs aligned with T2D labels.")

    # 2. Identify the 11 Hazy Images
    merged_qc = pd.merge(cohort_all, qc_df[["filename", "laplacian_variance"]], on="filename", how="left")
    hazy_11_files = set(merged_qc.sort_values("laplacian_variance")["filename"].head(11))
    print(f"[Setup] Identified 11 hazy radiographs (lowest Laplacian variance):")
    for f in sorted(hazy_11_files):
        print(f"   - {f}")

    cohort_no_hazy = cohort_all[~cohort_all["filename"].isin(hazy_11_files)].copy()
    print(f"[Setup] Cohort 1 (All CXRs):     {len(cohort_all)} images")
    print(f"[Setup] Cohort 2 (No Hazy 11):   {len(cohort_no_hazy)} images")

    # 3. Preload images to memory
    all_tensors = preload_images_to_memory(cohort_all, args.image_dir)
    no_hazy_indices = [i for i, f in enumerate(cohort_all["filename"]) if f not in hazy_11_files]
    no_hazy_tensors = all_tensors[no_hazy_indices]

    # 4. Train Model 1 (All CXRs)
    m1_metrics, oof_all = train_scratch_model(
        cohort_all,
        all_tensors,
        experiment_name="all_cxrs",
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # 5. Train Model 2 (Hazy-11 Excluded)
    m2_metrics, oof_no_hazy = train_scratch_model(
        cohort_no_hazy,
        no_hazy_tensors,
        experiment_name="no_hazy",
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # 6. Generate Comparison Table & Report
    summary_df = pd.DataFrame([
        {
            "Dataset": "Model 1: All CXRs",
            "N": len(cohort_all),
            "AUROC": round(m1_metrics["auroc"], 4),
            "AUPRC": round(m1_metrics["auprc"], 4),
            "Accuracy": f"{m1_metrics['accuracy'] * 100:.2f}%",
            "Sensitivity": f"{m1_metrics['sensitivity'] * 100:.2f}%",
            "Specificity": f"{m1_metrics['specificity'] * 100:.2f}%",
            "F1-Score": round(m1_metrics["f1"], 4),
        },
        {
            "Dataset": "Model 2: Hazy-11 Excluded",
            "N": len(cohort_no_hazy),
            "AUROC": round(m2_metrics["auroc"], 4),
            "AUPRC": round(m2_metrics["auprc"], 4),
            "Accuracy": f"{m2_metrics['accuracy'] * 100:.2f}%",
            "Sensitivity": f"{m2_metrics['sensitivity'] * 100:.2f}%",
            "Specificity": f"{m2_metrics['specificity'] * 100:.2f}%",
            "F1-Score": round(m2_metrics["f1"], 4),
        },
    ])

    summary_csv = args.output_dir / "scratch_models_comparison.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 75)
    print("           FINAL FROM-SCRATCH MODEL COMPARISON REPORT")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    print("=" * 75)

    # Also save to parent directory for quick access
    oof_all.to_csv("oof_all_cxrs.csv", index=False)
    oof_no_hazy.to_csv("oof_no_hazy.csv", index=False)
    summary_df.to_csv("scratch_models_comparison.csv", index=False)
    print("\n[Done] All models trained and OOF files saved successfully!")


if __name__ == "__main__":
    main()
