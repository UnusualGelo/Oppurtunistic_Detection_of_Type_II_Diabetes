#!/usr/bin/env python3
"""Chest X-ray T2D Prediction Script.

Evaluates chest X-rays to predict whether patients have Type 2 Diabetes (T2D).
Outputs:
  - 0: False (No T2D)
  - 1: True  (Has T2D)
along with prediction probability/confidence and accuracy scoring.

Usage:
  # Test the full clean cohort and print predictions + accuracy:
  .venv/bin/python predict_t2d_image.py

  # Test first 20 images from cohort:
  .venv/bin/python predict_t2d_image.py --limit 20

  # Test a single raw image:
  .venv/bin/python predict_t2d_image.py --image "Image Folder/CXR1001_IM-0004-1001.png"

Authors: Malek & Angelo
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as T
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

MODEL_ID = "microsoft/rad-dino"

# Preprocessing transforms for raw input images
IMAGE_TRANSFORM = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_feature_extractor(device: str = "cpu"):
    """Load pretrained RAD-DINO vision transformer."""
    print(f"[Model] Loading vision foundation model '{MODEL_ID}' on {device}...")
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_single_embedding(image_path: Path, model, device: str = "cpu") -> np.ndarray:
    """Extract 768-D embedding from a single image file."""
    img = Image.open(image_path).convert("RGB")
    tensor = IMAGE_TRANSFORM(img).unsqueeze(0).to(device)
    outputs = model(pixel_values=tensor)
    cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return cls_emb


def train_classifier_head(embeddings: np.ndarray, labels: np.ndarray):
    """Train regularized logistic regression head on vision embeddings."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    clf = LogisticRegression(
        penalty="l2",
        C=0.1,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_scaled, labels)
    return scaler, clf


def run_cohort_evaluation(npz_path: Path, limit: int = None, threshold: float = 0.5):
    """Run cross-validated prediction on the entire clean cohort with console output."""
    if not npz_path.exists():
        print(f"Error: Embeddings file not found: {npz_path}")
        print("Run 'python run_rad_dino_pipeline.py' first to generate embeddings.")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    filenames = data["filenames"]
    patient_ids = data["patient_ids"]

    n_total = len(labels)
    display_limit = limit if (limit is not None and limit > 0) else n_total

    print("=" * 80)
    print(f"       CHEST X-RAY TYPE 2 DIABETES (T2D) PREDICTION SCREENER")
    print(f"       Evaluating {len(labels)} patient radiographs (0 = False, 1 = True)")
    print("=" * 80)
    print(f"{'Image Tested':<30} | {'Predicted T2D':<15} | {'Prob':<6} | {'Actual':<6} | {'Result'}")
    print("-" * 80)

    # 5-fold Stratified Group K-Fold for unbiased prediction
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_probs = np.zeros(len(labels))

    for train_idx, val_idx in sgkf.split(embeddings, labels, groups=patient_ids):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(embeddings[train_idx])
        X_val = scaler.transform(embeddings[val_idx])

        clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(X_train, labels[train_idx])
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]

    oof_preds = (oof_probs >= threshold).astype(int)

    # Print out results row by row
    correct_count = 0
    for i in range(len(labels)):
        fname = str(filenames[i])
        pred = oof_preds[i]
        prob = oof_probs[i]
        actual = labels[i]
        is_correct = (pred == actual)
        if is_correct:
            correct_count += 1

        if i < display_limit:
            pred_text = "1 (True)" if pred == 1 else "0 (False)"
            match_str = "[CORRECT]  " if is_correct else "[INCORRECT]"
            print(f"{fname:<30} | {pred_text:<15} | {prob:.2f}   | {actual}      | {match_str}")

    if display_limit < len(labels):
        print(f"... and {len(labels) - display_limit} more radiographs evaluated.")

    # Final summary score
    acc = accuracy_score(labels, oof_preds)
    auc = roc_auc_score(labels, oof_probs) if len(np.unique(labels)) > 1 else 0.0
    auprc = average_precision_score(labels, oof_probs) if len(np.unique(labels)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(labels, oof_preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("=" * 80)
    print("                     ACCURACY & PERFORMANCE SCORECARD")
    print("=" * 80)
    print(f"  Total Images Tested:          {len(labels)}")
    print(f"  Correct Predictions:          {correct_count} / {len(labels)}")
    print(f"  Accuracy Score:               {acc * 100:.2f}%")
    print(f"  AUROC (Discrimination):       {auc:.4f}")
    print(f"  AUPRC (Precision-Recall):     {auprc:.4f}")
    print(f"  Sensitivity (True Positives): {sensitivity * 100:.2f}%  ({tp}/{tp+fn} T2D patients detected)")
    print(f"  Specificity (True Negatives): {specificity * 100:.2f}%  ({tn}/{tn+fp} Non-T2D correctly cleared)")
    print(f"\n  Confusion Matrix:")
    print(f"    - True Negatives  (Actual 0, Predicted 0): {tn}")
    print(f"    - False Positives (Actual 0, Predicted 1): {fp}")
    print(f"    - False Negatives (Actual 1, Predicted 0): {fn}")
    print(f"    - True Positives  (Actual 1, Predicted 1): {tp}")
    print("=" * 80)


def test_single_image(image_path: Path, npz_path: Path, device: str = "cpu"):
    """Classify a single raw chest X-ray image."""
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print(f"\n[Screener] Analyzing image: {image_path.name}...")

    # Load trained model weights from cohort embeddings
    if not npz_path.exists():
        print(f"Error: Precomputed training embeddings not found at {npz_path}")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    scaler, clf = train_classifier_head(data["embeddings"], data["labels"])

    model = load_feature_extractor(device=device)
    emb = extract_single_embedding(image_path, model, device=device)

    emb_scaled = scaler.transform(emb.reshape(1, -1))
    prob = clf.predict_proba(emb_scaled)[0, 1]
    pred = 1 if prob >= 0.5 else 0
    pred_str = "1 (True - Likely T2D)" if pred == 1 else "0 (False - Unlikely T2D)"

    print("=" * 60)
    print(f"  IMAGE:          {image_path.name}")
    print(f"  PREDICTED T2D:  {pred_str}")
    print(f"  CONFIDENCE:     {prob * 100:.1f}% probability of T2D")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None, help="Path to single X-ray image to test")
    parser.add_argument("--embeddings", type=Path, default=Path("rad_dino_output/rad_dino_embeddings.npz"))
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cohort images to test and print")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability cutoff")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    if args.image:
        test_single_image(args.image, args.embeddings, device=args.device)
    else:
        run_cohort_evaluation(args.embeddings, limit=args.limit, threshold=args.threshold)


if __name__ == "__main__":
    main()
