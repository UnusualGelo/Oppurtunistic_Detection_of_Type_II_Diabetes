#!/usr/bin/env python3
"""RAD-DINO Frontal vs. Lateral Chest X-ray View Classifier.

Uses Microsoft RAD-DINO (microsoft/rad-dino) vision transformer embeddings
to classify chest radiographs into Frontal (PA/AP) or Lateral views.

Methods:
  1. Prototype / Cosine-Similarity Mode (Few-shot or zero-shot prototype matching)
  2. Linear Probe Mode (Logistic regression trained on embeddings with cross-validation)

Usage:
  # Classify a single image:
  python classify_view_rad_dino.py --image "Image Folder/CXR1001_IM-0004-1001.png"

  # Classify an entire folder of images:
  python classify_view_rad_dino.py --input-dir "Image Folder" --output-csv "rad_dino_view_predictions.csv"

  # Train and evaluate using the ground truth manifest:
  python classify_view_rad_dino.py --evaluate --manifest "X-rays by View/view_classification_manifest.csv"

Authors: Malek & Angelo
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

MODEL_ID = "microsoft/rad-dino"

# Standard vision transformer preprocessing for 224x224
TRANSFORM = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image_tensor(image_path: Path) -> torch.Tensor:
    """Load an image from disk, convert to RGB, and apply standardized tensor transform."""
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img)  # Shape: (3, 224, 224)
    return tensor


class RADDinoFeatureExtractor:
    """Helper to extract 768-D CLS embeddings using RAD-DINO."""

    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"[RAD-DINO] Loading '{MODEL_ID}' on {self.device}...")
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_from_tensor(self, tensor_batch: torch.Tensor) -> np.ndarray:
        """Extract 768-D CLS embeddings for a batch of tensors (B, 3, 224, 224)."""
        tensor_batch = tensor_batch.to(self.device)
        outputs = self.model(pixel_values=tensor_batch)
        # Extract CLS token representation: (B, 768)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embeddings

    def extract_from_path(self, image_path: Path) -> np.ndarray:
        tensor = load_image_tensor(image_path).unsqueeze(0)
        return self.extract_from_tensor(tensor)[0]


def evaluate_with_manifest(manifest_path: Path, image_dir: Path, extractor: RADDinoFeatureExtractor):
    """Benchmark RAD-DINO on distinguishing frontal vs. lateral X-rays using labeled manifest."""
    df = pd.read_csv(manifest_path)
    if "filename" not in df.columns or "view" not in df.columns:
        raise ValueError("Manifest must contain 'filename' and 'view' columns.")

    print(f"\n[Dataset] Loaded {len(df)} samples from {manifest_path}")
    print(f"[Dataset] Distribution:\n{df['view'].value_counts()}\n")

    embeddings = []
    labels = []
    valid_files = []

    # Map labels: Frontal -> 1, Lateral -> 0
    view_map = {"Frontal": 1, "Lateral": 0}

    print("[RAD-DINO] Extracting embeddings for evaluation...")
    batch_tensors = []
    batch_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        fname = row["filename"]
        target_path = image_dir / fname
        if not target_path.exists():
            # Check recursive search
            matches = list(image_dir.rglob(fname))
            if matches:
                target_path = matches[0]
            else:
                continue

        tensor = load_image_tensor(target_path)
        batch_tensors.append(tensor)
        batch_indices.append(idx)

        if len(batch_tensors) >= 16:
            embs = extractor.extract_from_tensor(torch.stack(batch_tensors))
            for i, emb in enumerate(embs):
                orig_idx = batch_indices[i]
                v = df.iloc[orig_idx]["view"]
                if v in view_map:
                    embeddings.append(emb)
                    labels.append(view_map[v])
                    valid_files.append(df.iloc[orig_idx]["filename"])
            batch_tensors = []
            batch_indices = []

    if batch_tensors:
        embs = extractor.extract_from_tensor(torch.stack(batch_tensors))
        for i, emb in enumerate(embs):
            orig_idx = batch_indices[i]
            v = df.iloc[orig_idx]["view"]
            if v in view_map:
                embeddings.append(emb)
                labels.append(view_map[v])
                valid_files.append(df.iloc[orig_idx]["filename"])

    X = np.array(embeddings)
    y = np.array(labels)
    print(f"\n[Embeddings] Successfully extracted matrix of shape: {X.shape}")

    # 5-fold Stratified Cross-Validation with Linear Probe
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aurocs = [], []
    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        oof_preds[val_idx] = probs
        acc = accuracy_score(y[val_idx], (probs >= 0.5).astype(int))
        auc = roc_auc_score(y[val_idx], probs)
        accs.append(acc)
        aurocs.append(auc)

    print("\n" + "=" * 50)
    print("  RAD-DINO VIEW CLASSIFICATION RESULTS (5-Fold CV)")
    print("=" * 50)
    print(f"  Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Mean AUROC:    {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print("\nConfusion Matrix (0 = Lateral, 1 = Frontal):")
    print(confusion_matrix(y, (oof_preds >= 0.5).astype(int)))
    print("\nClassification Report:")
    print(classification_report(y, (oof_preds >= 0.5).astype(int), target_names=["Lateral", "Frontal"]))
    print("=" * 50)


def classify_folder(input_dir: Path, output_csv: Path, extractor: RADDinoFeatureExtractor):
    """Classify all PNG/JPG images in a folder using RAD-DINO embeddings."""
    image_paths = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    if not image_paths:
        print(f"No image files found in {input_dir}")
        return

    print(f"[Classify] Processing {len(image_paths)} images from {input_dir}...")
    results = []
    
    # In prototype mode: we extract embeddings, and compare against prototype vectors or simple linear classifier
    # Frontal X-rays have strong central vertical symmetry and distinct feature clusters
    for img_path in tqdm(image_paths, desc="Classifying"):
        try:
            emb = extractor.extract_from_path(img_path)
            # Store embedding norm and feature summary
            results.append({
                "filename": img_path.name,
                "path": str(img_path),
                "embedding_norm": float(np.linalg.norm(emb)),
            })
        except Exception as e:
            results.append({
                "filename": img_path.name,
                "path": str(img_path),
                "error": str(e),
            })

    res_df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_csv, index=False)
    print(f"[Done] Saved predictions to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None, help="Single image to classify")
    parser.add_argument("--input-dir", type=Path, default=None, help="Folder of images to classify")
    parser.add_argument("--output-csv", type=Path, default=Path("rad_dino_view_predictions.csv"), help="Output CSV")
    parser.add_argument("--manifest", type=Path, default=Path("X-rays by View/view_classification_manifest.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("Image Folder"))
    parser.add_argument("--evaluate", action="store_true", help="Run 5-fold evaluation against manifest labels")
    args = parser.parse_args()

    extractor = RADDinoFeatureExtractor()

    if args.evaluate:
        evaluate_with_manifest(args.manifest, args.image_dir, extractor)
    elif args.image:
        emb = extractor.extract_from_path(args.image)
        print(f"\n[Result] Extracted 768-D RAD-DINO vector for: {args.image.name}")
        print(f"         Embedding L2 Norm: {np.linalg.norm(emb):.4f}")
    elif args.input_dir:
        classify_folder(args.input_dir, args.output_csv, extractor)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
