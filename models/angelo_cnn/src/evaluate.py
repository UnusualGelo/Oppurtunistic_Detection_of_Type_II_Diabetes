from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from dataset import create_dataloaders
from model import DiabetesCNN


def evaluate_model():
    csv_file="patient.csv",
    image_dir="Image Folder",

    _, val_loader, _ = create_dataloaders(
        csv_file=csv_file,
        image_dir=image_dir,
        batch_size=16,
        random_state=42
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = DiabetesCNN().to(device)

    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "batchnorm_lr0001.pth"
    )

    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )

    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            all_probs.extend(
                probabilities.cpu().numpy().flatten()
            )
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels, dtype=int)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nPrediction distribution:")
    print("Min probability:", y_prob.min())
    print("Max probability:", y_prob.max())
    print("Mean probability:", y_prob.mean())

    print("\nPredicted classes:")
    print(np.bincount(y_pred, minlength=2))

    print("\nAccuracy:", accuracy_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_prob))

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        zero_division=0
    ))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="CNN")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Validation ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_model()
