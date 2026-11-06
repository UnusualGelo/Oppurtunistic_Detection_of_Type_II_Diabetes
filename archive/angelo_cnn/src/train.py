from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders
from model import DiabetesCNN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model():
    set_seed()

    # Update these paths to your working dataset paths
    csv_file="patient.csv",
    image_dir="Image Folder",

    train_loader, val_loader, test_loader = create_dataloaders(
        csv_file=csv_file,
        image_dir=image_dir,
        batch_size=16,
        random_state=42
    )

    # Select available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Training on:", device)

    model = DiabetesCNN().to(device)

    # Calculate class weight from training data only
    train_labels = train_loader.dataset.data["T2D"]
    negative = (train_labels == 0).sum()
    positive = (train_labels == 1).sum()

    pos_weight = torch.tensor(
        [negative / positive],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    epochs = 30
    patience = 5
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_weights = None

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += (
                    loss.item() * images.size(0)
                )

        val_loss = running_val_loss / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered")
            break

    # Restore best model
    model.load_state_dict(best_weights)

    # Save checkpoint and loss history
    output_dir = (
        Path(__file__).resolve().parents[1] / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        output_dir / "batchnorm_lr0001.pth"
    )

    np.savez(
        output_dir / "training_history.npz",
        train_loss=history["train_loss"],
        val_loss=history["val_loss"]
    )

    print("Best validation loss:", round(best_val_loss, 4))
    print("Model saved to:", output_dir / "best_model.pth")

    return model, test_loader, history


if __name__ == "__main__":
    train_model()
