
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset

from dataset import create_dataloaders
from model import DiabetesCNN


def main():
    train_loader, _, _ = create_dataloaders(
        csv_file="patient.csv",
        image_dir="Image Folder",
        batch_size=16,
        random_state=42
    )

    # Use a fixed subset of 32 training images
    dataset = Subset(
        train_loader.dataset,
        list(range(32))
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = DiabetesCNN().to(device)

    # Unweighted loss for debugging
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    for epoch in range(200):
        model.train()

        total_loss = 0.0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0

            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    predictions = (
                        torch.sigmoid(outputs).squeeze(1) >= 0.5
                    ).float()

                    correct += (
                        predictions == labels
                    ).sum().item()

            accuracy = correct / len(dataset)

            print(
                f"Epoch {epoch + 1} | "
                f"Loss: {total_loss / len(dataset):.4f} | "
                f"Training Accuracy: {accuracy:.2%}"
            )


if __name__ == "__main__":
    main()
