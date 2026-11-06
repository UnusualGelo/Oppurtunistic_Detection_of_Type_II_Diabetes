
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.data = dataframe.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = self.image_dir / row["image_name"]

        with Image.open(image_path) as img:
            image = img.convert("L")
            image = self.transform(image)

        label = torch.tensor(
            float(row["T2D"]),
            dtype=torch.float32
        )

        return image, label


def load_clinical_data(csv_file, image_dir):
    df = pd.read_csv(csv_file)

    # Check required columns
    required = ["image_name", "T2D"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    # Remove records without image filenames or labels
    df = df.dropna(subset=required).copy()

    # Validate diabetes labels
    if not df["T2D"].isin([0, 1]).all():
        raise ValueError("T2D labels must be 0 or 1")

    df["T2D"] = df["T2D"].astype(int)
    df["image_name"] = df["image_name"].astype(str)

    # Check duplicate filenames
    if df["image_name"].duplicated().any():
        raise ValueError("Duplicate image filenames found")

    # Verify images exist
    image_dir = Path(image_dir)

    missing = [
        name for name in df["image_name"]
        if not (image_dir / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images missing. "
            f"Examples: {missing[:5]}"
        )

    print(f"Valid images: {len(df)}")
    print("Class distribution:")
    print(df["T2D"].value_counts().sort_index())

    return df


def create_dataloaders(
    csv_file,
    image_dir,
    batch_size=16,
    random_state=42
):
    df = load_clinical_data(csv_file, image_dir)

    # 70% training, 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["T2D"],
        random_state=random_state
    )

    # Split remaining 30% into 15% validation,
    # 15% testing
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["T2D"],
        random_state=random_state
    )

    train_dataset = ChestXrayDataset(train_df, image_dir)
    val_dataset = ChestXrayDataset(val_df, image_dir)
    test_dataset = ChestXrayDataset(test_df, image_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print("\nDataset splits:")
    for name, split in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Testing", test_df)
    ]:
        print(f"{name}: {len(split)} images")
        print(split["T2D"].value_counts().sort_index())

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = (
        create_dataloaders(
            csv_file="patient.csv",
            image_dir="Image Folder",
            batch_size=16
        )
    )

    images, labels = next(iter(train_loader))

    print("\nBatch test:")
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Labels:", labels)
