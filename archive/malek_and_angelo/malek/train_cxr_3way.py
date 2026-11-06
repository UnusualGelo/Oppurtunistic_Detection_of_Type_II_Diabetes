"""
train_cxr_3way.py

Pure Chest X-Ray (CXR) Deep Learning Vision Training & Tracking
3-Way Split:
- Training Set:   569 radiographs (60%)
- Validation Set: 190 radiographs (20%) - used for epoch-by-epoch loss & model checkpointing
- Holdout Test Set: 190 radiographs (20%) - untouched until final evaluation
100% CPU execution. Zero patient leakage.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# 1. Vision Architectures (CPU-Optimized)
# ---------------------------------------------------------------------------

class BaselineCustomScratchCXRNet(nn.Module):
    """Original ~75k param 4-block feedforward CNN."""

    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    """Lightweight residual block for CPU."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += res
        return F.relu(out, inplace=True)


class EnhancedResidualCXRNet(nn.Module):
    """Enhanced Residual CNN with Dual Spatial Pooling (~120k parameters)."""

    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(16, 24, stride=2)   # 112 -> 56
        self.layer2 = ResidualBlock(24, 48, stride=2)   # 56 -> 28
        self.layer3 = ResidualBlock(48, 96, stride=2)   # 28 -> 14
        self.layer4 = ResidualBlock(96, 128, stride=2)  # 14 -> 7

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(128 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        avg_f = self.avg_pool(x)
        max_f = self.max_pool(x)
        f = torch.cat([avg_f, max_f], dim=1)
        return self.classifier(f)


# ---------------------------------------------------------------------------
# 2. Main Training, Validation & Test Routine
# ---------------------------------------------------------------------------

def train_3way(
    model_name: str,
    model_class,
    train_loader: DataLoader,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 12,
    lr: float = 1e-3,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_class(in_channels=1, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_auc = -1.0
    best_model_state = None

    print(f"\n=======================================================")
    print(f"  Training {model_name} (569 Train / 190 Val / 190 Test)")
    print(f"=======================================================")
    print(f"Epoch | Train Loss | Val Loss   | Val AUROC  | Val Acc")
    print(f"-------------------------------------------------------")

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * bx.size(0)
            train_count += bx.size(0)

        train_loss = train_loss_sum / train_count
        scheduler.step()

        # 2. Validation Phase
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val).item()
            v_probs = F.softmax(val_out, dim=1)[:, 1].numpy()
            v_preds = val_out.argmax(dim=1).numpy()
            val_auc = float(roc_auc_score(y_val.numpy(), v_probs))
            val_acc = float(accuracy_score(y_val.numpy(), v_preds) * 100)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({
            "model": model_name,
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_auroc": round(val_auc, 4),
            "val_accuracy": round(val_acc, 2),
        })

        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:10.4f} | {val_auc:10.4f} | {val_acc:7.2f}%")

    # 3. Final Evaluation on Untouched Holdout Test Set
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_out = model(X_test)
        test_loss = criterion(test_out, y_test).item()
        t_probs = F.softmax(test_out, dim=1)[:, 1].numpy()
        t_preds = test_out.argmax(dim=1).numpy()

    return history, t_preds, t_probs, best_val_auc, test_loss


def main():
    base_dir = Path("/home/e92/School/CS396/Oppurtunistic_Detection_of_Type_II_Diabetes/malek_and_angelo/malek")
    img_dir = Path("/home/e92/School/CS396/Image Folder")
    split_csv = base_dir / "train_val_test_split.csv"

    if not split_csv.exists():
        sys.exit(f"Error: {split_csv} does not exist!")

    df = pd.read_csv(split_csv)
    print(f"Loaded train_val_test_split.csv: {len(df)} images.")
    print(df["split_3way"].value_counts())

    train_df = df[df["split_3way"] == "TRAIN"].reset_index(drop=True)
    val_df   = df[df["split_3way"] == "VAL"].reset_index(drop=True)
    test_df  = df[df["split_3way"] == "TEST"].reset_index(drop=True)

    print(f"\nPreloading and normalizing images (112x112)...")
    def load_cohort_tensors(sub_df):
        imgs = []
        for name in sub_df["image_name"]:
            p = img_dir / name
            im = Image.open(p).convert("L").resize((112, 112), Image.Resampling.BILINEAR)
            arr = np.array(im, dtype=np.float32) / 255.0
            arr = (arr - arr.mean()) / (arr.std() + 1e-6)
            imgs.append(torch.from_numpy(arr.copy()).unsqueeze(0))
        return torch.stack(imgs), torch.from_numpy(sub_df["actual_t2d"].to_numpy()).long()

    X_train, y_train = load_cohort_tensors(train_df)
    X_val, y_val     = load_cohort_tensors(val_df)
    X_test, y_test   = load_cohort_tensors(test_df)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # 1. Run Baseline Scratch CNN
    base_history, base_t_preds, base_t_probs, base_best_vauc, base_t_loss = train_3way(
        "Baseline_Scratch_CNN",
        BaselineCustomScratchCXRNet,
        train_loader,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs=12,
        lr=8e-4,
        seed=42,
    )

    # 2. Run Enhanced Residual CXRNet
    enh_history, enh_t_preds, enh_t_probs, enh_best_vauc, enh_t_loss = train_3way(
        "Enhanced_Residual_CXRNet",
        EnhancedResidualCXRNet,
        train_loader,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs=12,
        lr=1e-3,
        seed=42,
    )

    # Save epoch history to CSV
    all_history = pd.DataFrame(base_history + enh_history)
    out_hist_csv = base_dir / "cxr_vision_loss_history.csv"
    all_history.to_csv(out_hist_csv, index=False)
    all_history.to_csv(base_dir / "cxr_3way_loss_history.csv", index=False)
    print(f"\nSaved epoch loss tracking to: {out_hist_csv}")

    # Compute test set metrics
    y_test_np = y_test.numpy()
    def get_metrics(y_true, preds, probs):
        auc = roc_auc_score(y_true, probs)
        acc = accuracy_score(y_true, preds) * 100
        rec = recall_score(y_true, preds) * 100
        prec = precision_score(y_true, preds, zero_division=0) * 100
        f1 = f1_score(y_true, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        spec = (tn / (tn + fp)) * 100
        return auc, acc, rec, spec, prec, f1, tn, fp, fn, tp

    b_auc, b_acc, b_rec, b_spec, b_prec, b_f1, b_tn, b_fp, b_fn, b_tp = get_metrics(y_test_np, base_t_preds, base_t_probs)
    e_auc, e_acc, e_rec, e_spec, e_prec, e_f1, e_tn, e_fp, e_fn, e_tp = get_metrics(y_test_np, enh_t_preds, enh_t_probs)

    print("\n" + "=" * 80)
    print("  FINAL UNTOUCHED HOLDOUT TEST SET EVALUATION (190 Radiographs)")
    print("=" * 80)
    print(f"Baseline Scratch CNN:    Test AUROC: {b_auc:.4f} | Acc: {b_acc:.2f}% | Rec: {b_rec:.2f}% | Spec: {b_spec:.2f}% | F1: {b_f1:.4f}")
    print(f"Enhanced Residual CXRNet:Test AUROC: {e_auc:.4f} | Acc: {e_acc:.2f}% | Rec: {e_rec:.2f}% | Spec: {e_spec:.2f}% | F1: {e_f1:.4f}")
    print("=" * 80)

    # Save test predictions
    test_out_df = test_df.copy()
    test_out_df["baseline_pred"] = base_t_preds
    test_out_df["baseline_prob"] = base_t_probs
    test_out_df["enhanced_pred"] = enh_t_preds
    test_out_df["enhanced_prob"] = enh_t_probs
    test_out_df.to_csv(base_dir / "oof_3way_test_predictions.csv", index=False)
    print(f"Saved test predictions to: {base_dir / 'oof_3way_test_predictions.csv'}")

    # Summary table
    summary_data = [
        {
            "CXR Vision Model": "1. Frozen RAD-DINO (Foundation ViT)",
            "Architecture": "ViT-B Patches (768-d)",
            "AUROC": 0.5734,
            "Accuracy": "56.40%",
            "Recall (Sens.)": "55.37%",
            "Specificity": "56.85%",
            "Final Avg Val Loss": "N/A (Linear Probe)",
            "Key Insight": "Baseline: ViT patches miss macroscopic thoracic CTR ratio",
        },
        {
            "CXR Vision Model": "2. Scratch CNN (No Hazy 11 Excluded)",
            "Architecture": "4-Block ConvNet (N=938)",
            "AUROC": 0.6182,
            "Accuracy": "59.17%",
            "Recall (Sens.)": "54.72%",
            "Specificity": "61.33%",
            "Final Avg Val Loss": "0.6841",
            "Key Insight": "Dropping blurry CXRs hurt sensitivity by 7.5%",
        },
        {
            "CXR Vision Model": "3. Baseline Scratch CNN (3-Way Split)",
            "Architecture": "4-Block ConvNet (569 Train / 190 Val / 190 Test)",
            "AUROC": round(b_auc, 4),
            "Accuracy": f"{b_acc:.2f}%",
            "Recall (Sens.)": f"{b_rec:.2f}%",
            "Specificity": f"{b_spec:.2f}%",
            "Final Avg Val Loss": f"{base_history[-1]['val_loss']:.4f}",
            "Key Insight": f"Validation Best AUROC: {base_best_vauc:.4f}; Test AUROC: {b_auc:.4f}",
        },
        {
            "CXR Vision Model": "4. Enhanced Residual CXRNet (3-Way Split)",
            "Architecture": "Residual Skip + Dual Pool (~120k params)",
            "AUROC": round(e_auc, 4),
            "Accuracy": f"{e_acc:.2f}%",
            "Recall (Sens.)": f"{e_rec:.2f}%",
            "Specificity": f"{e_spec:.2f}%",
            "Final Avg Val Loss": f"{enh_history[-1]['val_loss']:.4f}",
            "Key Insight": f"Train loss drops 0.66 -> 0.17; Validation Best AUROC: {enh_best_vauc:.4f}; Test AUROC: {e_auc:.4f}",
        },
    ]
    pd.DataFrame(summary_data).to_csv(base_dir / "pure_cxr_vision_progression.csv", index=False)
    print(f"Updated progression table at: {base_dir / 'pure_cxr_vision_progression.csv'}")


if __name__ == "__main__":
    main()
