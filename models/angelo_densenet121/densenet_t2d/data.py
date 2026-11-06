"""Image loading, deterministic preprocessing, and train-only augmentation.

Deterministic preprocessing (identical for train/val/test, cached to disk):
    grayscale -> robust 16-bit handling -> optional CLAHE -> aspect-preserving
    resize of the long side -> zero-pad to a square canvas.

Random augmentation (training only):
    small rotation / translation / scale, brightness & contrast jitter.
    NO horizontal flips: heart and aortic knob are on the left in a frontal CXR,
    and those are plausible T2D-related signals.

Normalization lives inside each model (see models.py), so the dataset always
returns 1xHxW float tensors in [0, 1].
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

try:
    import cv2
except ImportError:  # CLAHE becomes unavailable, everything else works
    cv2 = None


def load_grayscale(path: Path) -> np.ndarray:
    """Return a uint8 HxW array. 16-bit PNGs are rescaled with 0.5/99.5 percentiles."""
    with Image.open(path) as img:
        if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.asarray(img, dtype=np.float32)
            lo, hi = np.percentile(arr, [0.5, 99.5])
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
            return arr.astype(np.uint8)
        return np.asarray(img.convert("L"), dtype=np.uint8)


def letterbox(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = np.asarray(Image.fromarray(arr).resize((nw, nh), Image.Resampling.BILINEAR))
    canvas = np.zeros((size, size), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess(path: Path, size: int, clahe: bool, clahe_clip: float = 2.0) -> np.ndarray:
    arr = load_grayscale(path)
    if clahe:
        if cv2 is None:
            raise ImportError("CLAHE requested but opencv-python is not installed")
        arr = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(arr)
    return letterbox(arr, size)


def build_cache(names, image_dir, size, clahe, cache_dir) -> np.ndarray:
    """Preprocess every image once and store uint8 arrays (N, H, W) on disk."""
    key = json.dumps({"names": sorted(names), "dir": str(Path(image_dir).resolve()),
                      "size": size, "clahe": clahe}, sort_keys=True)
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    cache_file = Path(cache_dir) / f"images_{size}_{'clahe' if clahe else 'raw'}_{digest}.npz"
    order = sorted(names)
    if cache_file.exists():
        stored = np.load(cache_file, allow_pickle=False)
        arrays = stored["images"]
    else:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"[data] preprocessing {len(order)} images -> {cache_file.name}")
        arrays = np.stack([preprocess(Path(image_dir) / n, size, clahe) for n in order])
        np.savez_compressed(cache_file, images=arrays)
    index = {n: i for i, n in enumerate(order)}
    return arrays[[index[n] for n in names]]


def make_train_transform(cfg: dict):
    aug = cfg.get("augment", {})
    if not aug.get("enabled", True):
        return None
    return v2.Compose([
        v2.RandomAffine(
            degrees=aug.get("degrees", 7),
            translate=(aug.get("translate", 0.06),) * 2,
            scale=(1 - aug.get("scale", 0.08), 1 + aug.get("scale", 0.08)),
        ),
        v2.ColorJitter(brightness=aug.get("brightness", 0.15), contrast=aug.get("contrast", 0.15)),
    ])


class CXRDataset(Dataset):
    def __init__(self, images_uint8: np.ndarray, labels, transform=None):
        self.images = torch.from_numpy(np.ascontiguousarray(images_uint8)).unsqueeze(1)  # (N,1,H,W) uint8
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i].float().div_(255.0)
        if self.transform is not None:
            x = self.transform(x).clamp_(0, 1)
        return x, self.labels[i]


def load_split_table(splits_csv) -> pd.DataFrame:
    df = pd.read_csv(splits_csv)
    need = {"image_name", "patient_id", "label", "split", "fold"}
    if not need <= set(df.columns):
        raise ValueError(f"{splits_csv} missing columns {need - set(df.columns)}")
    return df
