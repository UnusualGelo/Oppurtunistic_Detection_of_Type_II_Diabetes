"""Frontal-vs-lateral chest X-ray view classifier (hand-crafted features + random forest)."""
import os, pickle
import cv2, numpy as np
from sklearn.ensemble import RandomForestClassifier

def _mirror(a, axis):
    """Best mirror-correlation allowing the symmetry axis to sit off-centre."""
    n = a.shape[axis]; best = -1.0
    for s in range(-12, 13, 2):
        x = a[:, max(0, s):n + min(0, s)] if axis == 1 else a[max(0, s):n + min(0, s)]
        x = x - x.mean(); x = x / (x.std() + 1e-6)
        m = x[:, ::-1] if axis == 1 else x[::-1]
        best = max(best, float((x * m).mean()))
    return best

def features(img):
    g = cv2.resize(img, (128, 128)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.5); g = (g - g.mean()) / (g.std() + 1e-6)
    full, cen = _mirror(g, 1), _mirror(g[:, 20:108], 1)
    top, bot = _mirror(g[:64], 1), _mirror(g[64:], 1)
    ud = _mirror(g, 0)                       # top/bottom symmetry -> rotated-frontal cue
    col, row = g.mean(0), g.mean(1)
    return np.r_[full, cen, top, bot, ud, col.std(), row.std(), np.abs(np.diff(col)).mean(),
                 cv2.resize(col[None], (16, 1)).ravel(), cv2.resize(row[None], (16, 1)).ravel(),
                 cv2.resize(g, (16, 16)).ravel()]

def train(X, y):
    return RandomForestClassifier(500, random_state=0, class_weight="balanced", n_jobs=-1).fit(X, y)

def save(model, path):
    with open(path, "wb") as f: pickle.dump(model, f)

def load(path):
    with open(path, "rb") as f: return pickle.load(f)
