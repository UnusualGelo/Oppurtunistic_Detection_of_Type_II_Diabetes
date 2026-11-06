"""Metrics, thresholds, and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve


def threshold_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {"threshold": float(thr), "accuracy": (tp + tn) / len(y), "sensitivity": sens,
            "specificity": spec, "precision": prec, "f1": f1,
            "balanced_accuracy": (sens + spec) / 2,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def ranking_metrics(y, p):
    return {"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "brier": float(np.mean((p - y) ** 2)), "n": int(len(y)), "n_pos": int(y.sum())}


def choose_threshold(y, p, method="youden", target_sensitivity=0.8):
    """Pick a threshold on DEVELOPMENT predictions only; then freeze it for test."""
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.clip(thr, 0, 1)
    if method == "youden":
        return float(thr[np.argmax(tpr - fpr)])
    if method == "sensitivity":
        ok = np.where(tpr >= target_sensitivity)[0]
        return float(thr[ok[0]])  # highest threshold reaching the target sensitivity
    if method == "fixed":
        return 0.5
    raise ValueError(method)


def bootstrap_ci(y, p, fn, n_boot=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    stats = []
    for _ in range(n_boot):  # stratified resampling keeps prevalence fixed
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats.append(fn(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_auc_diff(y, p_a, p_b, n_boot=2000, seed=0):
    """AUC(b) - AUC(a) on the same images, with 95% CI and a two-sided p-value."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(roc_auc_score(y[idx], p_b[idx]) - roc_auc_score(y[idx], p_a[idx]))
    diffs = np.asarray(diffs)
    observed = roc_auc_score(y, p_b) - roc_auc_score(y, p_a)
    p_val = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"auc_diff": float(observed), "ci95": [float(lo), float(hi)], "p_value": float(p_val)}


def full_report(y, p, thresholds: dict, n_boot=2000):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    out = ranking_metrics(y, p)
    out["roc_auc_ci95"] = bootstrap_ci(y, p, roc_auc_score, n_boot)
    out["average_precision_ci95"] = bootstrap_ci(y, p, average_precision_score, n_boot)
    out["prevalence"] = float(y.mean())
    out["at_threshold"] = {name: threshold_metrics(y, p, t) for name, t in thresholds.items()}
    return out
