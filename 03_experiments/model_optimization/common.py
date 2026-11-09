"""Exact dataset partition and scaler matching W02 EXP-유-003."""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "W02_2026-11-09_11-15" / "scripts" / "experiment2_baseline"))
from pixel_transform import PixelScaler255

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "02_data" / "datasets" / "MNIST_EMNIST_official_split_112000.npz"
EXPECTED_SHA = "0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879"
LABELS = "0123456789ABCDEF"

def fixed_split(path=DEFAULT_DATA):
    """Official X_train only; replicate baseline's first StratifiedGroupKFold."""
    path = Path(path)
    with path.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    if sha != EXPECTED_SHA:
        raise ValueError("Dataset SHA-256 mismatch: " + sha)
    with np.load(path, allow_pickle=False) as d:
        X = d["X_train"]
        y = d["y_train"].astype(np.int64)
        src = d["train_source_index"].astype(np.int64)
        split = d["train_source_split"].astype(np.int64)
        aug = d["train_is_augmented"]
    if X.shape != (96000, 28, 28) or int(aug.sum()) != 5349:
        raise ValueError("Unexpected data/augmentation count")
    groups = y * 1_000_000_000 + split * 100_000_000 + src
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr, va = next(cv.split(X, y, groups))
    if (len(tr), len(va)) != (76800, 19200):
        raise ValueError("Baseline split sizes not reproduced")
    if np.intersect1d(groups[tr], groups[va]).size:
        raise ValueError("Leakage across original/augmentation groups")
    return (X[tr].reshape(-1, 784).astype(np.float32), y[tr],
            X[va].reshape(-1, 784).astype(np.float32), y[va], sha)

def pipeline(model, scaled):
    return Pipeline([("scale_0_1", PixelScaler255()), ("model", model)]) if scaled else model

def scores(y_true, y_pred):
    ids = np.arange(16)
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=ids, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
        "per_class": {c: {"precision":float(p[i]),"recall":float(r[i]),"f1":float(f[i]),"support":int(n[i])} for i,c in enumerate(LABELS)}
    }
