#!/usr/bin/env python3
"""EXP유004: stateless image-by-image geometric centering, label-free.

Center-of-mass and bounding-box center share:
- an identical foreground threshold (32 on raw grayscale [0,255]);
- a fixed geometric image target (13.5,13.5);
- integer-pixel translation (no interpolation), zero-fill and NO wrapping;
- no fitted parameters, class labels, writer IDs, or Handmade statistics.

This module must remain importable by its stable name for joblib Pipeline loading.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

THRESHOLD = 32.0
TARGET = (13.5, 13.5)  # (x,y), 28x28 geometric center


def raw_centers(X: np.ndarray, method: str, threshold: float = THRESHOLD):
    """Return foreground-dependent (x,y) coordinates for every source image."""
    if method not in ("mass", "bbox"):
        raise ValueError(f"Unknown method {method!r}")
    a = np.asarray(X, dtype=np.float32).reshape(-1, 28, 28)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite source image intensity")
    mask = a >= threshold
    valid = mask.any(axis=(1, 2))
    if not valid.all():
        raise ValueError(f"{int((~valid).sum())} empty foreground masks at threshold {threshold}")

    if method == "mass":
        weight = np.where(mask, a, 0).astype(np.float64, copy=False)
        total = weight.sum(axis=(1, 2))
        cx = (weight * np.arange(28, dtype=float)[None, None, :]).sum(axis=(1, 2))/total
        cy = (weight * np.arange(28, dtype=float)[None, :, None]).sum(axis=(1, 2))/total
    else:
        xpresent = mask.any(axis=1)
        ypresent = mask.any(axis=2)
        xmin = xpresent.argmax(axis=1)
        xmax = 27 - xpresent[:, ::-1].argmax(axis=1)
        ymin = ypresent.argmax(axis=1)
        ymax = 27 - ypresent[:, ::-1].argmax(axis=1)
        cx = (xmin+xmax)/2
        cy = (ymin+ymax)/2
    return cx, cy


def center_shifts(X: np.ndarray, method: str):
    cx, cy = raw_centers(X, method)
    dx = np.rint(TARGET[0] - cx).astype(np.int32)
    dy = np.rint(TARGET[1] - cy).astype(np.int32)
    return np.column_stack([dx, dy])


def center_batch(X: np.ndarray, method: str):
    a = np.asarray(X, dtype=np.float32).reshape(-1,28,28)
    shifts = center_shifts(a, method)
    result = np.zeros_like(a)
    for dx,dy in np.unique(shifts,axis=0):
        dx,dy = int(dx),int(dy)
        ids = np.flatnonzero((shifts[:,0]==dx)&(shifts[:,1]==dy))
        ys0,ys1=max(0,-dy), min(28,28-dy)
        xs0,xs1=max(0,-dx), min(28,28-dx)
        if ys0>=ys1 or xs0>=xs1:
            continue
        result[ids,ys0+dy:ys1+dy,xs0+dx:xs1+dx] = a[ids,ys0:ys1,xs0:xs1]
    return result.reshape(-1,784)


class AutomaticCentering(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = "mass"):
        self.method = method

    def fit(self, X, y=None):
        if self.method not in ("mass","bbox"):
            raise ValueError(self.method)
        return self

    def transform(self, X):
        return center_batch(X, self.method)


def test_centering():
    """Geometry, sign, data-integrity, no-wrap, input immutability checks."""
    a=np.zeros((3,28,28),dtype=np.float32)
    a[0,6:10,4:8]=255
    a[1,18:22,20:24]=255
    a[2,12:16,10:14]=255
    raw=a.copy()
    for method in ("mass","bbox"):
        out=center_batch(a,method).reshape(-1,28,28)
        before=raw_centers(a,method)
        after=raw_centers(out,method)
        for i in range(3):
            assert abs(after[0][i]-13.5)<=0.50001
            assert abs(after[1][i]-13.5)<=0.50001
            assert abs(float(out[i].sum())-float(a[i].sum()))<1e-5
        assert np.allclose(raw,a), "Input was mutated"
        assert not np.any(out < 0)
        assert np.unique(out).tolist()==[0.0,255.0]
    edge=np.zeros((1,28,28),dtype=np.float32)
    edge[0,13,0]=255
    for method in ("mass","bbox"):
        centered=center_batch(edge,method).reshape(28,28)
        assert centered[13,14]==255
        assert centered[13,0]==0
    assert np.array_equal(center_batch(a,"mass"),center_batch(a,"mass"))
    print("CENTERING SELF-TEST PASSED: label-free methods and zero-filled integer shifts",flush=True)
