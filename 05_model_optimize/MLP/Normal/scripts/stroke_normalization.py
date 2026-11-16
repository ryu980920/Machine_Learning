#!/usr/bin/env python3
"""Training-fitted, class-blind 28x28 grayscale stroke width normalization."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import skeletonize
from skimage.measure import euler_number
from sklearn.base import BaseEstimator, TransformerMixin

KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

def grayscale(images):
    a = np.asarray(images)
    if a.ndim == 2 and a.shape[1] == 784:
        a = a.reshape(-1, 28, 28)
    elif a.ndim == 2 and a.shape == (28, 28):
        a = a[None]
    if a.ndim != 3 or a.shape[1:] != (28, 28):
        raise ValueError("Requires 28x28 grayscale images")
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 255:
        raise ValueError("Pixel values outside 0..255")
    return a.astype(np.float32, copy=False)

def stroke_width(img, threshold=64):
    mask = np.asarray(img) >= threshold
    if not mask.any():
        return np.nan
    radii = distance_transform_edt(mask)[skeletonize(mask)]
    return max(1., float(2 * np.median(radii) - 1)) if len(radii) else np.nan

def topology(img, threshold=128):
    mask = np.asarray(img) >= threshold
    return (int(label(mask, structure=np.ones((3, 3)))[1]),
            int(euler_number(mask, connectivity=2)))

def center_of_mass(img, threshold=32):
    im = np.asarray(img, dtype=np.float32)
    weighted = np.where(im >= threshold, im, 0)
    mass = float(weighted.sum())
    if mass == 0:
        return im.copy()
    yy, xx = np.indices((28, 28))
    dx = int(np.rint(13.5 - float((weighted * xx).sum()) / mass))
    dy = int(np.rint(13.5 - float((weighted * yy).sum()) / mass))
    out = np.zeros_like(im)
    sy0, sy1 = max(0, -dy), min(28, 28 - dy)
    sx0, sx1 = max(0, -dx), min(28, 28 - dx)
    if sy0 < sy1 and sx0 < sx1:
        out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = im[sy0:sy1, sx0:sx1]
    return out

class StrokeWidthNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, strength=.15, threshold=64, fit_quantiles=(.2, .8),
                 preserve_topology=True, apply_center=False):
        self.strength = strength
        self.threshold = threshold
        self.fit_quantiles = fit_quantiles
        self.preserve_topology = preserve_topology
        self.apply_center = apply_center

    def fit(self, X, y=None):
        if not 0 <= self.strength <= 1:
            raise ValueError("Invalid stroke strength")
        vals = np.array([stroke_width(im, self.threshold) for im in grayscale(X)], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 20:
            raise ValueError("Not enough training strokes")
        self.q_low_, self.q_high_ = np.quantile(vals, self.fit_quantiles)
        self.median_ = float(np.median(vals))
        self.n_fitted_ = len(vals)
        return self

    def transform(self, X):
        if not hasattr(self, "q_low_"):
            raise RuntimeError("Fit on official Training only")
        src = grayscale(X)
        out = np.empty_like(src)
        for i, img in enumerate(src):
            width = stroke_width(img, self.threshold)
            updated = img.copy()
            if np.isfinite(width) and self.strength > 0:
                target = (cv2.dilate(img, KERNEL) if width < self.q_low_ else
                          cv2.erode(img, KERNEL) if width > self.q_high_ else img)
                if target is not img:
                    orig_topo = topology(img)
                    alpha = self.strength
                    for _ in range(5):
                        test = np.clip(img * (1 - alpha) + target * alpha, 0, 255)
                        if not self.preserve_topology or topology(test) == orig_topo:
                            updated = test
                            break
                        alpha /= 2
            out[i] = center_of_mass(updated) if self.apply_center else updated
        return out.reshape(len(src), 784)
