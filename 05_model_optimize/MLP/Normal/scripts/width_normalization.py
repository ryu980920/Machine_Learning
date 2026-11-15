#!/usr/bin/env python3
"""Training-fitted, class-blind glyph horizontal extent normalization."""
from __future__ import annotations
import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from stroke_normalization import grayscale, center_of_mass, topology

def bounds(img, threshold=64):
    yy, xx = np.where(np.asarray(img) >= threshold)
    if len(xx) == 0:
        return None
    return int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max())

class GlyphWidthNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, strength=.15, threshold=64, quantiles=(.2, .8),
                 preserve_topology=True, apply_center=False):
        self.strength = strength
        self.threshold = threshold
        self.quantiles = quantiles
        self.preserve_topology = preserve_topology
        self.apply_center = apply_center

    def fit(self, X, y=None):
        if not 0 <= self.strength <= 1:
            raise ValueError("Invalid width strength")
        widths = []
        for im in grayscale(X):
            b = bounds(im, self.threshold)
            if b is not None:
                widths.append(b[1] - b[0] + 1)
        if len(widths) < 20:
            raise ValueError("Not enough training glyphs")
        self.q_low_, self.q_high_ = np.quantile(widths, self.quantiles)
        self.median_ = float(np.median(widths))
        self.n_fitted_ = len(widths)
        return self

    def transform(self, X):
        if not hasattr(self, "q_low_"):
            raise RuntimeError("Fit on official Training only")
        src = grayscale(X)
        out = np.empty_like(src)
        for i, img in enumerate(src):
            b = bounds(img, self.threshold)
            result = img.copy()
            if b is not None and self.strength > 0:
                w = b[1] - b[0] + 1
                goal = self.q_low_ if w < self.q_low_ else self.q_high_ if w > self.q_high_ else w
                if goal != w:
                    scale = float(np.clip(1 + self.strength * (goal / w - 1), .85, 1.15))
                    cx = (b[0] + b[1]) / 2
                    before = topology(img)
                    for _ in range(5):
                        matrix = np.array([[scale, 0, (1 - scale) * cx], [0, 1, 0]], np.float32)
                        changed = cv2.warpAffine(img, matrix, (28, 28),
                                                 flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        if not self.preserve_topology or topology(changed) == before:
                            result = changed
                            break
                        scale = 1 + (scale - 1) / 2
            out[i] = center_of_mass(result) if self.apply_center else result
        return out.reshape(len(src), 784)
