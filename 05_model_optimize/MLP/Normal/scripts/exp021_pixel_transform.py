"""Exact upstream transformer: 03_experiments/W02_2026-11-09_11-15/scripts/experiment2_baseline/pixel_transform.py"""
from sklearn.base import BaseEstimator, TransformerMixin
class PixelScaler255(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X / 255.0
