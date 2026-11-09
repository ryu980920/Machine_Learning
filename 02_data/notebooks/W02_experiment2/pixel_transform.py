"""Reusable 0-255 -> 0-1 transformer for sklearn pipelines saved by joblib."""
from sklearn.base import BaseEstimator, TransformerMixin

class PixelScaler255(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X / 255.0
