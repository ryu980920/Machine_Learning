#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen KNN Normal model from EXP-014 / EXP-015.

From the repository root:
  python 05_model_optimize/KNN/Normal/scripts/final_knn_normal.py
  python 05_model_optimize/KNN/Normal/scripts/final_knn_normal.py --evaluate-test

Default: reproduce official development Validation only; official Test not read.
--evaluate-test: first reproduce Validation, then evaluate held-out official Test once.
The 76,800-row development fit split is retained for exact reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'
sys.path.insert(0, str(ROOT / '03_experiments' / 'model_optimization'))
sys.path.insert(0, str(SCRIPTS))

from common import LABELS, EXPECTED_SHA, DEFAULT_DATA, fixed_split, scores
from EXP004_KNN_Normal_CoM import BatchedCoM
from EXP007_KNN_Normal_Augmentation import (
    SEED, sample_balanced, size_jitter, stroke_jitter, save_csv, save_cm,
)
from EXP011_KNN_Normal_Deskew import deskew
from EXP012_KNN_Normal_HOG import hog_features, hybrid_features
from EXP014_KNN_Normal_Targeted_Augmentation import select_extra

OUT = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'final_knn_normal'
EXPECTED_VALIDATION_CORRECT = 18851
WEAK_LABELS = ('D', 'B', '5', '3')
BASE_FRACTION = 0.30
STROKE_STRENGTH = 0.44
PIXEL_WEIGHT = 0.20
ADDITIONAL_COUNT = 7680


def prepare_features(images, com, name):
    print(f'Preprocessing {name} ({len(images)} rows)...', flush=True)
    centered = com.transform(images)
    corrected, _ = deskew(centered)
    del centered
    h = hog_features(corrected)
    result = hybrid_features(corrected, h, pixel_weight=PIXEL_WEIGHT)
    del corrected, h
    if result.shape != (len(images), 1108) or not np.isfinite(result).all():
        raise RuntimeError(f'Invalid {name} features: {result.shape}')
    return result


def assemble_training(xtr, ytr, com):
    # EXACT EXP-014 targeted_10: the source 76,800 training split is unchanged.
    selected = sample_balanced(ytr, fraction=BASE_FRACTION, seed=SEED)
    sized, _ = size_jitter(xtr, selected, seed=SEED)
    baseline_aug, _ = stroke_jitter(sized, strength=STROKE_STRENGTH)
    del sized
    base_x = np.concatenate([xtr, baseline_aug], axis=0)
    base_y = np.concatenate([ytr, ytr[selected]], axis=0)
    del baseline_aug
    if len(base_y) != 99840:
        raise RuntimeError('Base augmentation size mismatch')
    base_features = prepare_features(base_x, com, 'base official Training')
    del base_x

    weak_ids = [LABELS.index(s) for s in WEAK_LABELS]
    extra_indices = select_extra(ytr, selected, ADDITIONAL_COUNT,
                                 weak_ids, seed=SEED + 142)
    # This is deliberately the EXP-014 `augmented_features` process.
    extra_sized, _ = size_jitter(xtr, extra_indices, seed=SEED + 140)
    extra_stroke, _ = stroke_jitter(extra_sized, strength=STROKE_STRENGTH)
    del extra_sized
    extra_features = prepare_features(extra_stroke, com, 'targeted official Training')
    del extra_stroke

    train_features = np.concatenate([base_features, extra_features], axis=0)
    train_labels = np.concatenate([base_y, ytr[extra_indices]], axis=0)
    del base_features, extra_features
    if train_features.shape != (107520, 1108):
        raise RuntimeError(f'Unexpected training features: {train_features.shape}')
    if tuple(np.bincount(ytr[extra_indices], minlength=16)[weak_ids]) != (1920,) * 4:
        raise RuntimeError('Targeted class augmentation count mismatch')
    return train_features, train_labels


def verified_official_test():
    # Never call this function during validation-only runs.
    path = Path(DEFAULT_DATA)
    with path.open('rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if sha != EXPECTED_SHA:
        raise RuntimeError(f'Official dataset digest mismatch: {sha}')
    with np.load(path, allow_pickle=False) as d:
        images = d['X_test']
        labels = d['y_test'].astype(np.int64)
    if images.shape != (16000, 28, 28) or labels.shape != (16000,):
        raise RuntimeError(f'Unexpected official Test shape: {images.shape}, {labels.shape}')
    if not np.isin(labels, np.arange(16)).all():
        raise RuntimeError('Unexpected Test labels')
    return images.reshape(-1, 784).astype(np.float32), labels, sha


def write_evaluation(prefix, labels, predictions, data_sha):
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = scores(labels, predictions)
    count = int(np.sum(labels == predictions))
    save_cm(OUT / f'{prefix}_confusion_matrix.csv', metrics['confusion_matrix'])
    save_csv(OUT / f'{prefix}_per_class.csv', [
        {'class': label, **metrics['per_class'][label]} for label in LABELS
    ])
    result = {
        'name': prefix, 'samples': int(len(labels)), 'correct': count,
        'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
        'dataset_sha256': data_sha, 'training_rows': 107520,
        'model': {'name': 'KNeighborsClassifier', 'k': 3,
                  'weights': 'distance', 'metric': 'euclidean'},
        'preprocessing': 'CoM -> Deskew', 'features': 'Pixel 20% + HOG 80%',
        'base_augmentation': {'fraction': BASE_FRACTION, 'stroke_strength': STROKE_STRENGTH},
        'targeted_augmentation': {'classes': list(WEAK_LABELS), 'rows': ADDITIONAL_COUNT},
        'source_fit_split': 'EXP-014 official Training 76800, not full 96000',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scikit_learn': sklearn.__version__},
    }
    (OUT / f'{prefix}_metrics.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{prefix}: correct={count}/{len(labels)}, Accuracy={metrics["accuracy"]:.6f}, '
          f'Macro F1={metrics["macro_f1"]:.6f}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-test', action='store_true',
                        help='Run held-out official Test ONLY after frozen Validation reproduction')
    args = parser.parse_args()
    print('FINAL KNN NORMAL: frozen EXP-014 targeted_10 / EXP-015 k3_distance', flush=True)
    xtr, ytr, xva, yva, sha = fixed_split()
    if (len(ytr), len(yva)) != (76800, 19200):
        raise RuntimeError('Official split mismatch')
    com = BatchedCoM()
    start = time.perf_counter()
    x_train, y_train = assemble_training(xtr, ytr, com)
    del xtr, ytr
    x_valid = prepare_features(xva, com, 'official Validation')
    del xva
    model = KNeighborsClassifier(n_neighbors=3, weights='distance',
                                 metric='euclidean', algorithm='brute', n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_valid).astype(np.int64)
    correct = int(np.sum(predictions == yva))
    if correct != EXPECTED_VALIDATION_CORRECT:
        raise RuntimeError(f'Frozen model Validation mismatch: {correct} != '
                           f'{EXPECTED_VALIDATION_CORRECT}. DO NOT evaluate Test.')
    print('EXP-014 official Validation EXACTLY REPRODUCED', flush=True)
    write_evaluation('validation', yva, predictions, sha)
    del x_valid, predictions

    if not args.evaluate_test:
        print('Official Test NOT loaded or evaluated. To run it once, add --evaluate-test.', flush=True)
        print('Results:', OUT, flush=True)
        return

    print('Evaluating held-out official Test with FROZEN model (one-time final report)...', flush=True)
    xtest, ytest, test_sha = verified_official_test()
    x_test = prepare_features(xtest, com, 'official Test')
    del xtest
    pred_test = model.predict(x_test).astype(np.int64)
    write_evaluation('official_test', ytest, pred_test, test_sha)
    print(f'Elapsed seconds: {time.perf_counter()-start:.1f}; Results: {OUT}', flush=True)
    print('Do not tune model hyperparameters based on official Test results.', flush=True)


if __name__ == '__main__':
    main()
