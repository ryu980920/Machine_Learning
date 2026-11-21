#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-012: KNN Normal pixel vs HOG vs pixel+HOG.

Run from the repository root:
    python 05_model_optimize/KNN/Normal/scripts/EXP012_KNN_Normal_HOG.py

Place next to EXP004_KNN_Normal_CoM.py, EXP006_KNN_Normal_BB_Size_fixed.py,
EXP007_KNN_Normal_Augmentation.py, EXP011_KNN_Normal_Deskew.py.

Official training only (original + balanced 30% size/stroke augmentation).
Evaluation: unchanged official Validation and Hand-made (development only).
No MLP/CNN/held-out official Test data.
"""
from __future__ import annotations

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

from common import fixed_split, LABELS, scores
from EXP004_KNN_Normal_CoM import BatchedCoM, read_handmade, paired_stats
from EXP007_KNN_Normal_Augmentation import (
    sample_balanced, size_jitter, stroke_jitter, test_augmentation,
    knn_model, save_csv, save_cm, OFFICIAL_GUARDRAIL, SEED,
)
from EXP011_KNN_Normal_Deskew import deskew, test_deskew

OUTPUT = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'EXP012_hog'
AUGMENTATION_FRACTION = 0.30
STROKE_STRENGTH = 0.44
EXPECTED_EXP011_CONTROL = (18571, 2215)  # 0.96723958*19200; 0.865234375*2560
HOG_CELLS = 4
HOG_CELL_SIDE = 7
HOG_BINS = 9
HOG_FEATURE_DIM = 324
FEATURE_BATCH = 256


def hog_features(raw, batch_size=FEATURE_BATCH):
    """Unsigned 9-bin HOG, 7x7 cells, 2x2 L2-Hys-like normalized blocks.

    28x28 pixels => 4x4 cells => 3x3 overlapping blocks, 324 features.
    Gradient via central differences (numpy.gradient), bilinear vote to
    orientation bins, unsigned angles in [0,180). No label information.
    """
    images = np.asarray(raw, dtype=np.float32)
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError(f'Expected N x 784 array; got {images.shape}')
    output = np.empty((len(images), HOG_FEATURE_DIM), dtype=np.float32)
    iy, ix = np.indices((28, 28))
    cell_index = ((iy // HOG_CELL_SIDE) * HOG_CELLS + ix // HOG_CELL_SIDE).reshape(1, 784)
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        a = images[start:stop].reshape(-1, 28, 28) / 255.0
        gy, gx = np.gradient(a, axis=(1, 2))
        mag = np.hypot(gx, gy).reshape(-1, 784)
        ori = (np.arctan2(gy, gx) % np.pi).reshape(-1, 784) * (HOG_BINS / np.pi)
        low = np.floor(ori).astype(np.int32) % HOG_BINS
        high = (low + 1) % HOG_BINS
        high_weight = ori - np.floor(ori)
        n = stop - start
        batch_ix = np.arange(n, dtype=np.int32)[:, None]
        hist = np.zeros((n, 16 * HOG_BINS), dtype=np.float32)
        np.add.at(hist, (batch_ix, cell_index * HOG_BINS + low), mag * (1.0 - high_weight))
        np.add.at(hist, (batch_ix, cell_index * HOG_BINS + high), mag * high_weight)
        cells = hist.reshape(n, 4, 4, HOG_BINS)
        blocks = [cells[:, cy:cy+2, cx:cx+2, :].reshape(n, 36)
                  for cy in range(3) for cx in range(3)]
        stacked = np.concatenate(blocks, axis=1).reshape(n, 9, 36)
        norm = np.linalg.norm(stacked, axis=2, keepdims=True)
        normalized = stacked / np.maximum(norm, 1e-6)
        clipped = np.minimum(normalized, 0.2)
        renorm = np.linalg.norm(clipped, axis=2, keepdims=True)
        normalized = clipped / np.maximum(renorm, 1e-6)
        output[start:stop] = normalized.reshape(n, HOG_FEATURE_DIM)
    return output


def unit_rows(x):
    a = np.asarray(x, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)


def hybrid_features(pixel, hog, pixel_weight=0.5):
    """Feature-level fusion; pixel and HOG L2-normalized per sample."""
    if pixel_weight <= 0 or pixel_weight >= 1:
        raise ValueError('pixel_weight must lie strictly between zero and one')
    p = unit_rows(np.asarray(pixel, dtype=np.float32) / 255.0)
    h = unit_rows(hog)
    return np.ascontiguousarray(np.concatenate([
        np.sqrt(pixel_weight) * p,
        np.sqrt(1 - pixel_weight) * h,
    ], axis=1), dtype=np.float32)


def test_hog():
    a = np.zeros((3, 28, 28), dtype=np.float32)
    a[0, 5:22, 13:15] = 255
    a[1, 13:15, 5:22] = 255
    a[2] = 0
    h = hog_features(a.reshape(3, 784), batch_size=2)
    assert h.shape == (3, HOG_FEATURE_DIM)
    assert np.isfinite(h).all() and np.all(h >= 0)
    assert np.all(h[2] == 0) and np.linalg.norm(h[0] - h[1]) > 0.1
    mix = hybrid_features(a.reshape(3, 784), h)
    assert mix.shape == (3, 784 + HOG_FEATURE_DIM)
    assert np.isfinite(mix).all()
    print('HOG SELF-TEST PASSED: 324 dims, blank image, orientation sensitivity, hybrid', flush=True)


def evaluate(variant, xtrain, ytrain, xval, yval, xhand, yhand,
             writers, local_indices, control_predictions, candidate_rows,
             writer_rows, class_rows, records):
    print(f'\n[{variant}] Fit and predict...', flush=True)
    t = time.perf_counter()
    if variant == 'pixel_control':
        model = knn_model()  # identical to EXP-011: PixelScaler255 + KNN
    else:
        model = KNeighborsClassifier(n_neighbors=3, weights='distance',
                                     metric='minkowski', p=2, algorithm='brute', n_jobs=-1)
    model.fit(xtrain, ytrain)
    fit_seconds = time.perf_counter() - t
    t = time.perf_counter()
    pv = np.asarray(model.predict(xval), dtype=np.int64)
    val_seconds = time.perf_counter() - t
    t = time.perf_counter()
    ph = np.asarray(model.predict(xhand), dtype=np.int64)
    hand_seconds = time.perf_counter() - t
    off, hm = scores(yval, pv), scores(yhand, ph)
    correct = (int((yval == pv).sum()), int((yhand == ph).sum()))
    if variant == 'pixel_control':
        if correct != EXPECTED_EXP011_CONTROL:
            raise RuntimeError(f'EXP-011 control mismatch: got {correct}, expected {EXPECTED_EXP011_CONTROL}')
        control_predictions['official'], control_predictions['handmade'] = pv.copy(), ph.copy()
        print('EXP-011 BEST REPRODUCED EXACTLY', flush=True)
    eligible = off['accuracy'] >= OFFICIAL_GUARDRAIL
    candidate_rows.append({
        'variant': variant, 'train_n': len(ytrain), 'features': xtrain.shape[1],
        'official_correct': correct[0], 'official_accuracy': off['accuracy'],
        'official_macro_f1': off['macro_f1'], 'handmade_correct': correct[1],
        'handmade_accuracy': hm['accuracy'], 'handmade_macro_f1': hm['macro_f1'],
        'official_guardrail_pass': int(eligible),
        'fit_seconds': fit_seconds, 'official_predict_seconds': val_seconds,
        'handmade_predict_seconds': hand_seconds,
    })
    records[variant] = {'official': off, 'handmade': hm, 'features': xtrain.shape[1],
                        'fit_seconds': fit_seconds, 'official_predict_seconds': val_seconds,
                        'handmade_predict_seconds': hand_seconds}
    for w in (41, 42, 43, 44):
        mask = writers == w
        s = scores(yhand[mask], ph[mask])
        writer_rows.append({'variant': variant, 'writer': w, 'n': int(mask.sum()),
                            'accuracy': s['accuracy'], 'macro_f1': s['macro_f1']})
    for c in LABELS:
        p = hm['per_class'][c]
        class_rows.append({'variant': variant, 'class': c, 'support': p['support'],
                           'precision': p['precision'], 'recall': p['recall'], 'f1': p['f1']})
    save_csv(OUTPUT / 'candidate_results.csv', candidate_rows)
    save_csv(OUTPUT / 'by_writer.csv', writer_rows)
    save_csv(OUTPUT / 'per_class_handmade.csv', class_rows)
    save_cm(OUTPUT / f'official_cm_{variant}.csv', off['confusion_matrix'])
    save_cm(OUTPUT / f'handmade_cm_{variant}.csv', hm['confusion_matrix'])
    save_csv(OUTPUT / f'handmade_predictions_{variant}.csv', [
        {'writer': int(writers[i]), 'index': int(local_indices[i]),
         'true': LABELS[int(yhand[i])], 'predicted': LABELS[int(ph[i])],
         'correct': int(yhand[i] == ph[i])}
        for i in range(len(yhand))
    ])
    if variant != 'pixel_control':
        records[variant]['paired_vs_control'] = {
            'official': paired_stats(yval, control_predictions['official'], pv),
            'handmade': paired_stats(yhand, control_predictions['handmade'], ph),
        }
    (OUTPUT / 'metrics_partial.json').write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{variant:<18} Official={off['accuracy']:.6f} F1={off['macro_f1']:.6f} | "
          f"Hand-made={hm['accuracy']:.6f} F1={hm['macro_f1']:.6f} | "
          f"guardrail={'PASS' if eligible else 'FAIL'} | "
          f"predict_s={val_seconds:.1f}+{hand_seconds:.1f}", flush=True)
    del model


def main():
    test_augmentation()
    test_deskew()
    test_hog()
    xtr, ytr, xva, yva, sha = fixed_split()
    xhm, yhm, writers, local_indices = read_handmade()
    assert (len(ytr), len(yva), len(yhm)) == (76800, 19200, 2560)
    print(f'Training={len(ytr)}, Official Validation={len(yva)}, Hand-made={len(yhm)}', flush=True)
    selected = sample_balanced(ytr, fraction=AUGMENTATION_FRACTION, seed=SEED)
    sized, size_stats = size_jitter(xtr, selected, seed=SEED)
    extra, stroke_stats = stroke_jitter(sized, strength=STROKE_STRENGTH)
    raw_train = np.concatenate((xtr, extra), axis=0)
    train_labels = np.concatenate((ytr, ytr[selected]), axis=0)
    assert len(train_labels) == 99840
    print(f'Augmented official Training only: {len(selected)} extra rows', flush=True)
    com = BatchedCoM()
    t = time.perf_counter()
    xtrain_com = com.transform(raw_train)
    xval_com = com.transform(xva)
    xhand_com = com.transform(xhm)
    xtrain, _ = deskew(xtrain_com)
    xval, _ = deskew(xval_com)
    xhand, _ = deskew(xhand_com)
    print(f'CoM -> Deskew preprocessed all inputs in {time.perf_counter()-t:.1f}s', flush=True)
    del raw_train, xtrain_com, xval_com, xhand_com
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, by_writer, per_class, records, control_preds = [], [], [], {}, {}

    evaluate('pixel_control', xtrain, train_labels, xval, yva, xhand, yhm,
             writers, local_indices, control_preds, rows, by_writer, per_class, records)

    t = time.perf_counter()
    train_hog = hog_features(xtrain)
    val_hog = hog_features(xval)
    hand_hog = hog_features(xhand)
    print(f'HOG extraction complete in {time.perf_counter()-t:.1f}s', flush=True)

    evaluate('hog_only', train_hog, train_labels, val_hog, yva, hand_hog, yhm,
             writers, local_indices, control_preds, rows, by_writer, per_class, records)

    train_mix = hybrid_features(xtrain, train_hog, pixel_weight=0.5)
    val_mix = hybrid_features(xval, val_hog, pixel_weight=0.5)
    hand_mix = hybrid_features(xhand, hand_hog, pixel_weight=0.5)
    evaluate('pixel_hog_50_50', train_mix, train_labels, val_mix, yva, hand_mix, yhm,
             writers, local_indices, control_preds, rows, by_writer, per_class, records)

    valid = [r for r in rows if r['official_guardrail_pass']]
    winner = max(valid, key=lambda r: (r['handmade_macro_f1'], r['handmade_accuracy'],
                                      r['official_macro_f1'])) if valid else None
    payload = {
        'experiment': 'EXP-012', 'dataset_sha256': sha,
        'official_train_n': int(len(ytr)), 'augmented_train_n': len(selected),
        'fit_n': len(train_labels), 'official_validation_n': len(yva),
        'handmade_dev_n': len(yhm), 'augmentation': {'fraction': AUGMENTATION_FRACTION,
        'stroke_strength': STROKE_STRENGTH, 'target_long_sides': [16, 18, 22, 24],
        'size_stats': size_stats, 'stroke_stats': stroke_stats},
        'preprocessing': 'CoM then EXP-011 deskew on all splits',
        'knn': {'k': 3, 'weights': 'distance', 'metric': 'euclidean'},
        'hog': {'cell_px': 7, 'n_cells': 4, 'bins': 9,
                'block_cells': 2, 'block_stride_cells': 1, 'feature_dim': HOG_FEATURE_DIM},
        'hybrid': {'pixel_weight': 0.5, 'per_sample_l2_normalized': True},
        'official_guardrail': OFFICIAL_GUARDRAIL,
        'candidates': rows, 'selected_development_candidate': winner,
        'full_results': records,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'sklearn': sklearn.__version__},
        'note': 'Hand-made is repeatedly used for model development, NOT independent final testing. '
                'Only official Training used to create extra samples. No official Test used.',
    }
    (OUTPUT / 'metrics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nSELECTED DEVELOPMENT CANDIDATE:', winner['variant'] if winner else 'NONE', flush=True)
    print('Saved:', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
