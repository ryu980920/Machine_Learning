#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-006: KNN Normal geometry study: CoM vs BB-size vs BB-size-then-CoM.

Place in 05_model_optimize/KNN/Normal/scripts/ alongside EXP004_KNN_Normal_CoM.py.
Uses official Training fold only for fitting, Validation and 4-writer Hand-made
for development assessment. Does not load Original Test or final Test.

Size transform follows the *exact geometry and scipy affine_transform formula*
from MLP/Combined/scripts/exp_cmb004_geometry_normalization.py and
MLP/Normal/scripts/exp006_size_com_comparison.py (threshold=32,
train-only median bbox long-side target, order=1, constant zero fill).
CoM uses the existing MLP Normal Automatic_Centering_Transformer.py via
EXP004_KNN_Normal_CoM.BatchedCoM.
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.ndimage import affine_transform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / '03_experiments' / 'model_optimization'))
sys.path.insert(0, str(ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'))

from common import LABELS, fixed_split, pipeline, scores
from EXP004_KNN_Normal_CoM import BatchedCoM, read_handmade, paired_stats
from Automatic_Centering_Transformer import test_centering  # imported by EXP004

OUTPUT = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'EXP006_bb_size'
EXPECTED_COM_OFFICIAL_CORRECT = 18529  # EXP-005 k3_distance, N=19200
EXPECTED_COM_HANDMADE_CORRECT = 2017  # EXP-005 k3_distance, N=2560
GUARDRAIL_OFFICIAL = 0.9569791666666667  # EXP-005 selection guardrail (>=95.6979%)
THRESHOLD = 32
TARGET_COM = (13.5, 13.5)


def geometry(batch):
    """Same bbox long-side/center definition as MLP Combined EXP-CMB-004."""
    a = np.asarray(batch, dtype=np.float32).reshape(-1, 28, 28)
    fg = a >= THRESHOLD
    if not np.all(fg.any(axis=(1, 2))):
        raise ValueError('Empty foreground image at threshold 32: cannot size-normalize')
    xx = fg.any(axis=1)
    yy = fg.any(axis=2)
    xmin = xx.argmax(axis=1)
    xmax = 27 - xx[:, ::-1].argmax(axis=1)
    ymin = yy.argmax(axis=1)
    ymax = 27 - yy[:, ::-1].argmax(axis=1)
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    return (np.maximum(width, height), (ymin + ymax) / 2.0, (xmin + xmax) / 2.0)


def apply_size(batch, target):
    """Exact BB-size affine rule used by the MLP Combined reference code."""
    a = np.asarray(batch, dtype=np.float32).reshape(-1, 28, 28)
    longs, cy, cx = geometry(a)
    factors = float(target) / longs.astype(float)
    out = np.empty_like(a)
    for i in range(len(a)):
        inv = 1.0 / factors[i]
        center = (cy[i], cx[i])
        img = affine_transform(a[i], np.eye(2) * inv,
              offset=(center[0] * (1 - inv), center[1] * (1 - inv)),
              output_shape=(28, 28), order=1,
              mode='constant', cval=0.0, prefilter=False)
        out[i] = np.clip(img, 0, 255)
    return out.reshape(-1, 784), factors


class BatchedBBSize:
    """A standalone fit/transform object; target is trained on official train only."""
    def __init__(self, target_longside, batch_size=2048):
        self.target_longside = int(target_longside)
        self.batch_size = int(batch_size)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        a = np.asarray(X, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 784:
            raise ValueError(f'Expected (N,784), got {a.shape}')
        out = np.empty_like(a)
        for start in range(0, len(a), self.batch_size):
            stop = min(start + self.batch_size, len(a))
            out[start:stop], _ = apply_size(a[start:stop], self.target_longside)
        return out


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to write')
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_cm(path, matrix):
    write_csv(path, [
        {'true/pred': LABELS[i], **{LABELS[j]: int(matrix[i][j]) for j in range(16)}}
        for i in range(16)
    ])


def invariance_tests():
    test_centering()
    x = np.zeros((1, 28, 28), dtype=np.float32)
    x[0, 6:22, 9:19] = 255.0
    # Identity when target equals actual bbox long side
    y, factors = apply_size(x, 16)
    if not (np.allclose(y, x.reshape(1, 784)) and np.allclose(factors, 1.0)):
        raise AssertionError('BB-size identity test failed')
    scaled, _ = apply_size(x, 20)
    if not (scaled.shape == (1, 784) and np.isfinite(scaled).all()
            and 0 <= scaled.min() <= scaled.max() <= 255):
        raise AssertionError('BB-size range/shape test failed')
    print('GEOMETRY SELF-TEST PASSED: label-free BB resizing and MLP CoM', flush=True)


def model_for(method, target):
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance',
                               metric='minkowski', p=2)
    steps = []
    if method in ('bbox_size', 'size_then_com'):
        steps.append(('bbox_size', BatchedBBSize(target)))
    if method in ('com', 'size_then_com'):
        steps.append(('center_of_mass', BatchedCoM()))
    steps.extend(pipeline(knn, True).steps)
    return Pipeline(steps)


def score_rows(method, true, predictions):
    stats = scores(true, predictions)
    return {'method': method, 'n': len(true), 'accuracy': stats['accuracy'],
            'macro_f1': stats['macro_f1']}, stats


def main():
    invariance_tests()
    xtr, ytr, xva, yva, dataset_sha = fixed_split()
    xhm, yhm, writers, local_indices = read_handmade()
    print(f'Data: official train={len(ytr)}, validation={len(yva)}, Hand-made={len(yhm)}', flush=True)
    target = int(np.median(geometry(xtr)[0]))
    print('BB target long side from OFFICIAL TRAIN ONLY:', target, 'pixels', flush=True)
    if target != 20:
        raise AssertionError(f'Expected MLP reference target 20, got {target}. Check official split.')

    methods = ('com', 'bbox_size', 'size_then_com')
    pred_va_all, pred_hm_all = {}, {}
    results, writer_rows, class_rows, prediction_rows = [], [], [], []
    fit_times, predict_times = {}, {}
    scored = {}

    for method in methods:
        print(f'\n[{method}] Fitting official Training ONLY...', flush=True)
        model = model_for(method, target)
        begin = time.perf_counter()
        model.fit(xtr, ytr)
        fit_times[method] = time.perf_counter() - begin

        begin = time.perf_counter()
        pred_va = model.predict(xva).astype(np.int64)
        pred_hm = model.predict(xhm).astype(np.int64)
        predict_times[method] = time.perf_counter() - begin

        pred_va_all[method], pred_hm_all[method] = pred_va, pred_hm
        ovrow, ovs = score_rows(method, yva, pred_va)
        hmrow, hms = score_rows(method, yhm, pred_hm)
        scored[method] = {'official': ovs, 'handmade': hms}
        results.append({'method': method,
                        'official_accuracy': ovs['accuracy'], 'official_macro_f1': ovs['macro_f1'],
                        'handmade_accuracy': hms['accuracy'], 'handmade_macro_f1': hms['macro_f1'],
                        'guardrail_pass': int(ovs['accuracy'] >= GUARDRAIL_OFFICIAL)})
        print(f"{method:14s} Official={ovs['accuracy']:.6f} F1={ovs['macro_f1']:.6f} | "
              f"Hand-made={hms['accuracy']:.6f} F1={hms['macro_f1']:.6f} | "
              f"guardrail={'PASS' if ovs['accuracy'] >= GUARDRAIL_OFFICIAL else 'FAIL'}", flush=True)
        for writer in (41, 42, 43, 44):
            mask = writers == writer
            s = scores(yhm[mask], pred_hm[mask])
            writer_rows.append({'method': method, 'writer': writer, 'n': int(mask.sum()),
                                'accuracy': s['accuracy'], 'macro_f1': s['macro_f1']})
        for c in LABELS:
            s = hms['per_class'][c]
            class_rows.append({'method': method, 'class': c, 'n': s['support'],
                               'precision': s['precision'], 'recall': s['recall'], 'f1': s['f1']})

    if (int(np.sum(pred_va_all['com'] == yva)) != EXPECTED_COM_OFFICIAL_CORRECT
        or int(np.sum(pred_hm_all['com'] == yhm)) != EXPECTED_COM_HANDMADE_CORRECT):
        raise RuntimeError('EXP-005 CoM k3_distance reference did NOT reproduce. '
                           'Check environment/split/preprocessing before comparing results.')
    print('\nEXP-005 k3_distance CoM reference verified exactly (correct counts).', flush=True)
    for row in results:
        row['official_delta_pp_vs_com'] = 100 * (row['official_accuracy'] - results[0]['official_accuracy'])
        row['handmade_delta_pp_vs_com'] = 100 * (row['handmade_accuracy'] - results[0]['handmade_accuracy'])

    chosen_pool = [row for row in results if row['guardrail_pass']]
    chosen = max(chosen_pool, key=lambda row: (row['handmade_macro_f1'], row['handmade_accuracy'],
                                              row['official_macro_f1'])) if chosen_pool else None

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / 'candidate_results.csv', results)
    write_csv(OUTPUT / 'by_writer.csv', writer_rows)
    write_csv(OUTPUT / 'per_class_handmade.csv', class_rows)

    for method in methods:
        save_cm(OUTPUT / f'handmade_cm_{method}.csv', scored[method]['handmade']['confusion_matrix'])
        save_cm(OUTPUT / f'official_cm_{method}.csv', scored[method]['official']['confusion_matrix'])
        for i in range(len(yhm)):
            prediction_rows.append({'method': method, 'writer': int(writers[i]),
                                    'index': int(local_indices[i]), 'true': LABELS[int(yhm[i])],
                                    'predicted': LABELS[int(pred_hm_all[method][i])],
                                    'correct': int(pred_hm_all[method][i] == yhm[i])})
    write_csv(OUTPUT / 'handmade_predictions.csv', prediction_rows)
    transitions = {
        name: {
            'official': paired_stats(yva, pred_va_all['com'], pred_va_all[name]),
            'handmade': paired_stats(yhm, pred_hm_all['com'], pred_hm_all[name])
        } for name in methods if name != 'com'
    }
    report = {
        'experiment': 'EXP-006 KNN Normal BB-size vs CoM vs BB-size-then-CoM',
        'methods': list(methods),
        'reference': 'EXP-005 CoM + k3 distance',
        'training_n': int(len(ytr)), 'validation_n': int(len(yva)), 'handmade_n': int(len(yhm)),
        'official_dataset_sha256': dataset_sha,
        'target_bbox_longside': target, 'target_fit': 'median official Training bbox long side only',
        'bbox_threshold': THRESHOLD,
        'bbox_interpolation': 'scipy.ndimage.affine_transform order=1, constant zero fill, centered on bbox',
        'com_method': 'MLP Normal mass center, threshold 32, target (13.5,13.5), integer zero-filled shifts',
        'model': 'KNeighborsClassifier(n_neighbors=3,weights=distance,p=2)',
        'official_accuracy_guardrail': GUARDRAIL_OFFICIAL,
        'selection': 'Among guardrail passes: highest Hand-made Macro F1, then accuracy, official macro F1',
        'candidates': results, 'selected_development_candidate': chosen,
        'paired_vs_com': transitions,
        'fit_seconds': fit_times, 'predict_seconds': predict_times,
        'python': platform.python_version(), 'numpy': np.__version__,
        'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
        'limitations': 'Hand-made reused for development; Original Test and final professor Test untouched. '
                       'BB-size method is NOT BB-center: it changes scale around bbox center. '
                       'No augmentation. No independent final score.'
    }
    (OUTPUT / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nPAIRED vs CoM:')
    for name, pair in transitions.items():
        print(name, pair)
    if chosen:
        print('\nSELECTED DEVELOPMENT CANDIDATE:', chosen['method'])
        print(f"Official={chosen['official_accuracy']:.6f}, "
              f"Hand-made={chosen['handmade_accuracy']:.6f}, "
              f"MacroF1={chosen['handmade_macro_f1']:.6f}")
    else:
        print('No candidates pass official Validation guardrail.')
    print('Saved:', OUTPUT)


if __name__ == '__main__':
    main()
