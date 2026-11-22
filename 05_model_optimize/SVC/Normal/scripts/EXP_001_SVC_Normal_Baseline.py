#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVC Normal EXP-001: official validation + four-writer Hand-made development evaluation.

From the repository root:
 python 05_model_optimize/SVC/Normal/scripts/EXP_001_SVC_Normal_Baseline.py
 python 05_model_optimize/SVC/Normal/scripts/EXP_001_SVC_Normal_Baseline.py --pilot

Only the official Training/Validation split is read; official Test is never accessed.
Uses the repository's own SVC build_baseline() when available; fails rather than
silently substituting a different baseline if no SVC baseline is found.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[4]
COMMON_DIR = ROOT / '03_experiments' / 'model_optimization'
sys.path.insert(0, str(COMMON_DIR))
from common import fixed_split, pipeline, LABELS

WRITERS = (41, 42, 43, 44)
EXPECTED_SHA = {
    41: '0a829172a72787c770139c73de34cc8b751181829ff6fe08ec260247b9be5a2f',
    42: 'f364fb453d05b4fe2a705c2b8a80683b404608c7c257d664e3ec7ac86932d4c5',
    43: '10db20997c727c1bc1f1299f8e89626fadd3ec9618468d6759c86b0d2ddc4a66',
    44: '0e266a36e7b29edfcca0514836b5c383d13939948b4e335d70f61bb40bdc084d',
}
OUT_BASE = ROOT / '05_model_optimize' / 'SVC' / 'Normal' / 'results'


def csv_save(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_writer(writer):
    p = ROOT / '02_data' / 'Hand_made' / f'ml26_{writer}.npz'
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA[writer]:
        raise RuntimeError(f'Hand-made SHA256 mismatch for writer {writer}: {digest}')
    with np.load(p, allow_pickle=False) as d:
        if set(d.files) != {'img', 'label', 'label_id'}:
            raise ValueError(f'Unexpected keys in {p}: {d.files}')
        x = d['img']
        y = d['label_id'].astype(np.int64)
        labels = d['label']
    if x.shape != (640, 28, 28) or x.dtype != np.uint8 or y.shape != (640,):
        raise ValueError(f'Unexpected shape/dtype: {p}')
    if not np.array_equal(y, np.array([LABELS.index(str(t)) for t in labels])):
        raise ValueError(f'Label mapping mismatch: {p}')
    if not np.all(np.bincount(y, minlength=16) == 40):
        raise ValueError(f'Writer {writer} not balanced by class')
    return x.reshape(640, 784).astype(np.float32), y, digest


def baseline():
    # The team's original 03_experiments baseline is authoritative.
    # Do not silently replace it with sklearn defaults.
    candidates = [COMMON_DIR / 'svc.py', COMMON_DIR / 'svm.py']
    for p in candidates:
        if p.is_file():
            spec = importlib.util.spec_from_file_location('_project_svc_baseline', p)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, 'build_baseline'):
                continue
            result = module.build_baseline()
            if not (isinstance(result, tuple) and len(result) == 2):
                raise TypeError(f'{p}: build_baseline() should return (estimator, scaled)')
            estimator, scaled = result
            if not isinstance(estimator, SVC):
                raise TypeError(f'{p}: expected sklearn.svm.SVC, got {type(estimator)}')
            if not isinstance(scaled, (bool, np.bool_)):
                raise TypeError(f'{p}: expected boolean scaling flag')
            return estimator, bool(scaled), str(p.relative_to(ROOT))
    raise FileNotFoundError(
        'SVC baseline not found. Expected 03_experiments/model_optimization/svc.py '
        'or svm.py defining build_baseline() -> (SVC, scaled). '
        'Please check your repository for the original SVC baseline file.'
    )


def report(y_true, y_pred):
    return {'n': int(len(y_true)), 'correct': int((y_true == y_pred).sum()),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_f1': float(f1_score(y_true, y_pred, labels=np.arange(16), average='macro', zero_division=0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true', help='Stratified 8000 training / 3200 validation pilot; not the full baseline')
    args = ap.parse_args()
    suffix = 'EXP_001_SVC_Normal_PILOT' if args.pilot else 'EXP_001_SVC_Normal_Baseline'
    out = OUT_BASE / suffix

    base, scaled, source = baseline()
    print('SVC original baseline:', source, flush=True)
    print('SVC parameters:', base.get_params(), flush=True)
    print('Scaling:', scaled, flush=True)

    xtr, ytr, xval, yval, data_sha = fixed_split()
    if len(ytr) != 76800 or len(yval) != 19200:
        raise RuntimeError('Official split mismatch; expected 76800/19200')
    if args.pilot:
        ids, _ = train_test_split(np.arange(len(ytr)), train_size=8000, stratify=ytr, random_state=42)
        val_ids, _ = train_test_split(np.arange(len(yval)), train_size=3200, stratify=yval, random_state=42)
        xtr, ytr, xval, yval = xtr[ids], ytr[ids], xval[val_ids], yval[val_ids]
    print(f'Official Training={len(ytr)}, Validation={len(yval)}. Official Test NOT loaded.', flush=True)

    handmade = {w: load_writer(w) for w in WRITERS}
    clf = pipeline(base, scaled)
    t0 = time.perf_counter()
    print('[SVC Normal] fitting...', flush=True)
    clf.fit(xtr, ytr)
    fit_seconds = time.perf_counter() - t0
    print(f'fit_seconds={fit_seconds:.2f}', flush=True)

    pv = clf.predict(xval).astype(np.int64)
    official = report(yval, pv)
    print(f"Official Validation Accuracy={official['accuracy']:.6f}, F1={official['macro_f1']:.6f}", flush=True)
    truth, predictions, writer_rows, pred_rows = [], [], [], []
    for w in WRITERS:
        x, y, _ = handmade[w]
        p = clf.predict(x).astype(np.int64)
        m = report(y, p)
        writer_rows.append({'writer': w, **m})
        print(f"Writer {w}: {m['correct']}/640, Accuracy={m['accuracy']:.6f}, F1={m['macro_f1']:.6f}", flush=True)
        truth.append(y)
        predictions.append(p)
        pred_rows.extend({'writer': w, 'index': i, 'true': LABELS[int(a)],
                          'prediction': LABELS[int(b)], 'correct': int(a == b)}
                         for i, (a, b) in enumerate(zip(y, p)))
    yt, yp = np.concatenate(truth), np.concatenate(predictions)
    overall = report(yt, yp)
    print(f"Hand-made DEVELOPMENT: Accuracy={overall['accuracy']:.6f}, F1={overall['macro_f1']:.6f} ({overall['correct']}/2560)", flush=True)
    cm = confusion_matrix(yt, yp, labels=np.arange(16))
    pr, re, f1, support = precision_recall_fscore_support(yt, yp, labels=np.arange(16), zero_division=0)
    out.mkdir(parents=True, exist_ok=True)
    csv_save(out / 'writer_results.csv', writer_rows)
    csv_save(out / 'handmade_predictions.csv', pred_rows)
    csv_save(out / 'handmade_confusion_matrix.csv',
             [{'true\\pred': LABELS[i], **{LABELS[j]: int(cm[i, j]) for j in range(16)}} for i in range(16)])
    csv_save(out / 'per_class.csv',
             [{'class': LABELS[i], 'precision': float(pr[i]), 'recall': float(re[i]),
               'f1': float(f1[i]), 'support': int(support[i])} for i in range(16)])
    metrics = {'experiment': suffix, 'model_source': source, 'hyperparameters': base.get_params(),
               'scaled': scaled, 'official_train_n': len(ytr), 'official_val_n': len(yval),
               'official_validation': official, 'handmade_development': overall,
               'writers': writer_rows, 'fit_seconds': fit_seconds,
               'dataset_sha256': data_sha,
               'handmade_sha256': {str(w): handmade[w][2] for w in WRITERS},
               'official_test_used': False,
               'evaluation_role': 'repeated-use Hand-made development; NOT independent final test',
               'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                               'sklearn': sklearn.__version__}}
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('Saved:', out, flush=True)


if __name__ == '__main__':
    main()
