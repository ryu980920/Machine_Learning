#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-011: image-moment deskew for KNN Normal, retaining EXP-009 augmentation.

Run from repository root:
  python 05_model_optimize/KNN/Normal/scripts/EXP011_KNN_Normal_Deskew.py

Required scripts in the same folder:
  EXP004_KNN_Normal_CoM.py
  EXP006_KNN_Normal_BB_Size_fixed.py
  EXP007_KNN_Normal_Augmentation.py

Compare:
  com_control: CoM (EXP-009 best, no deskew)
  deskew_then_com: image-moment deskew, then CoM
  com_then_deskew: CoM, then image-moment deskew
All variants share the exact EXP-009 30% combined augmentation, stroke=0.44,
KNN k=3 distance Euclidean, official training only. No test-set leakage.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'
sys.path.insert(0, str(ROOT / '03_experiments' / 'model_optimization'))
sys.path.insert(0, str(SCRIPTS))

from common import fixed_split, LABELS, scores
from EXP004_KNN_Normal_CoM import BatchedCoM, read_handmade, paired_stats
from EXP007_KNN_Normal_Augmentation import (
    knn_model, sample_balanced, size_jitter, stroke_jitter,
    test_augmentation, save_csv, save_cm, OFFICIAL_GUARDRAIL, SEED,
)

OUTPUT = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'EXP011_deskew'
AUG_FRACTION = 0.30
STROKE_STRENGTH = 0.44
EXPECTED_CONTROL_CORRECT = (18381, 2153)
THRESHOLD = 32.0
MAX_SLOPE = 0.45
PIXEL_GRID_Y, PIXEL_GRID_X = np.indices((28, 28), dtype=np.float64)


def deskew(batch, batch_size=512):
    """Label-free moment shear correction, bilinear interpolation, constant-zero border.

    slope = mu11 / mu02 where mu11=cov(x,y) and mu02=var(y).
    For output (x,y), sample input (x + slope*(y-cy), y).
    Near-empty or y-degenerate glyphs remain unchanged. Clamp slope for safety.
    Input/output pixel range: 0..255. Input shape: (N,784).
    """
    src = np.asarray(batch, dtype=np.float32)
    if src.ndim != 2 or src.shape[1] != 784:
        raise ValueError(f'Expected (N,784), got {src.shape}')
    out = np.empty_like(src)
    clipped = skipped = changed = 0
    for i in range(len(src)):
        img = src[i].reshape(28, 28)
        mass = np.where(img >= THRESHOLD, img, 0).astype(np.float64)
        total = mass.sum()
        if total <= 1e-9:
            out[i] = src[i]
            skipped += 1
            continue
        cx = float(np.sum(mass * PIXEL_GRID_X) / total)
        cy = float(np.sum(mass * PIXEL_GRID_Y) / total)
        dx, dy = PIXEL_GRID_X - cx, PIXEL_GRID_Y - cy
        mu02 = float(np.sum(mass * dy * dy) / total)
        if mu02 < 1e-6:
            out[i] = src[i]
            skipped += 1
            continue
        slope = float(np.sum(mass * dx * dy) / total / mu02)
        bounded = float(np.clip(slope, -MAX_SLOPE, MAX_SLOPE))
        if bounded != slope:
            clipped += 1
        if abs(bounded) <= 1e-7:
            out[i] = src[i]
            skipped += 1
            continue
        xsrc = PIXEL_GRID_X + bounded * (PIXEL_GRID_Y - cy)
        corrected = map_coordinates(img, [PIXEL_GRID_Y, xsrc], order=1,
                                    mode='constant', cval=0.0, prefilter=False)
        out[i] = np.clip(corrected, 0, 255).reshape(784).astype(np.float32)
        changed += 1
    return out, {'total': int(len(src)), 'changed': changed,
                 'skipped': skipped, 'clamped': clipped,
                 'threshold': THRESHOLD, 'max_abs_slope': MAX_SLOPE,
                 'interpolation': 'bilinear', 'border': 'constant_zero'}


def test_deskew():
    a = np.zeros((3, 28, 28), dtype=np.float32)
    a[0, 6:23, 13:16] = 255.0
    for y in range(6, 23):
        x = int(round(12 + 0.3 * (y - 14)))
        a[1, y, x:x+3] = 255.0
    transformed, info = deskew(a.reshape(-1, 784))
    assert transformed.shape == (3, 784)
    assert np.all(np.isfinite(transformed)) and 0 <= transformed.min() <= transformed.max() <= 255
    assert np.all(transformed[2] == 0)
    assert info['skipped'] >= 1
    def slope_of(im):
        m = im.reshape(28, 28).astype(np.float64)
        t = m.sum()
        cx = (m * PIXEL_GRID_X).sum() / t
        cy = (m * PIXEL_GRID_Y).sum() / t
        return ((m * (PIXEL_GRID_X - cx) * (PIXEL_GRID_Y - cy)).sum()
                / (m * (PIXEL_GRID_Y - cy) ** 2).sum())
    assert abs(slope_of(transformed[1])) < abs(slope_of(a[1])), 'Synthetic skew not reduced'
    print('DESKEW SELF-TEST PASSED: shape, blank, shear correction, range', flush=True)


def process(batch, kind, com):
    if kind == 'com_control':
        return com.transform(batch), {'kind': 'com_only'}
    if kind == 'deskew_then_com':
        prepared, diagnostics = deskew(batch)
        return com.transform(prepared), diagnostics
    if kind == 'com_then_deskew':
        prepared = com.transform(batch)
        return deskew(prepared)
    raise ValueError(kind)


def main():
    test_augmentation()
    test_deskew()
    xtr, ytr, xva, yva, dataset_sha = fixed_split()
    xhm, yhm, writers, sample_ids = read_handmade()
    if (len(ytr), len(yva), len(yhm)) != (76800, 19200, 2560):
        raise ValueError('Unexpected split sizes')
    print('Training=76800; Official Validation=19200; Hand-made=2560', flush=True)
    idx = sample_balanced(ytr, fraction=AUG_FRACTION, seed=SEED)
    sized, size_stats = size_jitter(xtr, idx, seed=SEED)
    aug, stroke_stats = stroke_jitter(sized, strength=STROKE_STRENGTH)
    train_raw = np.concatenate((xtr, aug), axis=0)
    train_y = np.concatenate((ytr, ytr[idx]), axis=0)
    assert len(train_y) == 99840
    print(f'EXACT EXP-009 augmentation: {len(idx)} extra official Training only, '
          f'strength={STROKE_STRENGTH}', flush=True)

    com = BatchedCoM()
    candidate_rows = []
    writer_rows = []
    class_rows = []
    detail = {}
    preds = {}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for kind in ('com_control', 'deskew_then_com', 'com_then_deskew'):
        print(f'\n[{kind}] Preparing training and evaluation...', flush=True)
        t = time.perf_counter()
        train, train_diag = process(train_raw, kind, com)
        val, val_diag = process(xva, kind, com)
        hand, hand_diag = process(xhm, kind, com)
        prep_seconds = time.perf_counter() - t
        model = knn_model()
        t = time.perf_counter()
        model.fit(train, train_y)
        fit_seconds = time.perf_counter() - t
        t = time.perf_counter()
        pred_val = np.asarray(model.predict(val), dtype=np.int64)
        val_seconds = time.perf_counter() - t
        t = time.perf_counter()
        pred_hand = np.asarray(model.predict(hand), dtype=np.int64)
        hand_seconds = time.perf_counter() - t
        off, hm = scores(yva, pred_val), scores(yhm, pred_hand)
        correct = (int((yva == pred_val).sum()), int((yhm == pred_hand).sum()))
        if kind == 'com_control':
            if correct != EXPECTED_CONTROL_CORRECT:
                raise RuntimeError(f'EXP-009 reference mismatch: {correct} != {EXPECTED_CONTROL_CORRECT}')
            print('EXP-009 BEST REPRODUCED EXACTLY', flush=True)
        eligible = off['accuracy'] >= OFFICIAL_GUARDRAIL
        row = {
            'variant': kind, 'train_n': int(len(train_y)), 'augmentation_n': int(len(idx)),
            'official_correct': correct[0], 'official_accuracy': off['accuracy'],
            'official_macro_f1': off['macro_f1'], 'handmade_correct': correct[1],
            'handmade_accuracy': hm['accuracy'], 'handmade_macro_f1': hm['macro_f1'],
            'official_guardrail_pass': int(eligible),
            'prep_seconds': prep_seconds, 'fit_seconds': fit_seconds,
            'official_predict_seconds': val_seconds,
            'handmade_predict_seconds': hand_seconds,
        }
        candidate_rows.append(row)
        detail[kind] = {'official': off, 'handmade': hm,
                        'deskew_diagnostics': {'training': train_diag,
                                               'official': val_diag, 'handmade': hand_diag}}
        preds[kind] = (pred_val, pred_hand)
        for writer in (41, 42, 43, 44):
            mask = writers == writer
            ss = scores(yhm[mask], pred_hand[mask])
            writer_rows.append({'variant': kind, 'writer': writer,
                                'n': int(mask.sum()), 'accuracy': ss['accuracy'],
                                'macro_f1': ss['macro_f1']})
        for label_name in LABELS:
            p = hm['per_class'][label_name]
            class_rows.append({'variant': kind, 'class': label_name,
                               'support': p['support'], 'precision': p['precision'],
                               'recall': p['recall'], 'f1': p['f1']})
        save_csv(OUTPUT / 'candidate_results.csv', candidate_rows)
        save_csv(OUTPUT / 'by_writer.csv', writer_rows)
        save_csv(OUTPUT / 'per_class_handmade.csv', class_rows)
        save_cm(OUTPUT / f'official_cm_{kind}.csv', off['confusion_matrix'])
        save_cm(OUTPUT / f'handmade_cm_{kind}.csv', hm['confusion_matrix'])
        save_csv(OUTPUT / f'handmade_predictions_{kind}.csv', [
            {'writer': int(writers[i]), 'index': int(sample_ids[i]),
             'true': LABELS[int(yhm[i])], 'predicted': LABELS[int(pred_hand[i])],
             'correct': int(yhm[i] == pred_hand[i])}
            for i in range(len(yhm))
        ])
        (OUTPUT / f'diagnostics_{kind}.json').write_text(
            json.dumps(detail[kind]['deskew_diagnostics'], ensure_ascii=False, indent=2),
            encoding='utf-8')
        print(f"{kind:<20} Official={off['accuracy']:.6f} F1={off['macro_f1']:.6f} | "
              f"Hand-made={hm['accuracy']:.6f} F1={hm['macro_f1']:.6f} "
              f"| guardrail={'PASS' if eligible else 'FAIL'}", flush=True)
        del model, train, val, hand

    paired = {
        name: {
            'official': paired_stats(yva, preds['com_control'][0], preds[name][0]),
            'handmade': paired_stats(yhm, preds['com_control'][1], preds[name][1]),
        }
        for name in ('deskew_then_com', 'com_then_deskew')
    }
    valid = [row for row in candidate_rows if row['official_guardrail_pass']]
    winner = max(valid, key=lambda r: (r['handmade_macro_f1'], r['handmade_accuracy'],
                                        r['official_macro_f1'])) if valid else None
    metrics = {
        'experiment': 'EXP-011 KNN Normal moment-based deskew comparison',
        'official_dataset_sha256': dataset_sha,
        'train_n': int(len(ytr)), 'official_validation_n': int(len(yva)),
        'handmade_n': int(len(yhm)),
        'knn': {'n_neighbors': 3, 'weights': 'distance', 'p': 2},
        'augmentation': {'fraction': AUG_FRACTION, 'stroke_strength': STROKE_STRENGTH,
                         'augmented_n': len(idx), 'size_counts': size_stats,
                         'stroke_stats': stroke_stats},
        'deskew': {'threshold': THRESHOLD, 'max_abs_slope': MAX_SLOPE,
                   'interpolation': 'bilinear', 'border': 'zero'},
        'official_guardrail_accuracy': OFFICIAL_GUARDRAIL,
        'candidates': candidate_rows, 'selected_development_candidate': winner,
        'paired_vs_control': paired,
        'environment': {'python': platform.python_version(),
                        'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'note': 'Hand-made is repeatedly used development data; not independent final test. '
                'All augmentation uses official training only. No held-out test files loaded.',
    }
    (OUTPUT / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nPAIRED VS EXP-009 BEST:', paired, flush=True)
    print('SELECTED DEVELOPMENT CANDIDATE:', winner['variant'] if winner else 'NONE', flush=True)
    if winner:
        print(f"Official={winner['official_accuracy']:.6f}, Hand-made={winner['handmade_accuracy']:.6f}, "
              f"MacroF1={winner['handmade_macro_f1']:.6f}", flush=True)
    print('Saved:', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
