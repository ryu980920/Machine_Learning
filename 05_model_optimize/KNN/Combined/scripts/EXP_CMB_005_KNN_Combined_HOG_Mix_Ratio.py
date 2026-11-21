#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-CMB-005: final Pixel/HOG ratio check for KNN Combined.

Run from the repository root:
 python 05_model_optimize/KNN/Combined/scripts/EXP_CMB_005_KNN_Combined_HOG_Mix_Ratio.py

Frozen EXP-CMB-004 repeat_mixed_50 training construction, with only
feature mixing weight changed. Writer-held-out OOF is DEVELOPMENT only.
No official Test is loaded or used. Results are saved after every fold.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[4]
NORMAL_DIR = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'
COMBINED_DIR = ROOT / '05_model_optimize' / 'KNN' / 'Combined' / 'scripts'
for path in (NORMAL_DIR, COMBINED_DIR, ROOT / '03_experiments' / 'model_optimization'):
    sys.path.insert(0, str(path))

from common import LABELS, fixed_split
from EXP004_KNN_Normal_CoM import BatchedCoM
from final_knn_normal import assemble_training, prepare_features, EXPECTED_VALIDATION_CORRECT
from EXP_CMB_001_KNN_Combined_Baseline import WRITERS, load_writer, save_csv, save_cm, summary
from EXP_CMB_002_KNN_Combined_Optimized import fresh_knn, paired_counts
from EXP_CMB_003_KNN_Combined_Handmade_Augmentation import (
    SEED as AUG_SEED, balanced_nested_indices, make_aug_features,
)

OUT = ROOT / '05_model_optimize' / 'KNN' / 'Combined' / 'results' / 'EXP_CMB_005_hog_mix_ratio'
PIXEL_WEIGHTS = (0.10, 0.20, 0.35, 0.50)
CONTROL_WEIGHT = 0.20
CONTROL_BY_WRITER = {41: 588, 42: 597, 43: 620, 44: 598}
CONTROL_TOTAL_CORRECT = 2403
OFFICIAL_GUARDRAIL = 0.978


def variant_name(w):
    return f'pixel_{round(w * 100):02d}_hog_{round((1 - w) * 100):02d}'


def remix(frozen_hybrid, pixel_weight):
    """Reuse normalized Pixel/HOG from the EXACT frozen 20:80 pipeline.

    The original 1108-D vector has sqrt(.2)*normalized_pixel (784 dims)
    and sqrt(.8)*normalized_hog (324 dims). Reweight without recomputing HOG.
    """
    data = np.asarray(frozen_hybrid, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 1108:
        raise ValueError(f'Expected (N, 1108) hybrid features, got {data.shape}')
    if not 0 < pixel_weight < 1:
        raise ValueError('Pixel weight must be strictly between 0 and 1')
    if pixel_weight == CONTROL_WEIGHT:
        return data
    a = np.float32(np.sqrt(pixel_weight / CONTROL_WEIGHT))
    b = np.float32(np.sqrt((1 - pixel_weight) / (1 - CONTROL_WEIGHT)))
    output = np.empty_like(data)
    output[:, :784] = data[:, :784] * a
    output[:, 784:] = data[:, 784:] * b
    return output


def self_test():
    v = np.zeros((2, 1108), dtype=np.float32)
    v[:, 0] = np.float32(np.sqrt(CONTROL_WEIGHT))
    v[:, 784] = np.float32(np.sqrt(1 - CONTROL_WEIGHT))
    assert np.array_equal(remix(v, CONTROL_WEIGHT), v)
    for w in PIXEL_WEIGHTS:
        a = remix(v, w)
        assert a.shape == v.shape and np.isfinite(a).all()
        assert np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-6)
        assert np.allclose(a[:, 0] ** 2, w, atol=1e-6)
    print('MIX-RATIO SELF-TEST PASSED: 1108 dims, exact control, norm', flush=True)


def write_partial(fold_results):
    save_csv(OUT / 'fold_results_partial.csv', fold_results)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    self_test()
    start = time.perf_counter()
    xtr, ytr, xva, yva, official_sha = fixed_split()
    if (len(ytr), len(yva)) != (76800, 19200):
        raise RuntimeError('Official Training/Validation split mismatch')
    writers = {w: load_writer(w) for w in WRITERS}
    com = BatchedCoM()
    print('Official Test NOT loaded. Preparing EXP-CMB-004 frozen 20:80 features...', flush=True)
    official_x, official_y = assemble_training(xtr, ytr, com)
    del xtr, ytr
    official_val_x = prepare_features(xva, com, 'official Validation')
    del xva
    writer_features = {w: prepare_features(writers[w][0], com, f'writer {w}') for w in WRITERS}
    if official_x.shape != (107520, 1108) or official_val_x.shape != (19200, 1108):
        raise RuntimeError('Frozen official feature shapes mismatch')

    # Recheck frozen Normal; never evaluate official Test.
    normal = fresh_knn()
    normal.fit(official_x, official_y)
    normal_score = summary(yva, normal.predict(official_val_x))
    if normal_score['correct'] != EXPECTED_VALIDATION_CORRECT:
        raise RuntimeError('Final Normal 20:80 official Validation did not reproduce')
    del normal
    print('Frozen Normal official Validation EXACTLY reproduced', flush=True)

    fold_results = []
    predictions = {variant_name(w): {} for w in PIXEL_WEIGHTS}
    augmentation_records = []
    for heldout in WRITERS:
        train_writers = [w for w in WRITERS if w != heldout]
        if len(train_writers) != 3 or heldout in train_writers:
            raise RuntimeError('Held-out writer leakage')
        raw_x = np.concatenate([writers[w][0] for w in train_writers])
        raw_y = np.concatenate([writers[w][1] for w in train_writers])
        ids = np.concatenate([np.full(640, w, dtype=np.int32) for w in train_writers])
        original_x = np.concatenate([writer_features[w] for w in train_writers])
        if raw_y.shape != (1920,) or original_x.shape != (1920, 1108):
            raise RuntimeError('Hand-made training count mismatch')

        selected = balanced_nested_indices(ids, raw_y, seed=AUG_SEED + heldout)
        aug_indices = selected[100]
        aug_x, aug_info = make_aug_features(raw_x, aug_indices, com, 100, heldout)
        aug_y = raw_y[aug_indices]
        original_25 = selected[25]
        augmented_other_25 = np.setdiff1d(selected[50], selected[25], assume_unique=True)
        if (len(aug_indices), len(original_25), len(augmented_other_25)) != (1920, 480, 480):
            raise RuntimeError('Inconsistent previous experiment augmentation sampling')

        base_x = np.concatenate((
            official_x, original_x, aug_x,
            original_x[original_25], aug_x[augmented_other_25],
        ), axis=0)
        base_y = np.concatenate((
            official_y, raw_y, aug_y,
            raw_y[original_25], aug_y[augmented_other_25],
        ), axis=0)
        if base_x.shape != (112320, 1108) or base_y.shape != (112320,):
            raise RuntimeError('EXP-CMB-004 repeat_mixed_50 training shape mismatch')
        print(f'\n[Fold heldout={heldout}] train writers={train_writers}; frozen rows=112320', flush=True)
        for w in PIXEL_WEIGHTS:
            variant = variant_name(w)
            train_features = remix(base_x, w)
            test_features = remix(writer_features[heldout], w)
            val_features = remix(official_val_x, w)
            model = fresh_knn()
            t = time.perf_counter()
            model.fit(train_features, base_y)
            fit_seconds = time.perf_counter() - t
            t = time.perf_counter()
            pred_hand = np.asarray(model.predict(test_features), dtype=np.int64)
            hand_seconds = time.perf_counter() - t
            t = time.perf_counter()
            pred_official = np.asarray(model.predict(val_features), dtype=np.int64)
            official_seconds = time.perf_counter() - t
            del model, train_features, test_features, val_features
            hand = summary(writers[heldout][1], pred_hand)
            official = summary(yva, pred_official)
            if w == CONTROL_WEIGHT and hand['correct'] != CONTROL_BY_WRITER[heldout]:
                raise RuntimeError(f'EXP-CMB-004 20:80 control mismatch heldout={heldout}: '
                                   f"expected {CONTROL_BY_WRITER[heldout]}, got {hand['correct']}")
            predictions[variant][heldout] = pred_hand
            row = {
                'variant': variant, 'pixel_weight': w, 'hog_weight': 1 - w,
                'heldout_writer': heldout,
                'train_writers': ','.join(map(str, train_writers)),
                'official_derived': 107520, 'handmade_original': 1920,
                'handmade_augmented': 1920, 'handmade_repeated': 960,
                'train_rows': len(base_y),
                'handmade_correct': hand['correct'], 'handmade_accuracy': hand['accuracy'],
                'handmade_macro_f1': hand['macro_f1'],
                'official_correct': official['correct'],
                'official_accuracy': official['accuracy'],
                'official_macro_f1': official['macro_f1'],
                'guardrail_pass': int(official['accuracy'] >= OFFICIAL_GUARDRAIL),
                'fit_seconds': round(fit_seconds, 3),
                'handmade_predict_seconds': round(hand_seconds, 3),
                'official_predict_seconds': round(official_seconds, 3),
            }
            fold_results.append(row)
            write_partial(fold_results)
            print(f'  {variant:21s} OOF_fold={hand["accuracy"]:.6f} '
                  f'Official={official["accuracy"]:.6f} guardrail='
                  f'{"PASS" if row["guardrail_pass"] else "FAIL"}', flush=True)
        augmentation_records.append({'heldout_writer': heldout, 'train_writers': train_writers,
                                     'augmented_rows': 1920, 'repeated_rows': 960,
                                     'augmentation': aug_info})
        del base_x, base_y, raw_x, raw_y, original_x, aug_x, aug_y

    truth = np.concatenate([writers[w][1] for w in WRITERS])
    reference = np.concatenate([predictions[variant_name(CONTROL_WEIGHT)][w] for w in WRITERS])
    if int(np.sum(reference == truth)) != CONTROL_TOTAL_CORRECT:
        raise RuntimeError('EXP-CMB-004 repeat_mixed_50 OOF reference mismatch')

    candidates = []
    all_preds = []
    aggregates = {}
    for w in PIXEL_WEIGHTS:
        variant = variant_name(w)
        pred = np.concatenate([predictions[variant][writer] for writer in WRITERS])
        result = summary(truth, pred)
        paired = paired_counts(truth, reference, pred)
        min_official = min(row['official_accuracy'] for row in fold_results if row['variant'] == variant)
        eligible = min_official >= OFFICIAL_GUARDRAIL
        candidates.append({
            'variant': variant, 'pixel_weight': w, 'hog_weight': 1 - w,
            'oof_correct': result['correct'], 'oof_accuracy': result['accuracy'],
            'oof_macro_f1': result['macro_f1'],
            'delta_pp_vs_20_80': 100 * (result['accuracy'] - CONTROL_TOTAL_CORRECT / 2560),
            'min_official_accuracy': min_official, 'official_guardrail_pass': int(eligible),
            'paired_corrected': paired['corrected'], 'paired_new_errors': paired['new_errors'],
        })
        aggregates[variant] = {'oof': result, 'paired_vs_20_80': paired}
        save_cm(OUT / f'{variant}_oof_confusion_matrix.csv', truth, pred)
        save_csv(OUT / f'{variant}_per_class_oof.csv', [
            {'class': label, **result['per_class'][label]} for label in LABELS
        ])
        for i, heldout in enumerate(WRITERS):
            for sample in range(640):
                j = i * 640 + sample
                all_preds.append({'variant': variant, 'heldout_writer': heldout,
                                  'index': sample, 'true': LABELS[int(truth[j])],
                                  'predicted': LABELS[int(pred[j])],
                                  'correct': int(truth[j] == pred[j])})

    valid = [r for r in candidates if r['official_guardrail_pass']]
    if not valid:
        raise RuntimeError('All feature ratios failed official Validation guardrail')
    winner = max(valid, key=lambda r: (r['oof_macro_f1'], r['oof_accuracy'],
                                       r['min_official_accuracy']))
    save_csv(OUT / 'candidate_results.csv', candidates)
    save_csv(OUT / 'fold_results.csv', fold_results)
    save_csv(OUT / 'official_validation.csv', [
        {'variant': r['variant'], 'heldout_writer': r['heldout_writer'],
         'official_correct': r['official_correct'], 'official_accuracy': r['official_accuracy'],
         'official_macro_f1': r['official_macro_f1']} for r in fold_results
    ])
    save_csv(OUT / 'oof_predictions.csv', all_preds)
    report = {
        'experiment': 'EXP-CMB-005 Pixel/HOG mix ratio (writer-held-out development OOF)',
        'official_dataset_sha256': official_sha,
        'handmade_sha256': {str(w): writers[w][2] for w in WRITERS},
        'control_reference': 'EXP-CMB-004 repeat_mixed_50: 2403/2560 correct',
        'training_rows_per_fold': 112320,
        'ratios': [{'pixel': w, 'hog': 1 - w} for w in PIXEL_WEIGHTS],
        'fold_results': fold_results, 'candidate_results': candidates,
        'aggregates': aggregates,
        'winner_development_only': winner['variant'],
        'augmentation_records': augmentation_records,
        'official_validation_guardrail_min_accuracy': OFFICIAL_GUARDRAIL,
        'official_test_used': False,
        'warning': 'OOF is DEVELOPMENT ONLY: held-out writer is absent from fold fit, '
                   'but these four writers were repeatedly used to select Normal/Combined settings. '
                   'Do not treat these OOF values as independent final Test accuracy.',
        'environment': {'python': platform.python_version(),
                        'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'elapsed_seconds': round(time.perf_counter() - start, 2),
    }
    (OUT / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nEXP-CMB-005 SUMMARY (4-writer DEVELOPMENT OOF)', flush=True)
    for row in candidates:
        print(f"{row['variant']:21s} OOF={row['oof_accuracy']:.6f} "
              f"F1={row['oof_macro_f1']:.6f} official_min={row['min_official_accuracy']:.6f} "
              f"guardrail={'PASS' if row['official_guardrail_pass'] else 'FAIL'}", flush=True)
    print('SELECTED DEVELOPMENT CANDIDATE:', winner['variant'], flush=True)
    print('Saved:', OUT, flush=True)
    print('Official Test was not accessed.', flush=True)


if __name__ == '__main__':
    main()
