#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-CMB-004: training-writer sample duplication, 4-writer development OOF.

Place in 05_model_optimize/KNN/Combined/scripts/ and run from the repo root.
Prerequisites: existing Normal scripts and EXP_CMB_001/002/003 scripts.
Official Test is never accessed. No held-out writer samples enter model fit.
Duplicates are intentionally exact; with k=3 they may monopolize the vote.
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
for folder in (NORMAL_DIR, COMBINED_DIR, ROOT / '03_experiments' / 'model_optimization'):
    sys.path.insert(0, str(folder))

from common import LABELS, fixed_split
from EXP004_KNN_Normal_CoM import BatchedCoM
from final_knn_normal import assemble_training, prepare_features, EXPECTED_VALIDATION_CORRECT
from EXP_CMB_001_KNN_Combined_Baseline import WRITERS, load_writer, save_csv, save_cm, summary
from EXP_CMB_002_KNN_Combined_Optimized import fresh_knn, paired_counts
from EXP_CMB_003_KNN_Combined_Handmade_Augmentation import (
    SEED as AUG_SEED, balanced_nested_indices, make_aug_features,
)

OUT = ROOT / '05_model_optimize' / 'KNN' / 'Combined' / 'results' / 'EXP_CMB_004_sample_influence'
VARIANTS = ('control', 'repeat_original_50', 'repeat_augmented_50', 'repeat_mixed_50')
EXPECTED_CONTROL_BY_WRITER = {41: 587, 42: 596, 43: 621, 44: 597}
EXPECTED_CONTROL_OOF_CORRECT = 2401
MIN_OFFICIAL_VALIDATION_ACCURACY = 0.978


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    xtr, ytr, xva, yva, official_sha = fixed_split()
    if (len(ytr), len(yva)) != (76800, 19200):
        raise RuntimeError('Official Training/Validation split differs from baseline')
    writers = {w: load_writer(w) for w in WRITERS}
    com = BatchedCoM()
    print('Official Test NOT loaded. Frozen Normal feature pipeline...', flush=True)
    official_x, official_y = assemble_training(xtr, ytr, com)
    del xtr, ytr
    if official_x.shape != (107520, 1108):
        raise RuntimeError('Official-derived feature shape mismatch')
    official_val_x = prepare_features(xva, com, 'official Validation')
    del xva
    writer_features = {w: prepare_features(writers[w][0], com, f'writer {w}') for w in WRITERS}

    # Reproduce the frozen Normal official Validation before investigating variants.
    normal = fresh_knn()
    normal.fit(official_x, official_y)
    official_control = summary(yva, normal.predict(official_val_x))
    if official_control['correct'] != EXPECTED_VALIDATION_CORRECT:
        raise RuntimeError('Frozen Normal official Validation control was not reproduced')
    del normal

    predictions_by_variant = {name: {} for name in VARIANTS}
    fold_results = []
    records = []
    for heldout in WRITERS:
        train_writers = [w for w in WRITERS if w != heldout]
        if heldout in train_writers or len(train_writers) != 3:
            raise RuntimeError('Held-out writer data leakage')
        raw_x = np.concatenate([writers[w][0] for w in train_writers])
        raw_y = np.concatenate([writers[w][1] for w in train_writers])
        ids = np.concatenate([np.full(640, w, dtype=np.int32) for w in train_writers])
        original_x = np.concatenate([writer_features[w] for w in train_writers])
        if raw_y.shape != (1920,) or original_x.shape != (1920, 1108):
            raise RuntimeError('Hand-made training shape mismatch')
        # Use EXACT nested sampling and deterministic augmented rows from EXP-CMB-003.
        selected = balanced_nested_indices(ids, raw_y, seed=AUG_SEED + heldout)
        all_indices = selected[100]
        aug_x, aug_info = make_aug_features(raw_x, all_indices, com, 100, heldout)
        aug_y = raw_y[all_indices]
        base_x = np.concatenate((official_x, original_x, aug_x), axis=0)
        base_y = np.concatenate((official_y, raw_y, aug_y), axis=0)
        if base_x.shape != (111360, 1108) or base_y.shape != (111360,):
            raise RuntimeError('EXP-CMB-003 100% reference training shape mismatch')

        # 20 / class / writer = 960 samples. Mixed is 10 original and 10 augmented
        # per writer/class, hence all three variants add an equal 960 rows.
        original_50 = selected[50]
        augmented_50 = selected[50]
        original_25 = selected[25]
        augmented_other_25 = np.setdiff1d(selected[50], selected[25], assume_unique=True)
        if (len(original_50), len(augmented_50), len(original_25), len(augmented_other_25)) != (960, 960, 480, 480):
            raise RuntimeError('Invalid balanced sample counts')
        extra_parts = {
            'control': (None, None),
            'repeat_original_50': (original_x[original_50], raw_y[original_50]),
            'repeat_augmented_50': (aug_x[augmented_50], aug_y[augmented_50]),
            'repeat_mixed_50': (
                np.concatenate((original_x[original_25], aug_x[augmented_other_25])),
                np.concatenate((raw_y[original_25], aug_y[augmented_other_25])),
            ),
        }
        print(f'\n[Fold heldout={heldout}] train writers={train_writers}; control rows=111360', flush=True)
        for variant in VARIANTS:
            extra_x, extra_y = extra_parts[variant]
            fit_x = base_x if extra_x is None else np.concatenate((base_x, extra_x), axis=0)
            fit_y = base_y if extra_y is None else np.concatenate((base_y, extra_y), axis=0)
            expected_rows = 111360 if variant == 'control' else 112320
            if fit_x.shape != (expected_rows, 1108) or fit_y.shape != (expected_rows,):
                raise RuntimeError('Training row count mismatch')
            model = fresh_knn()
            t0 = time.perf_counter()
            model.fit(fit_x, fit_y)
            fit_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            pred_hand = model.predict(writer_features[heldout]).astype(np.int64)
            predict_hand_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            pred_official = model.predict(official_val_x).astype(np.int64)
            predict_official_s = time.perf_counter() - t0
            del model
            hand_stat = summary(writers[heldout][1], pred_hand)
            official_stat = summary(yva, pred_official)
            if variant == 'control' and hand_stat['correct'] != EXPECTED_CONTROL_BY_WRITER[heldout]:
                raise RuntimeError(f'EXP-CMB-003 control mismatch for heldout={heldout}: '
                                   f"{hand_stat['correct']} != {EXPECTED_CONTROL_BY_WRITER[heldout]}")
            predictions_by_variant[variant][heldout] = pred_hand
            row = {
                'variant': variant, 'heldout_writer': heldout,
                'train_writers': ','.join(map(str, train_writers)),
                'official_derived': 107520, 'handmade_original': 1920,
                'handmade_augmented': 1920, 'repeated_rows': expected_rows - 111360,
                'train_rows': expected_rows, 'handmade_correct': hand_stat['correct'],
                'handmade_accuracy': hand_stat['accuracy'], 'handmade_macro_f1': hand_stat['macro_f1'],
                'official_correct': official_stat['correct'],
                'official_accuracy': official_stat['accuracy'],
                'official_macro_f1': official_stat['macro_f1'],
                'official_guardrail_pass': int(official_stat['accuracy'] >= MIN_OFFICIAL_VALIDATION_ACCURACY),
                'fit_seconds': round(fit_seconds, 3),
                'handmade_predict_seconds': round(predict_hand_s, 3),
                'official_predict_seconds': round(predict_official_s, 3),
            }
            fold_results.append(row)
            save_csv(OUT / 'fold_results_partial.csv', fold_results)
            print(f'  {variant:21s} train={expected_rows} | OOF_fold={hand_stat["accuracy"]:.6f} '
                  f'Official={official_stat["accuracy"]:.6f} | guardrail='
                  f'{"PASS" if row["official_guardrail_pass"] else "FAIL"}', flush=True)
        records.append({'heldout_writer': heldout, 'train_writers': train_writers,
                        'augmentation': aug_info, 'repeat_count': 960})
        del base_x, base_y, raw_x, raw_y, aug_x, aug_y, original_x, extra_parts

    truth = np.concatenate([writers[w][1] for w in WRITERS])
    control_pred = np.concatenate([predictions_by_variant['control'][w] for w in WRITERS])
    if int((control_pred == truth).sum()) != EXPECTED_CONTROL_OOF_CORRECT:
        raise RuntimeError('EXP-CMB-003 100% OOF summary did not reproduce')
    result_rows = []
    aggregate = {}
    all_predictions = []
    for variant in VARIANTS:
        pred = np.concatenate([predictions_by_variant[variant][w] for w in WRITERS])
        s = summary(truth, pred)
        paired = paired_counts(truth, control_pred, pred)
        official_min = min(row['official_accuracy'] for row in fold_results if row['variant'] == variant)
        result_rows.append({
            'variant': variant, 'oof_correct': s['correct'], 'oof_accuracy': s['accuracy'],
            'oof_macro_f1': s['macro_f1'], 'delta_pp_vs_control':
                100.0 * (s['accuracy'] - EXPECTED_CONTROL_OOF_CORRECT / 2560),
            'min_official_accuracy': official_min,
            'official_guardrail_pass': int(official_min >= MIN_OFFICIAL_VALIDATION_ACCURACY),
        })
        aggregate[variant] = {'oof': s, 'paired_vs_control': paired}
        save_cm(OUT / f'{variant}_oof_confusion_matrix.csv', truth, pred)
        save_csv(OUT / f'{variant}_per_class_oof.csv', [
            {'class': label, **s['per_class'][label]} for label in LABELS
        ])
        for offset, w in enumerate(WRITERS):
            for i in range(640):
                j = offset * 640 + i
                all_predictions.append({
                    'variant': variant, 'heldout_writer': w, 'index': i,
                    'true': LABELS[int(truth[j])], 'predicted': LABELS[int(pred[j])],
                    'correct': int(pred[j] == truth[j]),
                })
    eligible = [row for row in result_rows if row['official_guardrail_pass']]
    if not eligible:
        raise RuntimeError('No candidate passed the prespecified official guardrail')
    selected = max(eligible, key=lambda r: (r['oof_macro_f1'], r['oof_accuracy'],
                                            -int(r['variant'] != 'control')))
    save_csv(OUT / 'candidate_results.csv', result_rows)
    save_csv(OUT / 'fold_results.csv', fold_results)
    save_csv(OUT / 'official_validation.csv', [
        {k: row[k] for k in ('variant', 'heldout_writer', 'official_correct', 'official_accuracy',
                            'official_macro_f1')} for row in fold_results
    ])
    save_csv(OUT / 'oof_predictions.csv', all_predictions)
    report = {
        'experiment': 'EXP-CMB-004 Hand-made sample duplication influence (four-writer development OOF)',
        'official_dataset_sha256': official_sha,
        'handmade_sha256': {str(w): writers[w][2] for w in WRITERS},
        'control_reference': 'EXP-CMB-003 handmade_aug_100',
        'training_rows': {'control': 111360, 'duplicated_variants': 112320},
        'variants': list(VARIANTS),
        'method': '960 existing Hand-made source rows duplicated exactly; equal count across 3 variants; '
                  'writer/class balanced; k=3 distance Euclidean unchanged',
        'fold_results': fold_results,
        'candidate_results': result_rows,
        'aggregate': aggregate,
        'selected_development_candidate': selected['variant'],
        'augmentation_records': records,
        'official_guardrail_min_accuracy': MIN_OFFICIAL_VALIDATION_ACCURACY,
        'official_test_used': False,
        'warning': 'This is repeated-use DEVELOPMENT writer-held-out OOF, NOT a truly independent test. '
                   'Exact duplication can fill k=3 neighbor slots and alter majority voting.',
        'elapsed_seconds': round(time.perf_counter() - started, 2),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'sklearn': sklearn.__version__},
    }
    (OUT / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nEXP-CMB-004 SUMMARY (development 4-writer OOF)', flush=True)
    for row in result_rows:
        print(f"{row['variant']:21s} OOF={row['oof_accuracy']:.6f} "
              f"F1={row['oof_macro_f1']:.6f} official_min={row['min_official_accuracy']:.6f} "
              f"guardrail={'PASS' if row['official_guardrail_pass'] else 'FAIL'}", flush=True)
    print('SELECTED DEVELOPMENT CANDIDATE:', selected['variant'], flush=True)
    print('Saved:', OUT, flush=True)
    print('Official Test was not accessed.', flush=True)


if __name__ == '__main__':
    main()
