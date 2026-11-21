#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-CMB-003: writer-balanced, training-only Hand-made augmentation.

Place in: 05_model_optimize/KNN/Combined/scripts/
Run from repository root:
 python 05_model_optimize/KNN/Combined/scripts/EXP_CMB_003_KNN_Combined_Handmade_Augmentation.py

Frozen optimized Normal source: 76,800 official Training + 30,720 official-derived
augmentation rows (107,520 total). Each of four writer-held-out folds adds
1,920 original samples from the other three writers. Control vs +480/+960/+1920
size/stroke-augmented Hand-made samples, respectively. Never loads official Test.
The 4-writer OOF is development-only: Normal features and targeted classes were
previously selected using these writers' development predictions.
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
for p in (NORMAL_DIR, COMBINED_DIR, ROOT / '03_experiments' / 'model_optimization'):
    sys.path.insert(0, str(p))

from common import LABELS, fixed_split
from EXP004_KNN_Normal_CoM import BatchedCoM
from EXP007_KNN_Normal_Augmentation import size_jitter, stroke_jitter
from final_knn_normal import assemble_training, prepare_features, EXPECTED_VALIDATION_CORRECT
from EXP_CMB_001_KNN_Combined_Baseline import WRITERS, load_writer, save_csv, save_cm, summary
from EXP_CMB_002_KNN_Combined_Optimized import fresh_knn, paired_counts

OUT = ROOT / '05_model_optimize' / 'KNN' / 'Combined' / 'results' / 'EXP_CMB_003_handmade_augmentation'
SEED = 20261123
AUG_RATIOS = (0, 25, 50, 100)
AUG_STRENGTH = 0.44
EXPECTED_CONTROL_WRITER_CORRECT = {41: 586, 42: 595, 43: 619, 44: 594}
EXPECTED_CONTROL_TOTAL = 2394
# Guardrail defined before evaluating the new variants, not changed afterward.
MIN_OFFICIAL_VALIDATION_ACCURACY = 0.978


def balanced_nested_indices(writer_ids, labels, seed):
    """Within each of 3 writers x 16 classes, deterministic 10/20/40 of 40.

    Ensures +25%, +50%, +100% are nested and no held-out writer is sampled.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for w in sorted(set(int(x) for x in writer_ids)):
        for c in range(16):
            idx = np.flatnonzero((writer_ids == w) & (labels == c))
            if len(idx) != 40:
                raise RuntimeError(f'Expected 40 samples for writer {w}, class {LABELS[c]}')
            parts.append(rng.permutation(idx))
    return {
        25: np.sort(np.concatenate([p[:10] for p in parts])),
        50: np.sort(np.concatenate([p[:20] for p in parts])),
        100: np.sort(np.concatenate(parts)),
    }


def make_aug_features(raw, selected, com, ratio, heldout):
    if len(selected) != (1920 * ratio // 100):
        raise RuntimeError(f'Invalid augmentation count for {ratio}%')
    resized, size_stats = size_jitter(raw, selected, seed=SEED + heldout + 100)
    strokes, stroke_stats = stroke_jitter(resized, strength=AUG_STRENGTH)
    del resized
    features = prepare_features(strokes, com, f'handmade augmented heldout={heldout} ratio={ratio}%')
    del strokes
    return features, {'size_counts': size_stats, 'stroke': stroke_stats}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    xtr, ytr, xva, yva, official_sha = fixed_split()
    if (len(ytr), len(yva)) != (76800, 19200):
        raise RuntimeError('Official split mismatch; refusing experiment')
    handmade = {w: load_writer(w) for w in WRITERS}
    com = BatchedCoM()
    print('Official Test NOT loaded. Preparing frozen optimized Normal features...', flush=True)
    official_x, official_y = assemble_training(xtr, ytr, com)
    del xtr, ytr
    if official_x.shape != (107520, 1108):
        raise RuntimeError('Frozen Normal train feature shape mismatch')
    official_val_x = prepare_features(xva, com, 'official Validation')
    del xva
    original_features = {w: prepare_features(handmade[w][0], com, f'writer {w}') for w in WRITERS}

    print('\n[Normal reference] Checking official Validation and 4-writer development data...', flush=True)
    normal = fresh_knn()
    normal.fit(official_x, official_y)
    reference_val = summary(yva, normal.predict(official_val_x).astype(np.int64))
    if reference_val['correct'] != EXPECTED_VALIDATION_CORRECT:
        raise RuntimeError('Official Validation frozen Normal reproduction failed')
    normal_by_writer = {w: normal.predict(original_features[w]).astype(np.int64) for w in WRITERS}
    normal_truth = np.concatenate([handmade[w][1] for w in WRITERS])
    normal_oof = np.concatenate([normal_by_writer[w] for w in WRITERS])
    normal_result = summary(normal_truth, normal_oof)
    if normal_result['correct'] != 2381:
        raise RuntimeError('Frozen optimized Normal handwriting reference mismatch')
    print('Frozen Normal EXACTLY reproduced: official=%0.6f, handmade=%0.6f' %
          (reference_val['accuracy'], normal_result['accuracy']), flush=True)
    del normal

    variants = {r: {} for r in AUG_RATIOS}
    metrics_by_fold = []
    official_rows = []
    augmentation_records = []
    for heldout in WRITERS:
        source_writers = [w for w in WRITERS if w != heldout]
        raw_source = np.concatenate([handmade[w][0] for w in source_writers], axis=0)
        raw_labels = np.concatenate([handmade[w][1] for w in source_writers])
        ids = np.concatenate([np.full(640, w, dtype=np.int32) for w in source_writers])
        original_x = np.concatenate([original_features[w] for w in source_writers])
        if original_x.shape != (1920, 1108) or set(np.unique(ids)) != set(source_writers):
            raise RuntimeError('Wrong training writers or source feature shape')
        selected = balanced_nested_indices(ids, raw_labels, seed=SEED + heldout)
        if not (set(selected[25]) <= set(selected[50]) <= set(selected[100])):
            raise RuntimeError('Augmentation indices not nested')
        print(f'\n[Fold heldout={heldout}] training writers={source_writers}, base=109440 rows', flush=True)
        for ratio in AUG_RATIOS:
            if ratio:
                new_x, aug_stats = make_aug_features(raw_source, selected[ratio], com, ratio, heldout)
                extra_y = raw_labels[selected[ratio]]
                fit_x = np.concatenate([official_x, original_x, new_x], axis=0)
                fit_y = np.concatenate([official_y, raw_labels, extra_y])
                del new_x
                augmentation_records.append({'heldout_writer': heldout, 'ratio': ratio,
                                             'source_writers': source_writers, **aug_stats})
            else:
                fit_x = np.concatenate([official_x, original_x], axis=0)
                fit_y = np.concatenate([official_y, raw_labels])
            expected_n = 109440 + (1920 * ratio // 100)
            if fit_x.shape != (expected_n, 1108) or fit_y.shape != (expected_n,):
                raise RuntimeError('Fit data dimension mismatch')
            model = fresh_knn()
            start = time.perf_counter()
            model.fit(fit_x, fit_y)
            fit_s = time.perf_counter() - start
            del fit_x, fit_y
            start = time.perf_counter()
            pred_hand = model.predict(original_features[heldout]).astype(np.int64)
            hand_s = time.perf_counter() - start
            start = time.perf_counter()
            pred_off = model.predict(official_val_x).astype(np.int64)
            off_s = time.perf_counter() - start
            del model
            hand_stats = summary(handmade[heldout][1], pred_hand)
            off_stats = summary(yva, pred_off)
            if ratio == 0 and hand_stats['correct'] != EXPECTED_CONTROL_WRITER_CORRECT[heldout]:
                raise RuntimeError(f'EXP-CMB-002 control mismatch for writer {heldout}: '
                                   f"{hand_stats['correct']} != {EXPECTED_CONTROL_WRITER_CORRECT[heldout]}")
            variants[ratio][heldout] = pred_hand
            row = {'variant': 'control' if ratio == 0 else f'handmade_aug_{ratio}',
                   'ratio_percent': ratio, 'heldout_writer': heldout,
                   'train_writers': ','.join(map(str, source_writers)),
                   'official_derived_train': 107520, 'original_handmade_train': 1920,
                   'augmented_handmade_train': ratio * 1920 // 100,
                   'total_train': expected_n,
                   'heldout_n': 640, 'handmade_correct': hand_stats['correct'],
                   'handmade_accuracy': hand_stats['accuracy'],
                   'handmade_macro_f1': hand_stats['macro_f1'],
                   'official_accuracy': off_stats['accuracy'],
                   'official_macro_f1': off_stats['macro_f1'],
                   'official_correct': off_stats['correct'],
                   'official_guardrail_pass': int(off_stats['accuracy'] >= MIN_OFFICIAL_VALIDATION_ACCURACY),
                   'fit_s': round(fit_s, 3), 'handmade_predict_s': round(hand_s, 3),
                   'official_predict_s': round(off_s, 3)}
            metrics_by_fold.append(row)
            official_rows.append({k: row[k] for k in ('variant', 'heldout_writer', 'official_correct',
                                                      'official_accuracy', 'official_macro_f1')})
            save_csv(OUT / 'fold_results_partial.csv', metrics_by_fold)
            save_csv(OUT / 'official_validation_partial.csv', official_rows)
            print(f'  ratio={ratio:3d}% train={expected_n}: '
                  f"handmade={hand_stats['accuracy']:.6f} "
                  f"official={off_stats['accuracy']:.6f} "
                  f"guardrail={'PASS' if row['official_guardrail_pass'] else 'FAIL'}", flush=True)
        del raw_source, raw_labels, original_x

    summary_rows = []
    aggregates = {}
    predictions = []
    for ratio in AUG_RATIOS:
        pred = np.concatenate([variants[ratio][w] for w in WRITERS])
        agg = summary(normal_truth, pred)
        paired = paired_counts(normal_truth, np.concatenate([variants[0][w] for w in WRITERS]), pred)
        min_official = min(r['official_accuracy'] for r in metrics_by_fold if r['ratio_percent'] == ratio)
        entry = {'variant': 'control' if ratio == 0 else f'handmade_aug_{ratio}',
                 'ratio_percent': ratio, 'correct': agg['correct'], 'accuracy': agg['accuracy'],
                 'macro_f1': agg['macro_f1'], 'min_fold_official_accuracy': min_official,
                 'official_guardrail_pass': int(min_official >= MIN_OFFICIAL_VALIDATION_ACCURACY),
                 'delta_pp_vs_control': 100 * (agg['accuracy'] - EXPECTED_CONTROL_TOTAL / 2560),
                 'paired_vs_control': paired}
        aggregates[entry['variant']] = {'oof': agg, 'paired_vs_control': paired}
        summary_rows.append({k: v for k, v in entry.items() if k != 'paired_vs_control'})
        save_cm(OUT / f'{entry["variant"]}_oof_confusion_matrix.csv', normal_truth, pred)
        save_csv(OUT / f'{entry["variant"]}_per_class_oof.csv', [
            {'class': lbl, **agg['per_class'][lbl]} for lbl in LABELS
        ])
        for wi, w in enumerate(WRITERS):
            for i in range(640):
                ix = wi * 640 + i
                predictions.append({'variant': entry['variant'], 'writer': w, 'index': i,
                                    'true': LABELS[int(normal_truth[ix])],
                                    'predicted': LABELS[int(pred[ix])],
                                    'correct': int(pred[ix] == normal_truth[ix])})
    if aggregates['control']['oof']['correct'] != EXPECTED_CONTROL_TOTAL:
        raise RuntimeError('EXP-CMB-002 combined OOF control not reproduced')
    eligible = [r for r in summary_rows if r['official_guardrail_pass']]
    if not eligible:
        raise RuntimeError('All variants failed prespecified official Validation guardrail')
    best = max(eligible, key=lambda r: (r['macro_f1'], r['accuracy'], -r['ratio_percent']))
    save_csv(OUT / 'fold_results.csv', metrics_by_fold)
    save_csv(OUT / 'official_validation.csv', official_rows)
    save_csv(OUT / 'candidate_results.csv', summary_rows)
    save_csv(OUT / 'oof_predictions.csv', predictions)
    report = {
        'experiment': 'EXP-CMB-003: 3-writer Hand-made size+stroke augmentation ratios (4-writer development OOF)',
        'official_dataset_sha256': official_sha,
        'handmade_sha256': {str(w): handmade[w][2] for w in WRITERS},
        'official_training_source': 76800,
        'official_derived_training': 107520,
        'handmade_original_per_fold': 1920,
        'handmade_augmented_per_fold': {str(r): 1920 * r // 100 for r in AUG_RATIOS},
        'total_training_per_fold': {str(r): 109440 + (1920*r // 100) for r in AUG_RATIOS},
        'heldout_samples_per_fold': 640,
        'method': {'preprocessing': 'CoM -> Deskew', 'features': 'Pixel 20% + HOG 80%',
                   'k': 3, 'weights': 'distance', 'distance': 'euclidean',
                   'handmade_aug': 'EXP007 size_jitter + stroke_jitter; train writer/class balanced',
                   'stroke_strength': AUG_STRENGTH, 'seed': SEED},
        'normal_reference': normal_result,
        'normal_official_validation': reference_val,
        'candidate_oof': aggregates,
        'candidate_results': summary_rows,
        'fold_results': metrics_by_fold,
        'augmentation_details': augmentation_records,
        'development_candidate': best['variant'],
        'official_guardrail_min_accuracy': MIN_OFFICIAL_VALIDATION_ACCURACY,
        'official_test_used': False,
        'warning': 'All four handwriting writers were used in earlier Normal model selection. Writer-held-out fit is applied in these folds but this is repeated-use development OOF, NOT an independent writer test.',
        'elapsed_seconds': round(time.perf_counter()-t0, 2),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'sklearn': sklearn.__version__},
    }
    (OUT / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nEXP-CMB-003 SUMMARY (4-writer development OOF)', flush=True)
    for row in summary_rows:
        print(f"{row['variant']:17s} OOF={row['accuracy']:.6f} F1={row['macro_f1']:.6f} "
              f"official_min={row['min_fold_official_accuracy']:.6f} "
              f"guardrail={'PASS' if row['official_guardrail_pass'] else 'FAIL'}", flush=True)
    print('SELECTED DEVELOPMENT CANDIDATE:', best['variant'], flush=True)
    print('Saved:', OUT, flush=True)
    print('Official Test was not accessed.', flush=True)


if __name__ == '__main__':
    main()
