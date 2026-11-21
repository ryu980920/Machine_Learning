#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-008: CoM KNN fixed k=3 distance; training-only augmentation-ratio comparison.

Place beside EXP004_KNN_Normal_CoM.py,
EXP006_KNN_Normal_BB_Size_fixed.py, and EXP007_KNN_Normal_Augmentation.py.
From repository root run:
    python 05_model_optimize/KNN/Normal/scripts/EXP008_KNN_Normal_Aug_Ratio.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
import platform
from pathlib import Path

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'
sys.path.insert(0, str(ROOT / '03_experiments' / 'model_optimization'))
sys.path.insert(0, str(SCRIPTS))

from common import fixed_split, LABELS, scores
from EXP004_KNN_Normal_CoM import BatchedCoM, read_handmade, paired_stats
from EXP007_KNN_Normal_Augmentation import (
    knn_model, sample_balanced, size_jitter, stroke_jitter,
    test_augmentation, save_csv, save_cm,
    EXPECTED_OFFICIAL_CORRECT, EXPECTED_HANDMADE_CORRECT,
    OFFICIAL_GUARDRAIL, SEED, TARGET_LONG_SIDES, STROKE_STRENGTH,
)

OUTPUT = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'EXP008_aug_ratio'
FRACTIONS = (0.10, 0.20, 0.30)


def main():
    test_augmentation()
    xtr, ytr, xva, yva, digest = fixed_split()
    xhm, yhm, writer_ids, sample_ids = read_handmade()
    if (len(ytr), len(yva), len(yhm)) != (76800, 19200, 2560):
        raise ValueError('Unexpected dataset sizes')
    com = BatchedCoM()
    xt = com.transform(xtr)
    xv = com.transform(xva)
    xh = com.transform(xhm)
    print('Training=76800 / Official Validation=19200 / Hand-made=2560', flush=True)

    configurations = [('control_com', 0.0, 'none')]
    for fraction in FRACTIONS:
        configurations.append((f'size_{int(fraction * 100):02d}', fraction, 'size'))
    for fraction in FRACTIONS:
        configurations.append((f'combined_{int(fraction * 100):02d}', fraction, 'combined'))

    candidate_rows = []
    writer_rows = []
    class_rows = []
    prediction_rows = []
    details = {}
    predictions = {}
    control_official = control_handmade = None
    stats_by_fraction = {}

    for name, fraction, kind in configurations:
        print(f'\n[{name}] Preparing training rows...', flush=True)
        if kind == 'none':
            xx, yy = xt, ytr
            augmentation_n = 0
        else:
            idx = sample_balanced(ytr, fraction=fraction, seed=SEED)
            # Use the same fixed size choices and deterministic random seed as EXP-007.
            size_values, counts = size_jitter(xtr, idx, seed=SEED)
            stats_by_fraction[f'{kind}_{fraction}'] = {'sizes': counts}
            if kind == 'combined':
                aug, stroke_stats = stroke_jitter(size_values, strength=STROKE_STRENGTH)
                stats_by_fraction[f'{kind}_{fraction}']['stroke'] = stroke_stats
            else:
                aug = size_values
            xx = np.concatenate([xt, com.transform(aug)], axis=0)
            yy = np.concatenate([ytr, ytr[idx]], axis=0)
            augmentation_n = len(idx)
        model = knn_model()
        start = time.perf_counter()
        model.fit(xx, yy)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        op = model.predict(xv).astype(np.int64)
        official_seconds = time.perf_counter() - start
        start = time.perf_counter()
        hp = model.predict(xh).astype(np.int64)
        handmade_seconds = time.perf_counter() - start
        official = scores(yva, op)
        handmade = scores(yhm, hp)
        official_correct = int(np.sum(op == yva))
        handmade_correct = int(np.sum(hp == yhm))
        if name == 'control_com':
            if official_correct != EXPECTED_OFFICIAL_CORRECT or handmade_correct != EXPECTED_HANDMADE_CORRECT:
                raise RuntimeError('EXP-005 control mismatch; stop and inspect environment')
            control_official, control_handmade = op.copy(), hp.copy()
            print('EXP-005 control reproduced exactly', flush=True)
        if name in ('size_20', 'combined_20'):
            expected = {'size_20': (18470, 2110), 'combined_20': (18401, 2123)}[name]
            if (official_correct, handmade_correct) != expected:
                raise RuntimeError(f'EXP-007 {name} replication mismatch: {(official_correct, handmade_correct)} vs {expected}')
            print('EXP-007 20% reference reproduced exactly', flush=True)
        pass_guard = official['accuracy'] >= OFFICIAL_GUARDRAIL
        row = {
            'variant': name, 'augmentation_fraction': fraction, 'kind': kind,
            'train_n': len(yy), 'augmented_n': augmentation_n,
            'official_correct': official_correct, 'official_accuracy': official['accuracy'],
            'official_macro_f1': official['macro_f1'],
            'handmade_correct': handmade_correct, 'handmade_accuracy': handmade['accuracy'],
            'handmade_macro_f1': handmade['macro_f1'],
            'official_guardrail_pass': int(pass_guard),
            'official_delta_pp_vs_control': 100 * (official['accuracy'] - EXPECTED_OFFICIAL_CORRECT / 19200),
            'handmade_delta_pp_vs_control': 100 * (handmade['accuracy'] - EXPECTED_HANDMADE_CORRECT / 2560),
            'fit_seconds': fit_seconds, 'official_predict_seconds': official_seconds,
            'handmade_predict_seconds': handmade_seconds,
        }
        candidate_rows.append(row)
        details[name] = {
            'official': {'accuracy': official['accuracy'], 'macro_f1': official['macro_f1'],
                         'confusion_matrix': official['confusion_matrix']},
            'handmade': {'accuracy': handmade['accuracy'], 'macro_f1': handmade['macro_f1'],
                         'confusion_matrix': handmade['confusion_matrix']},
        }
        predictions[name] = (op, hp)
        for w in (41, 42, 43, 44):
            mask = writer_ids == w
            sc = scores(yhm[mask], hp[mask])
            writer_rows.append({'variant': name, 'writer': w, 'n': int(mask.sum()),
                                'accuracy': sc['accuracy'], 'macro_f1': sc['macro_f1']})
        for label_name in LABELS:
            p = handmade['per_class'][label_name]
            class_rows.append({'variant': name, 'class': label_name, 'support': p['support'],
                               'precision': p['precision'], 'recall': p['recall'], 'f1': p['f1']})
        print(f"{name:<15} train={len(yy)} | Official={official['accuracy']:.6f} "
              f"F1={official['macro_f1']:.6f} | Hand-made={handmade['accuracy']:.6f} "
              f"F1={handmade['macro_f1']:.6f} | guardrail={'PASS' if pass_guard else 'FAIL'}", flush=True)
        del model, xx, yy

    valid = [row for row in candidate_rows if row['official_guardrail_pass']]
    winner = max(valid, key=lambda row: (row['handmade_macro_f1'], row['handmade_accuracy'],
                                        row['official_macro_f1'])) if valid else None
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_csv(OUTPUT / 'candidate_results.csv', candidate_rows)
    save_csv(OUTPUT / 'by_writer.csv', writer_rows)
    save_csv(OUTPUT / 'per_class_handmade.csv', class_rows)
    for name in predictions:
        op, hp = predictions[name]
        save_cm(OUTPUT / f'official_cm_{name}.csv', details[name]['official']['confusion_matrix'])
        save_cm(OUTPUT / f'handmade_cm_{name}.csv', details[name]['handmade']['confusion_matrix'])
        for i, pred in enumerate(hp):
            prediction_rows.append({'variant': name, 'writer': int(writer_ids[i]),
                                    'index': int(sample_ids[i]), 'true': LABELS[int(yhm[i])],
                                    'predicted': LABELS[int(pred)], 'correct': int(pred == yhm[i])})
    save_csv(OUTPUT / 'handmade_predictions.csv', prediction_rows)
    paired = {
        name: {
            'official': paired_stats(yva, control_official, predictions[name][0]),
            'handmade': paired_stats(yhm, control_handmade, predictions[name][1]),
        } for name in predictions if name != 'control_com'
    }
    result = {
        'experiment': 'EXP-008 KNN Normal augmentation fraction search',
        'reference': 'EXP-005 CoM k=3 distance', 'source_augmentation': 'EXP007_KNN_Normal_Augmentation.py',
        'official_dataset_sha256': digest, 'train_n': len(ytr), 'validation_n': len(yva),
        'handmade_n': len(yhm), 'seed': SEED, 'fractions': list(FRACTIONS),
        'target_long_sides': list(TARGET_LONG_SIDES), 'stroke_strength': STROKE_STRENGTH,
        'guardrail_official_accuracy': OFFICIAL_GUARDRAIL,
        'knn': {'n_neighbors': 3, 'weights': 'distance', 'metric': 'minkowski', 'p': 2},
        'candidates': candidate_rows, 'selected_development_candidate': winner,
        'pairwise_vs_control': paired, 'augmentation_stats': stats_by_fraction,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'sklearn': sklearn.__version__},
        'note': 'Training-only augmentation. Hand-made used for model development comparison; not final holdout. Original Test and final Test untouched.',
    }
    (OUTPUT / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nSELECTED DEVELOPMENT CANDIDATE:', winner['variant'] if winner else 'NONE', flush=True)
    if winner:
        print(f"Official={winner['official_accuracy']:.6f}, Hand-made={winner['handmade_accuracy']:.6f}, "
              f"MacroF1={winner['handmade_macro_f1']:.6f}", flush=True)
    print('Saved:', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
