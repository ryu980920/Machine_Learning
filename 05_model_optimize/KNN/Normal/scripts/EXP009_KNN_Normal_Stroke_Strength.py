#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-009: training-only stroke intensity search with size augmentation fixed at 30%.

Place beside EXP004_KNN_Normal_CoM.py,
EXP006_KNN_Normal_BB_Size_fixed.py, and EXP007_KNN_Normal_Augmentation.py.
From repository root run:
    python 05_model_optimize/KNN/Normal/scripts/EXP009_KNN_Normal_Stroke_Strength.py
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

OUTPUT = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'EXP009_stroke_strength'
AUG_FRACTION = 0.30
STRENGTHS = (0.12, 0.20, 0.28, 0.36, 0.44)


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

    configurations = [('control_com', None), ('size_30', 0.0)]
    configurations += [(f'combined_strength_{int(round(strength * 100)):02d}', strength)
                       for strength in STRENGTHS]
    idx = sample_balanced(ytr, fraction=AUG_FRACTION, seed=SEED)
    size_values, size_counts = size_jitter(xtr, idx, seed=SEED)
    augmented_n = len(idx)
    print(f'Augmentation fixed at {AUG_FRACTION:.0%}: {augmented_n} training-only rows', flush=True)

    candidate_rows = []
    writer_rows = []
    class_rows = []
    prediction_rows = []
    details = {}
    predictions = {}
    control_official = control_handmade = None
    stats_by_strength = {}

    for name, strength in configurations:
        print(f'\n[{name}] Preparing training rows...', flush=True)
        if strength is None:
            xx, yy = xt, ytr
            augmentation_n = 0
        else:
            if strength == 0:
                aug = size_values
                stroke_stats = {'changed': 0, 'topology_rejected': 0}
            else:
                aug, stroke_stats = stroke_jitter(size_values, strength=strength)
            stats_by_strength[name] = {'size_counts': size_counts, 'stroke': stroke_stats}
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
        if name == 'combined_strength_28':
            expected = (18395, 2127)  # EXP-008 combined_30: official and Hand-made counts
            if (official_correct, handmade_correct) != expected:
                raise RuntimeError(f'EXP-008 combined_30 reference mismatch: {(official_correct, handmade_correct)} vs {expected}')
            print('EXP-008 combined_30 reference reproduced exactly', flush=True)
        pass_guard = official['accuracy'] >= OFFICIAL_GUARDRAIL
        row = {
            'variant': name, 'augmentation_fraction': 0.0 if strength is None else AUG_FRACTION,
            'stroke_strength': strength,
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
            'official': paired_stats(yva, predictions['combined_strength_28'][0], predictions[name][0]),
            'handmade': paired_stats(yhm, predictions['combined_strength_28'][1], predictions[name][1]),
        } for name in predictions if name != 'combined_strength_28'
    }
    result = {
        'experiment': 'EXP-009 KNN Normal stroke strength comparison',
        'reference': 'EXP-008 combined_30 stroke strength 0.28; EXP-005 CoM k=3 distance',
        'source_augmentation': 'EXP007_KNN_Normal_Augmentation.py',
        'official_dataset_sha256': digest, 'train_n': len(ytr), 'validation_n': len(yva),
        'handmade_n': len(yhm), 'seed': SEED, 'augmentation_fraction': AUG_FRACTION,
        'tested_stroke_strengths': list(STRENGTHS),
        'target_long_sides': list(TARGET_LONG_SIDES), 'reference_stroke_strength': STROKE_STRENGTH,
        'guardrail_official_accuracy': OFFICIAL_GUARDRAIL,
        'knn': {'n_neighbors': 3, 'weights': 'distance', 'metric': 'minkowski', 'p': 2},
        'candidates': candidate_rows, 'selected_development_candidate': winner,
        'pairwise_vs_combined_30': paired, 'augmentation_stats': stats_by_strength,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'sklearn': sklearn.__version__},
        'note': 'Training-only augmentation. Fixed original training and balanced size augmentation at 30%. Only stroke blend strength changes among combined candidates. Hand-made is development data, not independent final test. Original Test and final Test untouched.',
    }
    (OUTPUT / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nSELECTED DEVELOPMENT CANDIDATE:', winner['variant'] if winner else 'NONE', flush=True)
    if winner:
        print(f"Official={winner['official_accuracy']:.6f}, Hand-made={winner['handmade_accuracy']:.6f}, "
              f"MacroF1={winner['handmade_macro_f1']:.6f}", flush=True)
    print('Saved:', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
