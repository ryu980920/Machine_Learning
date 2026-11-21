#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-010: compare KNN Minkowski distance metrics using EXP-009's exact training setup.

From repository root:
  python 05_model_optimize/KNN/Normal/scripts/EXP010_KNN_Normal_Distance_Metric.py

Default: pilot p=1,1.5,2,3 using the SAME stratified pilot train/eval subsets;
then exact full-set p=2 control AND the highest-ranked alternative by pilot
hand-made macro F1. Use --screen-only to skip expensive full evaluations.
Use --full-all to run all four metrics on all validation images (very slow).
"""
from __future__ import annotations

import argparse
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
    save_csv, save_cm, OFFICIAL_GUARDRAIL, SEED,
)

OUTPUT = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'EXP010_distance_metrics'
P_VALUES = (1.0, 1.5, 2.0, 3.0)
EXPECTED_FULL_P2 = (18381, 2153)  # EXP-009 0.44, 95.734375% and 84.1015625%


def stratified_indices(y, per_class, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for lab in range(16):
        candidates = np.flatnonzero(y == lab)
        if len(candidates) < per_class:
            raise ValueError(f'Class {lab} has insufficient samples')
        chosen.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def candidate_name(p):
    return 'p' + str(p).replace('.', '_')


def predict_variant(xtr, ytr, xva, yva, xhm, yhm, p, stage):
    model = KNeighborsClassifier(n_neighbors=3, weights='distance',
                                 metric='minkowski', p=p,
                                 algorithm='brute', n_jobs=1)
    t = time.perf_counter()
    model.fit(xtr, ytr)
    fit_sec = time.perf_counter() - t
    t = time.perf_counter()
    official_pred = model.predict(xva).astype(np.int64)
    off_sec = time.perf_counter() - t
    t = time.perf_counter()
    handmade_pred = model.predict(xhm).astype(np.int64)
    hm_sec = time.perf_counter() - t
    official = scores(yva, official_pred)
    handmade = scores(yhm, handmade_pred)
    rec = {
        'stage': stage, 'variant': candidate_name(p), 'p': p,
        'train_n': len(ytr), 'official_n': len(yva), 'handmade_n': len(yhm),
        'official_correct': int(np.sum(official_pred == yva)),
        'official_accuracy': official['accuracy'],
        'official_macro_f1': official['macro_f1'],
        'handmade_correct': int(np.sum(handmade_pred == yhm)),
        'handmade_accuracy': handmade['accuracy'],
        'handmade_macro_f1': handmade['macro_f1'],
        'official_guardrail_pass': (int(official['accuracy'] >= OFFICIAL_GUARDRAIL)
                                    if stage == 'full' else ''),
        'fit_seconds': fit_sec, 'official_predict_seconds': off_sec,
        'handmade_predict_seconds': hm_sec,
    }
    return rec, official_pred, handmade_pred, official, handmade


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen-only', action='store_true',
                        help='Only run stratified pilot on same subsets for all distances')
    parser.add_argument('--full-all', action='store_true',
                        help='Run all metrics on full dataset (computationally expensive)')
    parser.add_argument('--pilot-train-per-class', type=int, default=256,
                        help='Original + augmented pilot samples per class (default 256 total)')
    parser.add_argument('--pilot-official-per-class', type=int, default=32)
    parser.add_argument('--pilot-handmade-per-class-per-writer', type=int, default=4)
    args = parser.parse_args()
    if args.screen_only and args.full_all:
        parser.error('Choose only one of --screen-only and --full-all')
    if not 3 <= args.pilot_train_per_class <= 400:
        parser.error('--pilot-train-per-class must be between 3 and 400')
    if not 1 <= args.pilot_handmade_per_class_per_writer <= 40:
        parser.error('--pilot-handmade-per-class-per-writer must be between 1 and 40')

    test_augmentation()
    xtr, ytr, xva, yva, dataset_sha = fixed_split()
    xhm, yhm, writers, sample_ids = read_handmade()
    if (len(ytr), len(yva), len(yhm)) != (76800, 19200, 2560):
        raise RuntimeError('Dataset split mismatch')
    com = BatchedCoM()
    print('Preparing EXP-009 best: 30% combined augmentation, strength=0.44...', flush=True)
    idx = sample_balanced(ytr, fraction=0.30, seed=SEED)
    sized, size_counts = size_jitter(xtr, idx, seed=SEED)
    augmented, stroke_stats = stroke_jitter(sized, strength=0.44)
    train = np.concatenate((com.transform(xtr), com.transform(augmented)), axis=0)
    train_labels = np.concatenate((ytr, ytr[idx]))
    official = com.transform(xva)
    handmade = com.transform(xhm)
    del sized, augmented, xtr, xva, xhm
    train /= np.float32(255.0)
    official /= np.float32(255.0)
    handmade /= np.float32(255.0)
    if len(train_labels) != 99840:
        raise RuntimeError('Expected 99,840 training rows')
    print('Full train=99,840; official validation=19,200; handmade=2,560', flush=True)

    # Identical, deterministic stratified subsets across all pilot metrics.
    rng = np.random.default_rng(SEED + 100)
    pilot_tr = stratified_indices(train_labels, args.pilot_train_per_class, SEED + 101)
    pilot_va = stratified_indices(yva, args.pilot_official_per_class, SEED + 102)
    pilot_hm = []
    for writer in (41, 42, 43, 44):
        for lab in range(16):
            pool = np.flatnonzero((writers == writer) & (yhm == lab))
            pilot_hm.extend(rng.choice(pool, size=args.pilot_handmade_per_class_per_writer,
                                       replace=False).tolist())
    pilot_hm = np.asarray(sorted(pilot_hm), dtype=np.int64)
    print(f'PILOT ONLY: train={len(pilot_tr)}, official={len(pilot_va)}, handmade={len(pilot_hm)}', flush=True)
    print('Pilot accuracies are NOT comparable with full EXP-009 results.', flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pilot_rows = []
    full_rows = []
    full_predictions = {}
    full_detail = {}
    writer_rows = []
    class_rows = []
    handmade_prediction_rows = []
    for p in P_VALUES:
        print(f'\n[PILOT p={p:g}] Starting brute-force distance evaluation...', flush=True)
        result = predict_variant(train[pilot_tr], train_labels[pilot_tr],
                                 official[pilot_va], yva[pilot_va],
                                 handmade[pilot_hm], yhm[pilot_hm], p, 'pilot')
        row = result[0]
        pilot_rows.append(row)
        print(f"{row['variant']}: official={row['official_accuracy']:.6f}, "
              f"handmade={row['handmade_accuracy']:.6f}, "
              f"macroF1={row['handmade_macro_f1']:.6f}; "
              f"predict_s={row['official_predict_seconds'] + row['handmade_predict_seconds']:.1f}", flush=True)
        save_csv(OUTPUT / 'pilot_results.csv', pilot_rows)

    other = max((r for r in pilot_rows if r['p'] != 2.0),
                key=lambda r: (r['handmade_macro_f1'], r['handmade_accuracy'],
                               r['official_macro_f1']))
    full_ps = list(P_VALUES) if args.full_all else [2.0, other['p']]
    print('\nPilot selected alternative:', candidate_name(other['p']), flush=True)
    if args.screen_only:
        full_ps = []
        print('SCREEN-ONLY requested; no full-set comparison / no final model selection.', flush=True)
    else:
        print('FULL evaluation metrics:', ', '.join(map(str, full_ps)), flush=True)
        print('Fractional p metrics on full 99,840-row train can be VERY slow.', flush=True)
    for p in full_ps:
        print(f'\n[FULL p={p:g}] Running on complete validation and handwritten data...', flush=True)
        result = predict_variant(train, train_labels, official, yva,
                                 handmade, yhm, p, 'full')
        row, op, hp, off, hm = result
        if p == 2.0 and (row['official_correct'], row['handmade_correct']) != EXPECTED_FULL_P2:
            raise RuntimeError(f'EXP-009 p=2 control mismatch: '
                               f"{(row['official_correct'], row['handmade_correct'])} "
                               f'!= {EXPECTED_FULL_P2}')
        full_rows.append(row)
        name = candidate_name(p)
        full_predictions[name] = (op, hp)
        full_detail[name] = {'official': off, 'handmade': hm}
        for writer in (41, 42, 43, 44):
            m = writers == writer
            wr = scores(yhm[m], hp[m])
            writer_rows.append({'variant': name, 'writer': writer, 'n': int(m.sum()),
                                'accuracy': wr['accuracy'], 'macro_f1': wr['macro_f1']})
        for lab in LABELS:
            item = hm['per_class'][lab]
            class_rows.append({'variant': name, 'class': lab,
                               'support': item['support'], 'recall': item['recall'],
                               'precision': item['precision'], 'f1': item['f1']})
        for i in range(len(yhm)):
            handmade_prediction_rows.append({'variant': name, 'writer': int(writers[i]),
                                            'index': int(sample_ids[i]), 'true': LABELS[int(yhm[i])],
                                            'predicted': LABELS[int(hp[i])],
                                            'correct': int(hp[i] == yhm[i])})
        save_cm(OUTPUT / f'official_cm_{name}.csv', off['confusion_matrix'])
        save_cm(OUTPUT / f'handmade_cm_{name}.csv', hm['confusion_matrix'])
        save_csv(OUTPUT / 'full_results.csv', full_rows)
        print(f"{name}: official={row['official_accuracy']:.6f} F1={row['official_macro_f1']:.6f}, "
              f"handmade={row['handmade_accuracy']:.6f} F1={row['handmade_macro_f1']:.6f}, "
              f"guardrail={'PASS' if row['official_guardrail_pass'] else 'FAIL'}", flush=True)

    selected = None
    if full_rows:
        valid = [r for r in full_rows if r['official_guardrail_pass']]
        if valid:
            selected = max(valid, key=lambda r: (r['handmade_macro_f1'],
                                                  r['handmade_accuracy'],
                                                  r['official_macro_f1']))
        if writer_rows:
            save_csv(OUTPUT / 'by_writer.csv', writer_rows)
            save_csv(OUTPUT / 'per_class_handmade.csv', class_rows)
            save_csv(OUTPUT / 'handmade_predictions.csv', handmade_prediction_rows)
    paired = {}
    if 'p2_0' in full_predictions:
        baseline_off, baseline_hm = full_predictions['p2_0']
        for name, (op, hp) in full_predictions.items():
            if name != 'p2_0':
                paired[name] = {'official': paired_stats(yva, baseline_off, op),
                                'handmade': paired_stats(yhm, baseline_hm, hp)}
    summary = {
        'experiment': 'EXP-010 KNN Normal Minkowski p comparison',
        'official_dataset_sha256': dataset_sha,
        'train_n': len(train_labels), 'validation_n': len(yva), 'handmade_n': len(yhm),
        'augmentation': {'ratio': 0.30, 'stroke_strength': 0.44,
                         'sizes': size_counts, 'stroke_stats': stroke_stats},
        'knn': {'n_neighbors': 3, 'weights': 'distance', 'metric': 'minkowski',
                'algorithm': 'brute'},
        'pilot': {'train_n': len(pilot_tr), 'official_n': len(pilot_va),
                  'handmade_n': len(pilot_hm), 'rows': pilot_rows,
                  'note': 'Pilot uses a reduced training AND evaluation subset. '
                          'Do not compare pilot accuracy with EXP-009 full-set accuracy.'},
        'full': {'evaluated_p': full_ps, 'rows': full_rows,
                 'selected_development_candidate': selected,
                 'paired_vs_p2': paired},
        'guardrail_full_official_accuracy': OFFICIAL_GUARDRAIL,
        'environment': {'python': platform.python_version(),
                        'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'note': 'Official Training augmentation only. Hand-made used for development '
                'selection only. Held-out original/final Test not accessed.',
    }
    (OUTPUT / 'metrics.json').write_text(json.dumps(summary, indent=2,
                                                   ensure_ascii=False), encoding='utf-8')
    print('\nSELECTED FULL DEVELOPMENT CANDIDATE:',
          selected['variant'] if selected else 'NONE', flush=True)
    print('Saved:', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
