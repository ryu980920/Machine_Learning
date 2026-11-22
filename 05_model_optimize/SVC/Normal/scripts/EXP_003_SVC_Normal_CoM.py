#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVC Normal EXP-003: unchanged baseline versus CoM-centered SVC.

Run from the repository root:
  python 05_model_optimize/SVC/Normal/scripts/EXP_003_SVC_Normal_CoM.py

Uses official Training/Validation and Hand-made development data; never loads Test.
The original SVC and pixel scaler come from the team's baseline code.
The CoM transformer is reused from KNN Normal EXP004 (same MLP implementation).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[4]
SVC_SCRIPTS = ROOT / '05_model_optimize' / 'SVC' / 'Normal' / 'scripts'
KNN_SCRIPTS = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'
sys.path.insert(0, str(SVC_SCRIPTS))
sys.path.insert(0, str(KNN_SCRIPTS))
sys.path.insert(0, str(ROOT / '03_experiments' / 'model_optimization'))

from EXP_001_SVC_Normal_Baseline import WRITERS, OUT_BASE, baseline, load_writer, report
from EXP004_KNN_Normal_CoM import BatchedCoM
from common import LABELS, fixed_split, pipeline

BASELINE_DIR = OUT_BASE / 'EXP_001_SVC_Normal_Baseline'
OUT = OUT_BASE / 'EXP_003_SVC_Normal_CoM'


def save_csv(filename, rows, fieldnames):
    with (OUT / filename).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    baseline_metrics_path = BASELINE_DIR / 'metrics.json'
    baseline_predictions_path = BASELINE_DIR / 'handmade_predictions.csv'
    if not baseline_metrics_path.exists() or not baseline_predictions_path.exists():
        raise FileNotFoundError('SVC EXP-001 result missing. Run EXP_001_SVC_Normal_Baseline.py first.')
    prior = json.loads(baseline_metrics_path.read_text(encoding='utf-8'))
    with baseline_predictions_path.open('r', encoding='utf-8-sig', newline='') as f:
        prior_rows = list(csv.DictReader(f))
    if len(prior_rows) != 2560 or prior['handmade_development']['correct'] != 1576:
        raise RuntimeError('Unexpected EXP-001 predictions or baseline performance')
    if not np.isclose(prior['official_validation']['accuracy'], 0.9766666666666667, atol=1e-10):
        raise RuntimeError('EXP-001 official validation baseline did not match reported results')

    raw_svc, scaled, source = baseline()
    if not scaled:
        raise RuntimeError('The recorded EXP-001 baseline expects scaling=True')
    if prior.get('model_source') != source:
        raise RuntimeError('SVC baseline source differs from EXP-001')
    if prior.get('hyperparameters') != raw_svc.get_params():
        raise RuntimeError('SVC hyperparameters differ from EXP-001')
    x_train, y_train, x_val, y_val, official_sha = fixed_split()
    if len(y_train) != 76800 or len(y_val) != 19200:
        raise RuntimeError('Official split mismatch')
    if prior.get('dataset_sha256') != official_sha:
        raise RuntimeError('Official dataset digest differs from EXP-001')
    handmade = {writer: load_writer(writer) for writer in WRITERS}
    for writer in WRITERS:
        if prior['handmade_sha256'][str(writer)] != handmade[writer][2]:
            raise RuntimeError(f'Writer {writer} digest differs from EXP-001')

    base_pipe = pipeline(raw_svc, True)
    candidate = Pipeline([('center_of_mass', BatchedCoM()), *base_pipe.steps])
    print('EXP-003 SVC Normal: CoM before existing PixelScaler255 and SVC', flush=True)
    print('Official Training=76800, Validation=19200; Test NOT loaded', flush=True)
    print('Using saved EXP-001 predictions as unchanged baseline (no baseline refit)', flush=True)
    t0 = time.perf_counter()
    candidate.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - t0
    print(f'CoM SVC fit_seconds={fit_seconds:.2f}', flush=True)
    prediction_val = candidate.predict(x_val).astype(np.int64)
    val_scores = report(y_val, prediction_val)
    print(f"Official Validation: EXP-001={prior['official_validation']['accuracy']:.6f} CoM={val_scores['accuracy']:.6f}", flush=True)

    writer_results, pred_rows, true_all, before_all, after_all = [], [], [], [], []
    for writer in WRITERS:
        x, y, _ = handmade[writer]
        prediction = candidate.predict(x).astype(np.int64)
        old_rows = [r for r in prior_rows if int(r['writer']) == writer]
        old_rows = sorted(old_rows, key=lambda r: int(r['index']))
        if len(old_rows) != 640 or [int(r['index']) for r in old_rows] != list(range(640)):
            raise RuntimeError(f'Invalid EXP-001 saved predictions for writer {writer}')
        old_true = np.array([LABELS.index(r['true']) for r in old_rows], dtype=np.int64)
        old_pred = np.array([LABELS.index(r['prediction']) for r in old_rows], dtype=np.int64)
        if not np.array_equal(y, old_true):
            raise RuntimeError(f'Original labels have changed for writer {writer}')
        m = report(y, prediction)
        old_m = report(y, old_pred)
        writer_results.append({'writer': writer, 'n': len(y), 'baseline_accuracy': old_m['accuracy'],
                               'com_accuracy': m['accuracy'], 'baseline_macro_f1': old_m['macro_f1'],
                               'com_macro_f1': m['macro_f1'],
                               'delta_accuracy_pp': 100 * (m['accuracy'] - old_m['accuracy'])})
        print(f"Writer {writer}: Baseline={old_m['accuracy']:.6f}, CoM={m['accuracy']:.6f}", flush=True)
        true_all.append(y)
        before_all.append(old_pred)
        after_all.append(prediction)
        pred_rows.extend({'writer': writer, 'index': i, 'true': LABELS[int(yt)],
                          'baseline_prediction': LABELS[int(pb)], 'com_prediction': LABELS[int(pc)],
                          'baseline_correct': int(yt == pb), 'com_correct': int(yt == pc)}
                         for i, (yt, pb, pc) in enumerate(zip(y, old_pred, prediction)))

    yt = np.concatenate(true_all)
    pb = np.concatenate(before_all)
    pc = np.concatenate(after_all)
    baseline_hm = report(yt, pb)
    com_hm = report(yt, pc)
    if baseline_hm['correct'] != 1576 or not np.isclose(baseline_hm['macro_f1'], 0.604038, atol=0.000001):
        raise RuntimeError('EXP-001 saved Hand-made predictions are inconsistent')
    paired = {'corrected': int(np.sum((pb != yt) & (pc == yt))),
              'new_errors': int(np.sum((pb == yt) & (pc != yt))),
              'unchanged_correct': int(np.sum((pb == yt) & (pc == yt))),
              'unchanged_wrong': int(np.sum((pb != yt) & (pc != yt)))}
    OUT.mkdir(parents=True, exist_ok=True)
    save_csv('writer_results.csv', writer_results, list(writer_results[0]))
    save_csv('handmade_predictions.csv', pred_rows, list(pred_rows[0]))
    cm = confusion_matrix(yt, pc, labels=np.arange(16))
    save_csv('handmade_com_confusion_matrix.csv',
             [{'true/pred': LABELS[i], **{LABELS[j]: int(cm[i, j]) for j in range(16)}} for i in range(16)],
             ['true/pred', *LABELS])
    p, r, f, n = precision_recall_fscore_support(yt, pc, labels=np.arange(16), zero_division=0)
    save_csv('handmade_com_per_class.csv',
             [{'class': LABELS[i], 'precision': float(p[i]), 'recall': float(r[i]),
               'f1': float(f[i]), 'support': int(n[i])} for i in range(16)],
             ['class', 'precision', 'recall', 'f1', 'support'])
    summary = {'experiment': 'EXP-003 SVC Normal CoM', 'model_source': source,
               'hyperparameters': raw_svc.get_params(), 'scaling': scaled,
               'com': 'Same CoM implementation as KNN Normal EXP004 / MLP Automatic_Centering_Transformer',
               'official_validation': {'baseline': prior['official_validation'], 'com': val_scores,
                                       'delta_accuracy_pp': 100 * (val_scores['accuracy']-prior['official_validation']['accuracy'])},
               'handmade_development': {'baseline': baseline_hm, 'com': com_hm,
                                        'delta_accuracy_pp': 100 * (com_hm['accuracy']-baseline_hm['accuracy'])},
               'writers': writer_results, 'paired': paired, 'fit_seconds': fit_seconds,
               'official_train_n': len(y_train), 'official_val_n': len(y_val),
               'official_test_used': False,
               'evaluation_role': 'repeated-use development Hand-made; not independent test',
               'dataset_sha256': official_sha}
    (OUT / 'metrics.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"Hand-made: Baseline={baseline_hm['accuracy']:.6f}, CoM={com_hm['accuracy']:.6f} F1={com_hm['macro_f1']:.6f}", flush=True)
    print(f"Difference: Official={summary['official_validation']['delta_accuracy_pp']:+.4f}pp Hand-made={summary['handmade_development']['delta_accuracy_pp']:+.4f}pp", flush=True)
    print('Paired:', paired, flush=True)
    print('Saved:', OUT, flush=True)


if __name__ == '__main__':
    main()
