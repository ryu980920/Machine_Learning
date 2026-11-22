#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVC Combined EXP-CMB-001: full official training + 3 writers, 4-writer OOF.

From repository root:
 python 05_model_optimize/SVC/Combined/scripts/EXP_CMB_001_SVC_Combined_Hybrid_4Writer_OOF.py

Uses the *saved* EXP-007 Hybrid Normal predictions as comparison. Does not
retrain Normal or access the official Test. Each completed fold is checkpointed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(ROOT / '05_model_optimize' / 'SVC' / 'Normal' / 'scripts'),
    str(ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'),
    str(ROOT / '03_experiments' / 'model_optimization'),
]
from EXP_001_SVC_Normal_Baseline import WRITERS, baseline, load_writer, report
from EXP_005_SVC_Normal_HOG_SafePilot import predict_batches
from EXP_007_SVC_Normal_Full_Validation import features
from EXP011_KNN_Normal_Deskew import test_deskew
from EXP012_KNN_Normal_HOG import test_hog
from common import LABELS, fixed_split

NORMAL_DIR = (ROOT / '05_model_optimize' / 'SVC' / 'Normal' / 'results'
              / 'EXP_007_SVC_Normal_FULL_76800')
OUT = (ROOT / '05_model_optimize' / 'SVC' / 'Combined' / 'results'
       / 'EXP_CMB_001_SVC_Hybrid_4Writer_OOF')


def say(*items):
    print(*items, flush=True)


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def write_csv(path, rows, fields):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    os.replace(tmp, path)


def read_normal_predictions(handmade):
    metrics_file = NORMAL_DIR / 'hybrid_metrics.json'
    pred_file = NORMAL_DIR / 'hybrid_handmade_predictions.csv'
    if not metrics_file.is_file() or not pred_file.is_file():
        raise FileNotFoundError('EXP-007 Hybrid Normal metrics/predictions not found: ' + str(NORMAL_DIR))
    saved = json.loads(metrics_file.read_text(encoding='utf-8'))
    if (saved.get('status') != 'complete' or saved.get('feature') != 'hybrid'
            or saved.get('official_training') != 76800
            or saved.get('handmade_development', {}).get('correct') != 2345):
        raise RuntimeError('EXP-007 Hybrid Normal reference differs from established result')
    entries = {}
    with pred_file.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key = (int(row['writer']), int(row['index']))
            if key in entries:
                raise RuntimeError(f'Duplicate Normal prediction: {key}')
            true = row['true']
            pred = row.get('predicted', row.get('prediction'))
            if true not in LABELS or pred not in LABELS:
                raise RuntimeError(f'Unexpected saved class: {row}')
            entries[key] = (LABELS.index(true), LABELS.index(pred))
    expected = len(WRITERS) * 640
    if len(entries) != expected:
        raise RuntimeError(f'Expected {expected} Normal predictions, got {len(entries)}')
    normal_preds = {}
    for w in WRITERS:
        actual = handmade[w][1]
        rows = [entries[(w, i)] for i in range(640)]
        if not np.array_equal(actual, np.array([x[0] for x in rows])):
            raise RuntimeError(f'Normal predictions labels mismatch for writer {w}')
        normal_preds[w] = np.asarray([x[1] for x in rows], dtype=np.int64)
    overall = report(np.concatenate([handmade[w][1] for w in WRITERS]),
                     np.concatenate([normal_preds[w] for w in WRITERS]))
    if overall['correct'] != 2345:
        raise RuntimeError('Normal predictions did not reproduce 2345/2560')
    return saved, normal_preds


def confusion_csv(path, yt, yp):
    cm = confusion_matrix(yt, yp, labels=np.arange(16))
    write_csv(path, [{'true': LABELS[i], **{LABELS[j]: int(cm[i, j]) for j in range(16)}}
                     for i in range(16)], ['true', *LABELS])


def run(args):
    test_deskew()
    test_hog()
    svc, scaled, source = baseline()
    if not scaled or svc.kernel != 'rbf':
        raise RuntimeError('Unexpected project SVC baseline (expected RBF and scaled=True)')
    if args.cache_size < 100:
        raise ValueError('cache-size should be at least 100 MB')
    OUT.mkdir(parents=True, exist_ok=True)
    say('SVC Combined EXP-CMB-001 / full 76,800 official + 1,920 writer training per fold')
    say('Feature=CoM -> Deskew -> Pixel:HOG 20:80; SVC C=1, gamma=scale')
    say('Official Test NOT loaded. Hand-made OOF is repeatedly-used DEVELOPMENT data.')
    say('SVC baseline source:', source)
    start = time.perf_counter()
    xtrain, ytrain, xval, yval, data_sha = fixed_split()
    if (len(ytrain), len(yval)) != (76800, 19200):
        raise RuntimeError('Official train/validation split differs from 76800/19200')
    handmade = {w: load_writer(w) for w in WRITERS}
    normal_metrics, normal_preds = read_normal_predictions(handmade)
    if normal_metrics.get('dataset_sha256') != data_sha:
        raise RuntimeError('Different official dataset from Normal EXP-007')
    say(f'Data and saved Normal reference loaded in {time.perf_counter()-start:.2f}s')
    say('Frozen Normal reference: official=%.6f handmade=%.6f' % (
        normal_metrics['official_validation']['accuracy'],
        normal_metrics['handmade_development']['accuracy']))
    train_feat = features(xtrain, 'hybrid', 'Official Train 76800')
    del xtrain
    gc.collect()
    val_feat = features(xval, 'hybrid', 'Official Validation 19200')
    features_by_writer = {w: features(handmade[w][0], 'hybrid', f'Writer {w}') for w in WRITERS}
    assert train_feat.shape == (76800, 1108) and val_feat.shape == (19200, 1108)
    for w in WRITERS:
        assert features_by_writer[w].shape == (640, 1108)
    say('Preprocessing completed; Fold training begins.')
    fold_rows = []
    combined_preds = {}
    for heldout in WRITERS:
        fold_json = OUT / f'fold_writer_{heldout}.json'
        fold_csv = OUT / f'fold_writer_{heldout}_predictions.csv'
        if fold_json.exists() and fold_csv.exists() and not args.overwrite:
            record = json.loads(fold_json.read_text(encoding='utf-8'))
            if (record.get('status') != 'complete' or record.get('heldout_writer') != heldout
                    or record.get('official_dataset_sha256') != data_sha
                    or record.get('train_n') != 78720):
                raise RuntimeError(f'Existing fold checkpoint is incompatible: {fold_json}')
            with fold_csv.open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            if len(rows) != 640 or any(int(row['index']) != i for i, row in enumerate(rows)):
                raise RuntimeError(f'Invalid fold prediction checkpoint: {fold_csv}')
            pred = np.array([LABELS.index(row['combined_pred']) for row in rows], dtype=np.int64)
            truth = handmade[heldout][1]
            if not np.array_equal(truth, [LABELS.index(row['true']) for row in rows]):
                raise RuntimeError('Checkpoint truth mismatch')
            if report(truth, pred)['correct'] != record['combined_handmade']['correct']:
                raise RuntimeError('Checkpoint prediction metric mismatch')
            combined_preds[heldout] = pred
            fold_rows.append(record)
            say(f'[Writer {heldout}] Existing completed checkpoint reused; no refit.')
            continue
        if (fold_json.exists() or fold_csv.exists()) and not args.overwrite:
            raise RuntimeError(f'Partial/conflicting checkpoint for writer {heldout}; inspect: {OUT}')
        training_writers = [w for w in WRITERS if w != heldout]
        extra_x = np.concatenate([features_by_writer[w] for w in training_writers], axis=0)
        extra_y = np.concatenate([handmade[w][1] for w in training_writers], axis=0)
        full_x = np.concatenate([train_feat, extra_x], axis=0)
        full_y = np.concatenate([ytrain, extra_y], axis=0)
        if full_x.shape != (78720, 1108) or len(extra_y) != 1920:
            raise RuntimeError('Combined train shape mismatch')
        del extra_x, extra_y
        gc.collect()
        model = clone(svc).set_params(C=1.0, gamma='scale', cache_size=args.cache_size)
        say(f'\n[Fold heldout={heldout}] Training writers={training_writers}; rows=78720')
        say(f'[Fold heldout={heldout}] fit START')
        start = time.perf_counter()
        model.fit(full_x, full_y)
        fit_seconds = round(time.perf_counter()-start, 2)
        say(f'[Fold heldout={heldout}] fit DONE {fit_seconds:.2f}s; support vectors={len(model.support_)}')
        del full_x, full_y
        gc.collect()
        hp, htime = predict_batches(model, features_by_writer[heldout],
                                     f'Writer {heldout}', args.batch_size)
        hmetric = report(handmade[heldout][1], hp)
        say(f'[Fold heldout={heldout}] Hand-made={hmetric["accuracy"]:.6f} ({hmetric["correct"]}/640)')
        vp, vtime = predict_batches(model, val_feat,
                                     f'Official Val fold {heldout}', args.batch_size)
        vmetric = report(yval, vp)
        say(f'[Fold heldout={heldout}] Official Validation={vmetric["accuracy"]:.6f}')
        record = {
            'status': 'complete', 'heldout_writer': heldout,
            'train_writers': training_writers, 'official_dataset_sha256': data_sha,
            'official_train_n': 76800, 'handmade_train_n': 1920, 'train_n': 78720,
            'heldout_n': 640, 'fit_seconds': fit_seconds,
            'handmade_predict_seconds': htime, 'official_predict_seconds': vtime,
            'support_vectors': int(len(model.support_)),
            'normal_handmade': report(handmade[heldout][1], normal_preds[heldout]),
            'combined_handmade': hmetric, 'combined_official_validation': vmetric,
        }
        pred_rows = [{
            'writer': heldout, 'index': i, 'true': LABELS[int(handmade[heldout][1][i])],
            'normal_pred': LABELS[int(normal_preds[heldout][i])],
            'combined_pred': LABELS[int(hp[i])],
            'normal_correct': int(normal_preds[heldout][i] == handmade[heldout][1][i]),
            'combined_correct': int(hp[i] == handmade[heldout][1][i]),
        } for i in range(640)]
        write_csv(fold_csv, pred_rows,
                  ['writer','index','true','normal_pred','combined_pred','normal_correct','combined_correct'])
        atomic_json(fold_json, record)
        combined_preds[heldout] = hp
        fold_rows.append(record)
        del model, vp, hp
        gc.collect()
    yt = np.concatenate([handmade[w][1] for w in WRITERS])
    normal = np.concatenate([normal_preds[w] for w in WRITERS])
    combined = np.concatenate([combined_preds[w] for w in WRITERS])
    nr = report(yt, normal)
    cr = report(yt, combined)
    if nr['correct'] != 2345:
        raise RuntimeError('Normal reference no longer matches established full-training result')
    paired = {
        'corrected': int(np.count_nonzero((normal != yt) & (combined == yt))),
        'new_errors': int(np.count_nonzero((normal == yt) & (combined != yt))),
        'unchanged_correct': int(np.count_nonzero((normal == yt) & (combined == yt))),
        'unchanged_wrong': int(np.count_nonzero((normal != yt) & (combined != yt))),
    }
    all_predictions = []
    for heldout in WRITERS:
        with (OUT / f'fold_writer_{heldout}_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
            all_predictions.extend(list(csv.DictReader(f)))
    write_csv(OUT / 'oof_predictions.csv', all_predictions,
              ['writer','index','true','normal_pred','combined_pred','normal_correct','combined_correct'])
    write_csv(OUT / 'fold_results.csv', [
        {
            'heldout_writer': r['heldout_writer'],
            'train_writers': ','.join(str(w) for w in r['train_writers']),
            'train_n': r['train_n'], 'fit_seconds': r['fit_seconds'],
            'normal_accuracy': r['normal_handmade']['accuracy'],
            'combined_accuracy': r['combined_handmade']['accuracy'],
            'combined_macro_f1': r['combined_handmade']['macro_f1'],
            'combined_correct': r['combined_handmade']['correct'],
            'official_accuracy': r['combined_official_validation']['accuracy'],
            'official_macro_f1': r['combined_official_validation']['macro_f1'],
        } for r in fold_rows],
        ['heldout_writer','train_writers','train_n','fit_seconds','normal_accuracy',
         'combined_accuracy','combined_macro_f1','combined_correct','official_accuracy','official_macro_f1'])
    confusion_csv(OUT / 'normal_oof_confusion_matrix.csv', yt, normal)
    confusion_csv(OUT / 'combined_oof_confusion_matrix.csv', yt, combined)
    p0, r0, f0, n0 = precision_recall_fscore_support(yt, normal, labels=np.arange(16), zero_division=0)
    p1, r1, f1, n1 = precision_recall_fscore_support(yt, combined, labels=np.arange(16), zero_division=0)
    write_csv(OUT / 'per_class_oof.csv', [
        {'class': LABELS[i], 'support': int(n0[i]),
         'normal_recall': float(r0[i]), 'normal_f1': float(f0[i]),
         'combined_recall': float(r1[i]), 'combined_f1': float(f1[i])}
        for i in range(16)],
        ['class','support','normal_recall','normal_f1','combined_recall','combined_f1'])
    outcome = {
        'experiment': 'EXP-CMB-001 SVC Combined Hybrid full 4-writer OOF',
        'official_train_n': 76800, 'official_validation_n': 19200,
        'combined_training_per_fold_n': 78720, 'handmade_training_per_fold_n': 1920,
        'heldout_per_fold_n': 640,
        'features': 'CoM -> Deskew -> hybrid pixel20/HOG80 (1108 dimensions)',
        'parameters': {'kernel':'rbf','C':1.0,'gamma':'scale','cache_size_mb':args.cache_size},
        'official_dataset_sha256': data_sha,
        'handmade_sha256': {str(w): handmade[w][2] for w in WRITERS},
        'official_test_used': False,
        'normal_reference_official_validation': normal_metrics['official_validation'],
        'normal_oof_development': nr, 'combined_oof_development': cr,
        'paired': paired, 'folds': fold_rows,
        'evaluation_note': 'Hand-made was used in prior Normal optimization; these writer-heldout '
                           'folds exclude heldout training but are NOT an independent final test.',
    }
    atomic_json(OUT / 'metrics.json', outcome)
    say('\nSVC COMBINED EXP-CMB-001 SUMMARY')
    say(f'Normal   Hand-made={nr["accuracy"]:.6f}, MacroF1={nr["macro_f1"]:.6f} ({nr["correct"]}/2560)')
    say(f'Combined Hand-made={cr["accuracy"]:.6f}, MacroF1={cr["macro_f1"]:.6f} ({cr["correct"]}/2560)')
    say(f'Change={100*(cr["accuracy"]-nr["accuracy"]):+.4f} percentage points')
    say('Paired:', paired)
    say('Saved:', OUT)
    say('Official Test was NOT loaded.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256, choices=[64,128,256,512])
    parser.add_argument('--cache-size', type=int, default=200)
    parser.add_argument('--overwrite', action='store_true', help='Intentionally rerun and replace fold results')
    run(parser.parse_args())
