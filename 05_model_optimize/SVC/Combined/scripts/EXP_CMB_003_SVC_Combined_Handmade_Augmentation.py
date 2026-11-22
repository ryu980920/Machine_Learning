#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVC Combined EXP-CMB-003: full official training, writer-heldout augmentation.

Run from repository root:
 python 05_model_optimize/SVC/Combined/scripts/EXP_CMB_003_SVC_Combined_Handmade_Augmentation.py
 python 05_model_optimize/SVC/Combined/scripts/EXP_CMB_003_SVC_Combined_Handmade_Augmentation.py --ratio 50
 python 05_model_optimize/SVC/Combined/scripts/EXP_CMB_003_SVC_Combined_Handmade_Augmentation.py --ratio 100
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
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(ROOT / '05_model_optimize' / 'SVC' / 'Combined' / 'scripts'),
    str(ROOT / '05_model_optimize' / 'SVC' / 'Normal' / 'scripts'),
    str(ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'),
    str(ROOT / '05_model_optimize' / 'KNN' / 'Combined' / 'scripts'),
    str(ROOT / '03_experiments' / 'model_optimization'),
]
from EXP_001_SVC_Normal_Baseline import WRITERS, baseline, load_writer, report
from EXP_005_SVC_Normal_HOG_SafePilot import predict_batches
from EXP_007_SVC_Normal_Full_Validation import features
from EXP_CMB_001_SVC_Combined_Hybrid_4Writer_OOF import atomic_json, write_csv
from EXP_CMB_003_KNN_Combined_Handmade_Augmentation import balanced_nested_indices, SEED, AUG_STRENGTH
from EXP007_KNN_Normal_Augmentation import size_jitter, stroke_jitter
from EXP011_KNN_Normal_Deskew import test_deskew
from EXP012_KNN_Normal_HOG import test_hog
from common import LABELS, fixed_split

CONTROL = ROOT / '05_model_optimize' / 'SVC' / 'Combined' / 'results' / 'EXP_CMB_001_SVC_Hybrid_4Writer_OOF'
OUT = ROOT / '05_model_optimize' / 'SVC' / 'Combined' / 'results' / 'EXP_CMB_003_SVC_Hybrid_Handmade_Augmentation'
EXPECTED_CONTROL = {41: 572, 42: 604, 43: 624, 44: 603}
FIELDS = ['ratio_percent', 'heldout_writer', 'train_writers', 'official_train_n', 'handmade_original_train_n',
          'handmade_augmented_train_n', 'train_n', 'handmade_correct', 'handmade_accuracy',
          'handmade_macro_f1', 'official_correct', 'official_accuracy', 'official_macro_f1',
          'fit_seconds', 'support_vectors', 'handmade_predict_seconds', 'official_predict_seconds']


def say(*items):
    print(*items, flush=True)


def read_control(handmade, sha):
    path = CONTROL / 'metrics.json'
    if not path.is_file():
        raise FileNotFoundError(f'Complete EXP-CMB-001 control is required: {path}')
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('official_dataset_sha256') != sha or record.get('combined_oof_development', {}).get('correct') != 2403:
        raise RuntimeError('SVC Combined EXP-CMB-001 control mismatch')
    preds = {}
    for w in WRITERS:
        chk = record.get('handmade_sha256', {}).get(str(w))
        if chk != handmade[w][2]:
            raise RuntimeError(f'Control Hand-made SHA mismatch for writer {w}')
        p = CONTROL / f'fold_writer_{w}_predictions.csv'
        if not p.is_file():
            raise FileNotFoundError(f'Missing control fold predictions: {p}')
        with p.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 640 or any(int(row['index']) != i for i, row in enumerate(rows)):
            raise RuntimeError(f'Invalid control row order: writer {w}')
        if [LABELS.index(row['true']) for row in rows] != handmade[w][1].tolist():
            raise RuntimeError(f'Control truth mismatch: writer {w}')
        preds[w] = np.array([LABELS.index(row['combined_pred']) for row in rows], dtype=np.int64)
        if report(handmade[w][1], preds[w])['correct'] != EXPECTED_CONTROL[w]:
            raise RuntimeError(f'Control prediction mismatch: writer {w}')
    return record, preds


def load_checkpoint(path, csvpath, args, heldout, expected_train, official_sha, handmade, control_preds):
    if not path.exists() and not csvpath.exists():
        return None
    if path.exists() != csvpath.exists():
        raise RuntimeError(f'Incomplete checkpoint: {path} / {csvpath}')
    record = json.loads(path.read_text(encoding='utf-8'))
    if (record.get('status') != 'complete' or record.get('ratio_percent') != args.ratio
            or record.get('heldout_writer') != heldout or record.get('train_n') != expected_train
            or record.get('official_dataset_sha256') != official_sha
            or record.get('handmade_sha256') != {str(w): handmade[w][2] for w in WRITERS}
            or record.get('parameters') != {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf', 'cache_size_mb': args.cache_size}
            or record.get('augmentation_strength') != AUG_STRENGTH
            or record.get('augmentation_seed') != SEED + heldout):
        raise RuntimeError(f'Incompatible checkpoint: {path}')
    with csvpath.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 640 or any(int(r['index']) != i for i, r in enumerate(rows)):
        raise RuntimeError(f'Invalid checkpoint predictions: {csvpath}')
    yt = handmade[heldout][1]
    if not np.array_equal(yt, [LABELS.index(r['true']) for r in rows]):
        raise RuntimeError('Checkpoint truth labels mismatch')
    if not np.array_equal(control_preds[heldout], [LABELS.index(r['control_pred']) for r in rows]):
        raise RuntimeError('Checkpoint control labels mismatch')
    pred = np.array([LABELS.index(r['augmented_pred']) for r in rows], dtype=np.int64)
    if report(yt, pred)['correct'] != record['handmade']['correct']:
        raise RuntimeError('Checkpoint scores mismatch')
    return record, pred


def run(args):
    if args.cache_size < 100:
        raise ValueError('--cache-size must be at least 100')
    test_deskew()
    test_hog()
    svc, scaled, source = baseline()
    if not scaled or svc.kernel != 'rbf':
        raise RuntimeError('Unexpected project baseline')
    say(f'SVC Combined EXP-CMB-003: augmentation ratio {args.ratio}%')
    say('FULL official Training=76800; writer training=1920 + augmented=', 1920*args.ratio//100)
    say('Official Test NOT loaded. Writer-heldout OOF is repeatedly used DEVELOPMENT data.')
    say('SVC source:', source)
    OUT.mkdir(parents=True, exist_ok=True)
    xtrain, ytrain, xval, yval, sha = fixed_split()
    if (len(ytrain), len(yval)) != (76800, 19200):
        raise RuntimeError('Official split sizes changed')
    handmade = {w: load_writer(w) for w in WRITERS}
    control, control_preds = read_control(handmade, sha)
    say('Control EXP-CMB-001 verified: 2403/2560, accuracy=0.938672')
    train_feat = features(xtrain, 'hybrid', 'Official Train 76800')
    del xtrain
    gc.collect()
    val_feat = features(xval, 'hybrid', 'Official Validation 19200')
    del xval
    gc.collect()
    feat_by_writer = {w: features(handmade[w][0], 'hybrid', f'Writer {w}') for w in WRITERS}
    if train_feat.shape != (76800, 1108) or val_feat.shape != (19200, 1108):
        raise RuntimeError('Unexpected feature shape')
    expected_aug = args.ratio * 1920 // 100
    expected_train = 78720 + expected_aug
    records = {}
    preds = {}
    for heldout in WRITERS:
        jsonpath = OUT / f'ratio_{args.ratio}_fold_writer_{heldout}.json'
        csvpath = OUT / f'ratio_{args.ratio}_fold_writer_{heldout}_predictions.csv'
        if not args.overwrite:
            found = load_checkpoint(jsonpath, csvpath, args, heldout, expected_train, sha, handmade, control_preds)
            if found is not None:
                records[heldout], preds[heldout] = found
                say(f'[Writer {heldout}] completed checkpoint reused')
                continue
        source_writers = [w for w in WRITERS if w != heldout]
        src_raw = np.concatenate([handmade[w][0] for w in source_writers])
        src_y = np.concatenate([handmade[w][1] for w in source_writers])
        src_writer = np.repeat(np.asarray(source_writers), 640)
        indices = balanced_nested_indices(src_writer, src_y, seed=SEED + heldout)[args.ratio]
        if len(indices) != expected_aug or np.any(src_writer[indices] == heldout):
            raise RuntimeError('Augmentation selection mismatch or heldout leak')
        say(f'\n[Fold heldout={heldout}] train writers={source_writers} augmented={len(indices)}')
        t = time.perf_counter()
        resized, size_stats = size_jitter(src_raw, indices, seed=SEED + heldout + 100)
        augmented_raw, stroke_stats = stroke_jitter(resized, strength=AUG_STRENGTH)
        del resized, src_raw
        aug_feat = features(augmented_raw, 'hybrid', f'Augmented fold {heldout}')
        del augmented_raw
        say(f'[Fold heldout={heldout}] augmentation prepared in {time.perf_counter()-t:.2f}s')
        src_feat = np.concatenate([feat_by_writer[w] for w in source_writers])
        train_x = np.concatenate([train_feat, src_feat, aug_feat])
        train_y = np.concatenate([ytrain, src_y, src_y[indices]])
        del src_feat, aug_feat, src_y
        gc.collect()
        if train_x.shape != (expected_train, 1108) or train_y.shape != (expected_train,):
            raise RuntimeError('Wrong train matrix shape')
        model = clone(svc).set_params(C=1.0, gamma='scale', cache_size=args.cache_size)
        say(f'[Fold heldout={heldout}] fit START rows={len(train_y)}')
        t = time.perf_counter()
        model.fit(train_x, train_y)
        fit_sec = round(time.perf_counter() - t, 2)
        del train_x, train_y
        gc.collect()
        say(f'[Fold heldout={heldout}] fit DONE {fit_sec}s support_vectors={len(model.support_)}')
        hp, hp_sec = predict_batches(model, feat_by_writer[heldout], f'Writer {heldout}', args.batch_size)
        hm = report(handmade[heldout][1], hp)
        say(f'[Fold heldout={heldout}] Hand-made accuracy={hm["accuracy"]:.6f} ({hm["correct"]}/640)')
        vp, vp_sec = predict_batches(model, val_feat, f'Official Val fold {heldout}', args.batch_size)
        vm = report(yval, vp)
        say(f'[Fold heldout={heldout}] Official Validation accuracy={vm["accuracy"]:.6f}')
        rec = {
            'status': 'complete', 'ratio_percent': args.ratio, 'heldout_writer': heldout,
            'train_writers': source_writers, 'official_dataset_sha256': sha,
            'handmade_sha256': {str(w): handmade[w][2] for w in WRITERS},
            'official_train_n': 76800, 'original_handmade_train_n': 1920,
            'augmented_handmade_train_n': expected_aug, 'train_n': expected_train,
            'augmentation_seed': SEED + heldout, 'augmentation_strength': AUG_STRENGTH,
            'size_stats': size_stats, 'stroke_stats': stroke_stats,
            'parameters': {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf', 'cache_size_mb': args.cache_size},
            'fit_seconds': fit_sec, 'support_vectors': int(len(model.support_)),
            'handmade_predict_seconds': hp_sec, 'official_predict_seconds': vp_sec,
            'handmade': hm, 'official_validation': vm,
        }
        pred_rows = [
            {'writer': heldout, 'index': i, 'true': LABELS[int(handmade[heldout][1][i])],
             'control_pred': LABELS[int(control_preds[heldout][i])],
             'augmented_pred': LABELS[int(hp[i])]} for i in range(640)
        ]
        write_csv(csvpath, pred_rows, ['writer', 'index', 'true', 'control_pred', 'augmented_pred'])
        atomic_json(jsonpath, rec)
        records[heldout] = rec
        preds[heldout] = hp
        del model, vp, hp
        gc.collect()
    yt = np.concatenate([handmade[w][1] for w in WRITERS])
    cp = np.concatenate([control_preds[w] for w in WRITERS])
    ap = np.concatenate([preds[w] for w in WRITERS])
    cm = confusion_matrix(yt, ap, labels=np.arange(16))
    write_csv(OUT / f'ratio_{args.ratio}_confusion_matrix.csv', [
        {'true': LABELS[i], **{LABELS[j]: int(cm[i,j]) for j in range(16)}} for i in range(16)
    ], ['true', *LABELS])
    paired = {
        'corrected': int(np.sum((cp != yt) & (ap == yt))),
        'new_errors': int(np.sum((cp == yt) & (ap != yt))),
        'unchanged_correct': int(np.sum((cp == yt) & (ap == yt))),
        'unchanged_wrong': int(np.sum((cp != yt) & (ap != yt))),
    }
    control_result = report(yt, cp)
    aug_result = report(yt, ap)
    fold_rows = []
    for w in WRITERS:
        rec = records[w]
        fold_rows.append({
            'ratio_percent': args.ratio, 'heldout_writer': w,
            'train_writers': ','.join(map(str, rec['train_writers'])),
            'official_train_n': 76800, 'handmade_original_train_n': 1920,
            'handmade_augmented_train_n': expected_aug, 'train_n': expected_train,
            'handmade_correct': rec['handmade']['correct'], 'handmade_accuracy': rec['handmade']['accuracy'],
            'handmade_macro_f1': rec['handmade']['macro_f1'],
            'official_correct': rec['official_validation']['correct'],
            'official_accuracy': rec['official_validation']['accuracy'],
            'official_macro_f1': rec['official_validation']['macro_f1'],
            'fit_seconds': rec['fit_seconds'], 'support_vectors': rec['support_vectors'],
            'handmade_predict_seconds': rec['handmade_predict_seconds'],
            'official_predict_seconds': rec['official_predict_seconds'],
        })
    write_csv(OUT / f'ratio_{args.ratio}_fold_results.csv', fold_rows, FIELDS)
    write_csv(OUT / f'ratio_{args.ratio}_oof_predictions.csv', [
        {'writer': w, 'index': i, 'true': LABELS[int(handmade[w][1][i])],
         'control_pred': LABELS[int(control_preds[w][i])], 'augmented_pred': LABELS[int(preds[w][i])]}
        for w in WRITERS for i in range(640)
    ], ['writer', 'index', 'true', 'control_pred', 'augmented_pred'])
    outcome = {
        'experiment': 'EXP-CMB-003 SVC Combined Hybrid train-only Hand-made augmentation',
        'ratio_percent': args.ratio, 'full_official_train_n': 76800, 'official_validation_n': 19200,
        'original_handmade_train_per_fold': 1920, 'augmented_handmade_train_per_fold': expected_aug,
        'train_per_fold': expected_train, 'official_dataset_sha256': sha,
        'feature': 'CoM -> Deskew -> hybrid pixel20/HOG80',
        'control_reference': str(CONTROL.relative_to(ROOT)),
        'control_oof': control_result, 'augmented_oof': aug_result,
        'change_percentage_points': 100 * (aug_result['accuracy'] - control_result['accuracy']),
        'paired': paired, 'folds': [records[w] for w in WRITERS],
        'official_test_used': False,
        'note': 'Hand-made OOF is repeatedly-used development data, NOT independent final Test.',
    }
    atomic_json(OUT / f'ratio_{args.ratio}_metrics.json', outcome)
    say(f'\nSVC COMBINED EXP-CMB-003 SUMMARY ratio={args.ratio}%')
    say(f'Control={control_result["accuracy"]:.6f} ({control_result["correct"]}/2560) F1={control_result["macro_f1"]:.6f}')
    say(f'Augmented={aug_result["accuracy"]:.6f} ({aug_result["correct"]}/2560) F1={aug_result["macro_f1"]:.6f}')
    say(f'Delta={outcome["change_percentage_points"]:+.4f}pp  Paired={paired}')
    say('Saved:', OUT)
    say('Official Test NOT loaded.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=int, choices=[25, 50, 100], default=25)
    parser.add_argument('--batch-size', type=int, choices=[64, 128, 256, 512], default=256)
    parser.add_argument('--cache-size', type=int, default=200)
    parser.add_argument('--overwrite', action='store_true')
    run(parser.parse_args())
