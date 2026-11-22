#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVC Combined EXP-CMB-005: 100% Hand-made augmentation plus targeted D/B augmentation.

Run from the repository root. Reuses saved EXP-CMB-003 ratio-100 predictions as control.
Official Training 76,800 in all folds, official Test never loaded.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
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

AUG100 = ROOT / '05_model_optimize' / 'SVC' / 'Combined' / 'results' / 'EXP_CMB_003_SVC_Hybrid_Handmade_Augmentation'
OUT = ROOT / '05_model_optimize' / 'SVC' / 'Combined' / 'results' / 'EXP_CMB_005_SVC_Targeted_DB_Augmentation'
EXPECTED_REFERENCE = {41: 580, 42: 610, 43: 629, 44: 611}
TARGET_CLASSES = (LABELS.index('D'), LABELS.index('B'))
TARGET_SEED = 20261127
FIELDS = ['extra_per_class_writer', 'heldout_writer', 'official_train_n', 'handmade_original_train_n',
          'handmade_base_augmented_n', 'handmade_targeted_augmented_n', 'train_n',
          'handmade_correct', 'handmade_accuracy', 'handmade_macro_f1', 'official_correct',
          'official_accuracy', 'official_macro_f1', 'fit_seconds', 'support_vectors']


def say(*args):
    print(*args, flush=True)


def read_reference(handmade, official_sha):
    path = AUG100 / 'ratio_100_metrics.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing completed EXP-CMB-003 100% control: {path}')
    result = json.loads(path.read_text(encoding='utf-8'))
    if (result.get('official_dataset_sha256') != official_sha or
            result.get('augmented_oof', {}).get('correct') != 2430 or
            result.get('ratio_percent') != 100):
        raise RuntimeError('EXP-CMB-003 100% control metrics mismatch')
    predictions = {}
    for w in WRITERS:
        p = AUG100 / f'ratio_100_fold_writer_{w}_predictions.csv'
        if not p.is_file():
            raise FileNotFoundError(f'Missing 100% control predictions: {p}')
        with p.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 640 or any(int(r['index']) != i for i, r in enumerate(rows)):
            raise RuntimeError(f'Control row order mismatch: writer {w}')
        truth = np.array([LABELS.index(r['true']) for r in rows], dtype=np.int64)
        pred = np.array([LABELS.index(r['augmented_pred']) for r in rows], dtype=np.int64)
        if not np.array_equal(truth, handmade[w][1]) or report(truth, pred)['correct'] != EXPECTED_REFERENCE[w]:
            raise RuntimeError(f'Control truth/predictions mismatch: writer {w}')
        predictions[w] = pred
    if sum(report(handmade[w][1], predictions[w])['correct'] for w in WRITERS) != 2430:
        raise RuntimeError('100% control overall incorrect')
    return predictions


def targeted_indices(writer_ids, labels, per_group, seed):
    rng = np.random.default_rng(seed)
    groups = []
    for w in sorted(set(int(w) for w in writer_ids)):
        for cls in TARGET_CLASSES:
            candidates = np.flatnonzero((writer_ids == w) & (labels == cls))
            if len(candidates) != 40:
                raise RuntimeError(f'Expected 40 source examples for writer={w} class={LABELS[cls]}')
            groups.append(rng.permutation(candidates)[:per_group])
    selected = np.sort(np.concatenate(groups))
    if len(selected) != 3 * 2 * per_group or not np.isin(labels[selected], TARGET_CLASSES).all():
        raise RuntimeError('Targeted augmentation selection invalid')
    return selected


def checkpoint(jsonpath, csvpath, args, w, sha, handmade, control_preds, expected_n):
    if not jsonpath.exists() and not csvpath.exists():
        return None
    if jsonpath.exists() != csvpath.exists():
        raise RuntimeError(f'Incomplete checkpoint: {jsonpath}, {csvpath}')
    rec = json.loads(jsonpath.read_text(encoding='utf-8'))
    expected_hashes = {str(k): handmade[k][2] for k in WRITERS}
    if (rec.get('status') != 'complete' or rec.get('heldout_writer') != w or
            rec.get('extra_per_class_writer') != args.extra_per_class_writer or
            rec.get('train_n') != expected_n or rec.get('official_dataset_sha256') != sha or
            rec.get('handmade_sha256') != expected_hashes or
            rec.get('target_seed') != TARGET_SEED or
            rec.get('augmentation_seed') != SEED + w or
            rec.get('augmentation_strength') != AUG_STRENGTH or
            rec.get('parameters') != {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf',
                                       'cache_size_mb': args.cache_size}):
        raise RuntimeError(f'Incompatible checkpoint: {jsonpath}')
    with csvpath.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 640 or any(int(r['index']) != i for i, r in enumerate(rows)):
        raise RuntimeError('Invalid checkpoint row order')
    true = np.array([LABELS.index(r['true']) for r in rows], dtype=np.int64)
    control = np.array([LABELS.index(r['control_pred']) for r in rows], dtype=np.int64)
    pred = np.array([LABELS.index(r['targeted_pred']) for r in rows], dtype=np.int64)
    if (not np.array_equal(true, handmade[w][1]) or
            not np.array_equal(control, control_preds[w]) or
            report(true, pred)['correct'] != rec['handmade']['correct']):
        raise RuntimeError('Checkpoint prediction mismatch')
    return rec, pred


def run(args):
    if args.cache_size < 100:
        raise ValueError('--cache-size must be >= 100')
    test_deskew()
    test_hog()
    svc, scaled, source = baseline()
    if not scaled or svc.kernel != 'rbf':
        raise RuntimeError('Unexpected repository SVC baseline')
    n_target = 6 * args.extra_per_class_writer
    n_total = 76800 + 1920 + 1920 + n_target
    prefix = f'extra_{args.extra_per_class_writer}'
    say('EXP-CMB-005: FULL official Training=76800; 4-writer heldout; 100% base augmentation')
    say(f'Targeted D/B: {args.extra_per_class_writer} per writer/class, +{n_target} per fold; total rows={n_total}')
    say('SVC C=1, gamma=scale; CoM -> Deskew -> Pixel:HOG 20:80')
    say('Official Test NOT loaded; Hand-made is repeatedly-used DEVELOPMENT data.')
    say('SVC source:', source)
    xtr, ytr, xv, yv, sha = fixed_split()
    if (len(ytr), len(yv)) != (76800, 19200):
        raise RuntimeError('Official data split changed')
    handmade = {w: load_writer(w) for w in WRITERS}
    control_preds = read_reference(handmade, sha)
    say('100% augmented control verified: 2430/2560 (94.9219%)')
    OUT.mkdir(parents=True, exist_ok=True)
    official_feat = features(xtr, 'hybrid', 'Official Train 76800')
    del xtr
    gc.collect()
    validation_feat = features(xv, 'hybrid', 'Official Validation 19200')
    del xv
    gc.collect()
    writer_feat = {w: features(handmade[w][0], 'hybrid', f'Writer {w}') for w in WRITERS}
    if official_feat.shape != (76800, 1108) or validation_feat.shape != (19200, 1108):
        raise RuntimeError('Unexpected feature shapes')
    records, predictions = {}, {}
    for heldout in WRITERS:
        jsonpath = OUT / f'{prefix}_fold_writer_{heldout}.json'
        csvpath = OUT / f'{prefix}_fold_writer_{heldout}_predictions.csv'
        if not args.overwrite:
            saved = checkpoint(jsonpath, csvpath, args, heldout, sha, handmade,
                               control_preds, n_total)
            if saved is not None:
                records[heldout], predictions[heldout] = saved
                say(f'[Fold heldout={heldout}] complete checkpoint reused')
                continue
        train_writers = [w for w in WRITERS if w != heldout]
        raw = np.concatenate([handmade[w][0] for w in train_writers])
        source_y = np.concatenate([handmade[w][1] for w in train_writers])
        writer_ids = np.repeat(np.array(train_writers, dtype=np.int64), 640)
        if set(writer_ids) != set(train_writers) or heldout in writer_ids:
            raise RuntimeError('Heldout writer leakage')
        all_base_indices = balanced_nested_indices(writer_ids, source_y, seed=SEED+heldout)[100]
        if len(all_base_indices) != 1920:
            raise RuntimeError('Base augmentation index mismatch')
        targets = targeted_indices(writer_ids, source_y,
                                   args.extra_per_class_writer, TARGET_SEED+heldout)
        say(f'\n[Fold heldout={heldout}] writers={train_writers}; base aug=1920; D/B extra={len(targets)}')
        t = time.perf_counter()
        resized_base, base_size_stats = size_jitter(raw, all_base_indices,
                                                    seed=SEED+heldout+100)
        base_raw, base_stroke_stats = stroke_jitter(resized_base, strength=AUG_STRENGTH)
        del resized_base
        base_feat = features(base_raw, 'hybrid', f'Base augmentation fold {heldout}')
        del base_raw
        resized_target, target_size_stats = size_jitter(raw, targets,
                                                        seed=TARGET_SEED+heldout+1000)
        target_raw, target_stroke_stats = stroke_jitter(resized_target, strength=AUG_STRENGTH)
        del resized_target, raw
        target_feat = features(target_raw, 'hybrid', f'D/B augmentation fold {heldout}')
        del target_raw
        say(f'[Fold heldout={heldout}] augmentation prepared {time.perf_counter()-t:.2f}s')
        original_feat = np.concatenate([writer_feat[w] for w in train_writers])
        fit_x = np.concatenate([official_feat, original_feat, base_feat, target_feat])
        fit_y = np.concatenate([ytr, source_y, source_y[all_base_indices], source_y[targets]])
        del original_feat, base_feat, target_feat, source_y
        gc.collect()
        if fit_x.shape != (n_total, 1108) or fit_y.shape != (n_total,):
            raise RuntimeError('Combined training shapes invalid')
        model = clone(svc).set_params(C=1.0, gamma='scale', cache_size=args.cache_size)
        say(f'[Fold heldout={heldout}] fit START rows={n_total}')
        t = time.perf_counter()
        model.fit(fit_x, fit_y)
        fit_sec = round(time.perf_counter()-t, 2)
        del fit_x, fit_y
        gc.collect()
        say(f'[Fold heldout={heldout}] fit DONE {fit_sec}s support_vectors={len(model.support_)}')
        pred, hp_sec = predict_batches(model, writer_feat[heldout],
                                       f'Writer {heldout}', args.batch_size)
        hm = report(handmade[heldout][1], pred)
        say(f'[Fold heldout={heldout}] Hand-made={hm["accuracy"]:.6f} ({hm["correct"]}/640)')
        val_pred, val_sec = predict_batches(model, validation_feat,
                                             f'Official Val fold {heldout}', args.batch_size)
        vm = report(yv, val_pred)
        say(f'[Fold heldout={heldout}] Official Validation={vm["accuracy"]:.6f}')
        rec = {
            'status': 'complete', 'heldout_writer': heldout,
            'train_writers': train_writers, 'extra_per_class_writer': args.extra_per_class_writer,
            'official_dataset_sha256': sha,
            'handmade_sha256': {str(w): handmade[w][2] for w in WRITERS},
            'official_train_n': 76800, 'original_handmade_train_n': 1920,
            'base_augmented_handmade_n': 1920, 'targeted_augmented_handmade_n': n_target,
            'train_n': n_total, 'augmentation_seed': SEED+heldout,
            'augmentation_strength': AUG_STRENGTH, 'target_seed': TARGET_SEED,
            'base_size_stats': base_size_stats, 'base_stroke_stats': base_stroke_stats,
            'target_size_stats': target_size_stats, 'target_stroke_stats': target_stroke_stats,
            'parameters': {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf',
                           'cache_size_mb': args.cache_size},
            'fit_seconds': fit_sec, 'support_vectors': int(len(model.support_)),
            'handmade_predict_seconds': hp_sec, 'official_predict_seconds': val_sec,
            'handmade': hm, 'official_validation': vm,
        }
        write_csv(csvpath, [
            {'writer': heldout, 'index': i, 'true': LABELS[int(handmade[heldout][1][i])],
             'control_pred': LABELS[int(control_preds[heldout][i])],
             'targeted_pred': LABELS[int(pred[i])]} for i in range(640)
        ], ['writer', 'index', 'true', 'control_pred', 'targeted_pred'])
        atomic_json(jsonpath, rec)
        records[heldout], predictions[heldout] = rec, pred
        del model, val_pred, pred
        gc.collect()
    truth = np.concatenate([handmade[w][1] for w in WRITERS])
    control = np.concatenate([control_preds[w] for w in WRITERS])
    candidate = np.concatenate([predictions[w] for w in WRITERS])
    baseline_result = report(truth, control)
    result = report(truth, candidate)
    paired = {
        'corrected': int(np.sum((control != truth) & (candidate == truth))),
        'new_errors': int(np.sum((control == truth) & (candidate != truth))),
        'unchanged_correct': int(np.sum((control == truth) & (candidate == truth))),
        'unchanged_wrong': int(np.sum((control != truth) & (candidate != truth))),
    }
    cm = confusion_matrix(truth, candidate, labels=np.arange(16))
    write_csv(OUT / f'{prefix}_confusion_matrix.csv', [
        {'true': LABELS[i], **{LABELS[j]: int(cm[i,j]) for j in range(16)}} for i in range(16)
    ], ['true', *LABELS])
    write_csv(OUT / f'{prefix}_oof_predictions.csv', [
        {'writer': w, 'index': i, 'true': LABELS[int(handmade[w][1][i])],
         'control_pred': LABELS[int(control_preds[w][i])],
         'targeted_pred': LABELS[int(predictions[w][i])]} for w in WRITERS for i in range(640)
    ], ['writer', 'index', 'true', 'control_pred', 'targeted_pred'])
    fold_rows = []
    for w in WRITERS:
        rec = records[w]
        fold_rows.append({
            'extra_per_class_writer': args.extra_per_class_writer, 'heldout_writer': w,
            'official_train_n': 76800, 'handmade_original_train_n': 1920,
            'handmade_base_augmented_n': 1920, 'handmade_targeted_augmented_n': n_target,
            'train_n': n_total, 'handmade_correct': rec['handmade']['correct'],
            'handmade_accuracy': rec['handmade']['accuracy'],
            'handmade_macro_f1': rec['handmade']['macro_f1'],
            'official_correct': rec['official_validation']['correct'],
            'official_accuracy': rec['official_validation']['accuracy'],
            'official_macro_f1': rec['official_validation']['macro_f1'],
            'fit_seconds': rec['fit_seconds'], 'support_vectors': rec['support_vectors'],
        })
    write_csv(OUT / f'{prefix}_fold_results.csv', fold_rows, FIELDS)
    classes = []
    for name in ('D', 'B', '0', '8'):
        label = LABELS.index(name)
        mask = truth == label
        classes.append({
            'class': name, 'n': int(mask.sum()),
            'control_correct': int((control[mask] == label).sum()),
            'targeted_correct': int((candidate[mask] == label).sum()),
            'control_recall': float(np.mean(control[mask] == label)),
            'targeted_recall': float(np.mean(candidate[mask] == label)),
        })
    write_csv(OUT / f'{prefix}_critical_classes.csv', classes,
              ['class', 'n', 'control_correct', 'targeted_correct', 'control_recall', 'targeted_recall'])
    output = {
        'experiment': 'EXP-CMB-005 SVC Combined targeted D/B after 100% augmentation',
        'extra_per_class_writer': args.extra_per_class_writer,
        'official_training_n': 76800, 'official_validation_n': 19200,
        'original_handmade_n_per_fold': 1920, 'base_augmented_n_per_fold': 1920,
        'targeted_augmented_n_per_fold': n_target, 'train_n_per_fold': n_total,
        'official_dataset_sha256': sha, 'official_test_used': False,
        'feature': 'CoM -> Deskew -> Pixel20/HOG80',
        'reference_experiment': str(AUG100.relative_to(ROOT)),
        'reference_oof': baseline_result, 'targeted_oof': result,
        'change_percentage_points': 100 * (result['accuracy'] - baseline_result['accuracy']),
        'paired': paired, 'critical_classes': classes,
        'folds': [records[w] for w in WRITERS],
        'note': 'Hand-made OOF repeatedly used for model development, not independent Test.',
    }
    atomic_json(OUT / f'{prefix}_metrics.json', output)
    say(f'\nEXP-CMB-005 SUMMARY D/B +{n_target} /fold')
    say(f'100% control={baseline_result["accuracy"]:.6f} ({baseline_result["correct"]}/2560)')
    say(f'Targeted    ={result["accuracy"]:.6f} ({result["correct"]}/2560), MacroF1={result["macro_f1"]:.6f}')
    say(f'Change={output["change_percentage_points"]:+.4f}pp  Paired={paired}')
    say('Critical class recalls:', [(r['class'], round(r['control_recall'],4),
                                      round(r['targeted_recall'],4)) for r in classes])
    say('Official Validation by fold:', [round(records[w]['official_validation']['accuracy'],6)
                                          for w in WRITERS])
    say('Saved:', OUT)
    say('Official Test NOT loaded.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra-per-class-writer', type=int, choices=[20, 40], default=20)
    parser.add_argument('--batch-size', type=int, choices=[64, 128, 256, 512], default=256)
    parser.add_argument('--cache-size', type=int, default=200)
    parser.add_argument('--overwrite', action='store_true')
    run(parser.parse_args())
