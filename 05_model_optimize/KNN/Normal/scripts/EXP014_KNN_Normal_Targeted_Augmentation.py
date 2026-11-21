#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-014: class-targeted official-training augmentation for KNN Normal.

Run from repository root:
python 05_model_optimize/KNN/Normal/scripts/EXP014_KNN_Normal_Targeted_Augmentation.py

Requires EXP004, EXP006_fixed, EXP007, EXP011, EXP012, EXP013 scripts
and EXP013_hog_mix_ratio/handmade_predictions_pixel_20_hog_80.csv.
Hand-made is used ONLY to select weak classes and evaluate, NEVER to fit.
Such repeated model selection can overfit the Hand-made development set.
The official Test is never loaded.
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'
sys.path.insert(0, str(ROOT / '03_experiments' / 'model_optimization'))
sys.path.insert(0, str(SCRIPTS))

from common import LABELS, fixed_split, scores
from EXP004_KNN_Normal_CoM import BatchedCoM, read_handmade, paired_stats
from EXP007_KNN_Normal_Augmentation import (
    sample_balanced, size_jitter, stroke_jitter, test_augmentation,
    save_csv, save_cm, OFFICIAL_GUARDRAIL, SEED,
)
from EXP011_KNN_Normal_Deskew import deskew, test_deskew
from EXP012_KNN_Normal_HOG import hog_features, hybrid_features, test_hog

OUT = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'EXP014_targeted_augmentation'
PRED_CSV = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'EXP013_hog_mix_ratio' / 'handmade_predictions_pixel_20_hog_80.csv'
BASE_FRACTION = 0.30
STROKE_STRENGTH = 0.44
PIXEL_WEIGHT = 0.20
N_WEAK_CLASSES = 4
EXTRA_COUNTS = (3840, 7680)   # +5%, +10% of original official train (76,800)
CONTROL_CORRECT = (18851, 2360)  # exact EXP-013 20:80 baseline


def weak_classes_from_dev_predictions(path, y_hm, writers, indices):
    if not path.is_file():
        raise FileNotFoundError(f'EXP-013 prediction file not found: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(y_hm):
        raise RuntimeError(f'EXP-013 prediction count mismatch: {len(rows)} != {len(y_hm)}')
    counts = Counter()
    pairs = Counter()
    correct = 0
    for i, row in enumerate(rows):
        true = LABELS[int(y_hm[i])]
        if (int(row['writer']) != int(writers[i]) or
                int(row['index']) != int(indices[i]) or
                row['true'] != true):
            raise RuntimeError(f'EXP-013 prediction order or true labels mismatch at row {i}')
        pred = row['predicted']
        if pred not in LABELS:
            raise RuntimeError(f'Unexpected prediction {pred}')
        if pred == true:
            correct += 1
        else:
            counts[true] += 1
            pairs[(true, pred)] += 1
    if correct != CONTROL_CORRECT[1]:
        raise RuntimeError(f'EXP-013 Hand-made prediction reference mismatch: {correct}')
    rank = sorted(range(16), key=lambda c: (-counts[LABELS[c]], c))
    selected = rank[:N_WEAK_CLASSES]
    if len(selected) != N_WEAK_CLASSES:
        raise RuntimeError('Insufficient targeted classes')
    return selected, counts, pairs


def select_extra(y, already_selected, total, target_classes, seed):
    """Choose additional official-training samples disjoint from the baseline aug indices."""
    rng = np.random.default_rng(seed)
    blocked = np.zeros(len(y), dtype=bool)
    blocked[already_selected] = True
    classes = list(target_classes)
    if total % len(classes):
        raise ValueError('Extra count must be divisible by target class count')
    per_class = total // len(classes)
    pieces = []
    for cls in classes:
        available = np.flatnonzero((y == cls) & ~blocked)
        if len(available) < per_class:
            raise RuntimeError(f'Insufficient unused training examples for {LABELS[cls]}: '
                               f'needed={per_class}, available={len(available)}')
        pieces.append(rng.choice(available, size=per_class, replace=False))
    chosen = np.sort(np.concatenate(pieces))
    if len(chosen) != total or len(np.unique(chosen)) != total or np.any(blocked[chosen]):
        raise RuntimeError('Additional sample selection check failed')
    return chosen


def augmented_features(x_train, indices, com, name):
    print(f'[{name}] Generating {len(indices)} official-training-only augmented rows...', flush=True)
    sized, size_stats = size_jitter(x_train, indices, seed=SEED + 140)
    strokes, stroke_stats = stroke_jitter(sized, strength=STROKE_STRENGTH)
    centered = com.transform(strokes)
    corrected, deskew_stats = deskew(centered)
    hog = hog_features(corrected)
    features = hybrid_features(corrected, hog, pixel_weight=PIXEL_WEIGHT)
    del sized, strokes, centered, corrected, hog
    return features, {'size': size_stats, 'stroke': stroke_stats, 'deskew': deskew_stats}


def evaluate(name, features_train, labels_train, val_features, y_val,
             hm_features, y_hm, writers, indices, records, rows, writer_rows,
             class_rows, baseline_predictions):
    print(f'\n[{name}] Fit / predict ({len(labels_train)} training rows)...', flush=True)
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean',
                               algorithm='brute', n_jobs=-1)
    t = time.perf_counter()
    knn.fit(features_train, labels_train)
    fit_s = time.perf_counter() - t
    t = time.perf_counter()
    pred_val = np.asarray(knn.predict(val_features), dtype=np.int64)
    val_s = time.perf_counter() - t
    t = time.perf_counter()
    pred_hm = np.asarray(knn.predict(hm_features), dtype=np.int64)
    hm_s = time.perf_counter() - t
    official = scores(y_val, pred_val)
    handmade = scores(y_hm, pred_hm)
    correct = (int((pred_val == y_val).sum()), int((pred_hm == y_hm).sum()))
    if name == 'control':
        if correct != CONTROL_CORRECT:
            raise RuntimeError(f'EXP-013 20:80 control mismatch: {correct} != {CONTROL_CORRECT}')
        baseline_predictions['official'] = pred_val.copy()
        baseline_predictions['handmade'] = pred_hm.copy()
        print('EXP-013 20:80 baseline reproduced exactly', flush=True)
    record = {
        'variant': name, 'train_n': int(len(labels_train)),
        'extra_n': int(len(labels_train) - 99840),
        'official_correct': correct[0], 'official_accuracy': official['accuracy'],
        'official_macro_f1': official['macro_f1'],
        'handmade_correct': correct[1], 'handmade_accuracy': handmade['accuracy'],
        'handmade_macro_f1': handmade['macro_f1'],
        'official_guardrail_pass': int(official['accuracy'] >= OFFICIAL_GUARDRAIL),
        'fit_seconds': fit_s, 'official_predict_seconds': val_s,
        'handmade_predict_seconds': hm_s,
    }
    rows.append(record)
    records[name] = {'row': record, 'official': official, 'handmade': handmade}
    if name != 'control':
        records[name]['paired_vs_control'] = {
            'official': paired_stats(y_val, baseline_predictions['official'], pred_val),
            'handmade': paired_stats(y_hm, baseline_predictions['handmade'], pred_hm),
        }
    for w in (41, 42, 43, 44):
        m = writers == w
        score = scores(y_hm[m], pred_hm[m])
        writer_rows.append({'variant': name, 'writer': w, 'n': int(m.sum()),
                            'accuracy': score['accuracy'], 'macro_f1': score['macro_f1']})
    for c in LABELS:
        s = handmade['per_class'][c]
        class_rows.append({'variant': name, 'class': c, 'support': s['support'],
                           'precision': s['precision'], 'recall': s['recall'], 'f1': s['f1']})
    save_csv(OUT / 'candidate_results.csv', rows)
    save_csv(OUT / 'by_writer.csv', writer_rows)
    save_csv(OUT / 'per_class_handmade.csv', class_rows)
    save_cm(OUT / f'official_cm_{name}.csv', official['confusion_matrix'])
    save_cm(OUT / f'handmade_cm_{name}.csv', handmade['confusion_matrix'])
    save_csv(OUT / f'handmade_predictions_{name}.csv', [
        {'writer': int(writers[i]), 'index': int(indices[i]),
         'true': LABELS[int(y_hm[i])], 'predicted': LABELS[int(pred_hm[i])],
         'correct': int(pred_hm[i] == y_hm[i])} for i in range(len(y_hm))
    ])
    (OUT / 'metrics_partial.json').write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{name:<17} Official={official["accuracy"]:.6f} F1={official["macro_f1"]:.6f} | '
          f'Hand-made={handmade["accuracy"]:.6f} F1={handmade["macro_f1"]:.6f} | '
          f'guardrail={"PASS" if record["official_guardrail_pass"] else "FAIL"} | '
          f'predict_s={val_s:.1f}+{hm_s:.1f}', flush=True)
    del knn


def main():
    test_augmentation()
    test_deskew()
    test_hog()
    xtr, ytr, xva, yva, sha = fixed_split()
    xhm, yhm, writers, indices = read_handmade()
    if (len(ytr), len(yva), len(yhm)) != (76800, 19200, 2560):
        raise RuntimeError('Unexpected dataset sizes')
    weak, error_counts, confused_pairs = weak_classes_from_dev_predictions(
        PRED_CSV, yhm, writers, indices)
    print('Weak classes selected from EXP-013 Hand-made DEVELOPMENT predictions:',
          [LABELS[c] for c in weak], flush=True)
    print('Errors in selected classes:', {LABELS[c]: error_counts[LABELS[c]] for c in weak}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    save_csv(OUT / 'selected_classes.csv', [
        {'class': c, 'handmade_errors': error_counts[c],
         'selected_for_targeted_augmentation': int(c in {LABELS[k] for k in weak})}
        for c in LABELS
    ])
    save_csv(OUT / 'top_confusions.csv', [
        {'true': pair[0], 'predicted': pair[1], 'count': count}
        for pair, count in sorted(confused_pairs.items(), key=lambda x: (-x[1], x[0]))
    ])
    selected = sample_balanced(ytr, fraction=BASE_FRACTION, seed=SEED)
    base_size, base_size_stats = size_jitter(xtr, selected, seed=SEED)
    base_extra, base_stroke_stats = stroke_jitter(base_size, strength=STROKE_STRENGTH)
    raw_base_train = np.concatenate([xtr, base_extra], axis=0)
    base_labels = np.concatenate([ytr, ytr[selected]], axis=0)
    if len(base_labels) != 99840:
        raise RuntimeError('EXP-013 baseline augmentation count mismatch')
    del base_size, base_extra
    com = BatchedCoM()
    t = time.perf_counter()
    base_images, _ = deskew(com.transform(raw_base_train))
    val_images, _ = deskew(com.transform(xva))
    hand_images, _ = deskew(com.transform(xhm))
    del raw_base_train
    print(f'Preprocessing done in {time.perf_counter()-t:.1f}s', flush=True)
    t = time.perf_counter()
    base_feats = hybrid_features(base_images, hog_features(base_images), pixel_weight=PIXEL_WEIGHT)
    val_feats = hybrid_features(val_images, hog_features(val_images), pixel_weight=PIXEL_WEIGHT)
    hm_feats = hybrid_features(hand_images, hog_features(hand_images), pixel_weight=PIXEL_WEIGHT)
    del base_images, val_images, hand_images
    print(f'Pixel:HOG 20:80 features done in {time.perf_counter()-t:.1f}s', flush=True)
    rows, writer_rows, class_rows, records, control_preds = [], [], [], {}, {}
    evaluate('control', base_feats, base_labels, val_feats, yva, hm_feats, yhm,
             writers, indices, records, rows, writer_rows, class_rows, control_preds)

    variants = [
        ('uniform_5', EXTRA_COUNTS[0], list(range(16)), SEED + 141),
        ('targeted_5', EXTRA_COUNTS[0], weak, SEED + 142),
        ('targeted_10', EXTRA_COUNTS[1], weak, SEED + 142),
    ]
    selection_records = {}
    for name, n, classes, seed in variants:
        ix = select_extra(ytr, selected, n, classes, seed=seed)
        extras, stats = augmented_features(xtr, ix, com, name)
        combined_features = np.concatenate([base_feats, extras], axis=0)
        combined_labels = np.concatenate([base_labels, ytr[ix]], axis=0)
        selection_records[name] = {
            'added_n': int(len(ix)), 'classes': [LABELS[k] for k in classes],
            'counts': {LABELS[k]: int(np.sum(ytr[ix] == k)) for k in classes},
            'stats': stats,
        }
        evaluate(name, combined_features, combined_labels, val_feats, yva,
                 hm_feats, yhm, writers, indices, records, rows, writer_rows,
                 class_rows, control_preds)
        del extras, combined_features, combined_labels

    eligible = [r for r in rows if r['official_guardrail_pass']]
    winner = max(eligible, key=lambda r: (r['handmade_macro_f1'],
                                         r['handmade_accuracy'], r['official_accuracy']))
    summary = {
        'experiment': 'EXP-014', 'dataset_sha256': sha,
        'official_train_n': len(ytr), 'official_validation_n': len(yva),
        'handmade_development_n': len(yhm), 'baseline_train_n': len(base_labels),
        'baseline_aug_fraction': BASE_FRACTION, 'stroke_strength': STROKE_STRENGTH,
        'base_size_stats': base_size_stats, 'base_stroke_stats': base_stroke_stats,
        'knn': {'k': 3, 'weights': 'distance', 'metric': 'euclidean'},
        'preprocessing': 'CoM -> Deskew', 'pixel_weight': PIXEL_WEIGHT,
        'weak_classes': [LABELS[c] for c in weak],
        'selection_source': str(PRED_CSV.relative_to(ROOT)),
        'targeting_counts': {LABELS[c]: error_counts[LABELS[c]] for c in weak},
        'additional_augmentation': selection_records,
        'candidates': rows, 'selected_development_candidate': winner,
        'results': records,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'sklearn': sklearn.__version__},
        'note': 'Target classes chosen from repeatedly used Hand-made development labels. '
                'Only official Training images were fitted/augmented. Do not claim '
                'unbiased generalization. Official Test not accessed.',
    }
    (OUT / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nSELECTED DEVELOPMENT CANDIDATE:', winner['variant'], flush=True)
    print('Saved:', OUT, flush=True)


if __name__ == '__main__':
    main()
