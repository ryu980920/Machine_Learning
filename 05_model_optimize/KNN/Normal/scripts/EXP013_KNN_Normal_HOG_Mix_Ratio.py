#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-013: Pixel/HOG feature weighting for KNN Normal.

Run in repository root:
python 05_model_optimize/KNN/Normal/scripts/EXP013_KNN_Normal_HOG_Mix_Ratio.py

Requires existing EXP004, EXP007, EXP011, EXP012 scripts in the same directory.
Original official Training only; no official Test or Hand-made training.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
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
from EXP011_KNN_Normal_Deskew import deskew, test_deskew
from EXP012_KNN_Normal_HOG import hog_features, hybrid_features, test_hog, unit_rows

OUT = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'EXP013_hog_mix_ratio'
WEIGHTS = (0.0, 0.2, 0.35, 0.5, 0.65)
FRACTION = 0.30
STROKE = 0.44
REFERENCE_HOG = (18667, 2341)  # verified from EXP-012 (official, handmade correct)
REFERENCE_MIX50 = (18854, 2341)


def features(pixel, hog, weight):
    if weight == 0.0:
        return hog
    return hybrid_features(pixel, hog, pixel_weight=weight)


def evaluate(name, weight, xtr, ytr, xva, yva, xhm, yhm, writers, indices,
             records, rows, by_writer, per_class, baseline_preds):
    print(f'\n[{name}] KNN fit / predict...', flush=True)
    model = KNeighborsClassifier(n_neighbors=3, weights='distance',
                                 metric='euclidean', algorithm='brute', n_jobs=-1)
    t = time.perf_counter()
    model.fit(xtr, ytr)
    fit_s = time.perf_counter() - t
    t = time.perf_counter()
    pv = np.asarray(model.predict(xva), dtype=np.int64)
    val_s = time.perf_counter() - t
    t = time.perf_counter()
    ph = np.asarray(model.predict(xhm), dtype=np.int64)
    hm_s = time.perf_counter() - t
    off, handmade = scores(yva, pv), scores(yhm, ph)
    correct = (int(np.sum(pv == yva)), int(np.sum(ph == yhm)))
    if weight == 0.0:
        if correct != REFERENCE_HOG:
            raise RuntimeError(f'EXP-012 HOG reference mismatch: expected {REFERENCE_HOG}, got {correct}')
        baseline_preds['official'], baseline_preds['handmade'] = pv.copy(), ph.copy()
        print('EXP-012 HOG baseline reproduced exactly', flush=True)
    if weight == 0.5:
        if correct != REFERENCE_MIX50:
            raise RuntimeError(f'EXP-012 50:50 reference mismatch: expected {REFERENCE_MIX50}, got {correct}')
        print('EXP-012 50:50 reference reproduced exactly', flush=True)
    row = {
        'variant': name, 'pixel_weight': weight, 'hog_weight': 1.0 - weight,
        'train_n': len(ytr), 'n_features': xtr.shape[1],
        'official_correct': correct[0], 'official_accuracy': off['accuracy'],
        'official_macro_f1': off['macro_f1'],
        'handmade_correct': correct[1], 'handmade_accuracy': handmade['accuracy'],
        'handmade_macro_f1': handmade['macro_f1'],
        'official_guardrail_pass': int(off['accuracy'] >= OFFICIAL_GUARDRAIL),
        'fit_seconds': fit_s, 'official_predict_seconds': val_s,
        'handmade_predict_seconds': hm_s,
    }
    rows.append(row)
    for w in (41, 42, 43, 44):
        mask = writers == w
        sc = scores(yhm[mask], ph[mask])
        by_writer.append({'variant': name, 'writer': w, 'n': int(np.sum(mask)),
                          'accuracy': sc['accuracy'], 'macro_f1': sc['macro_f1']})
    for c in LABELS:
        s = handmade['per_class'][c]
        per_class.append({'variant': name, 'class': c, 'support': s['support'],
                          'precision': s['precision'], 'recall': s['recall'], 'f1': s['f1']})
    records[name] = {'row': row, 'official': off, 'handmade': handmade}
    if weight != 0.0:
        records[name]['paired_vs_hog'] = {
            'official': paired_stats(yva, baseline_preds['official'], pv),
            'handmade': paired_stats(yhm, baseline_preds['handmade'], ph),
        }
    save_csv(OUT / 'candidate_results.csv', rows)
    save_csv(OUT / 'by_writer.csv', by_writer)
    save_csv(OUT / 'per_class_handmade.csv', per_class)
    save_cm(OUT / f'official_cm_{name}.csv', off['confusion_matrix'])
    save_cm(OUT / f'handmade_cm_{name}.csv', handmade['confusion_matrix'])
    save_csv(OUT / f'handmade_predictions_{name}.csv', [
        {'writer': int(writers[i]), 'index': int(indices[i]),
         'true': LABELS[int(yhm[i])], 'predicted': LABELS[int(ph[i])],
         'correct': int(yhm[i] == ph[i])} for i in range(len(yhm))
    ])
    (OUT / 'metrics_partial.json').write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{name:<19} Official={off['accuracy']:.6f} F1={off['macro_f1']:.6f} | "
          f"Hand-made={handmade['accuracy']:.6f} F1={handmade['macro_f1']:.6f} | "
          f"guardrail={'PASS' if row['official_guardrail_pass'] else 'FAIL'} | "
          f'predict_s={val_s:.1f}+{hm_s:.1f}', flush=True)
    del model


def main():
    test_augmentation()
    test_deskew()
    test_hog()
    assert np.isclose(sum((0.35, 0.65)), 1.0)
    xtr, ytr, xva, yva, sha = fixed_split()
    xhm, yhm, writers, indices = read_handmade()
    assert (len(ytr), len(yva), len(yhm)) == (76800, 19200, 2560)
    selected = sample_balanced(ytr, fraction=FRACTION, seed=SEED)
    sized, size_stats = size_jitter(xtr, selected, seed=SEED)
    extra, stroke_stats = stroke_jitter(sized, strength=STROKE)
    raw_train = np.concatenate((xtr, extra), axis=0)
    train_labels = np.concatenate((ytr, ytr[selected]), axis=0)
    assert len(train_labels) == 99840
    print(f'Training={len(ytr)}, Official Validation={len(yva)}, Hand-made={len(yhm)}', flush=True)
    print(f'Augmented official Training only: {len(selected)} rows', flush=True)
    t = time.perf_counter()
    com = BatchedCoM()
    xtrain, _ = deskew(com.transform(raw_train))
    xval, _ = deskew(com.transform(xva))
    xhand, _ = deskew(com.transform(xhm))
    del raw_train, extra, sized
    print(f'CoM -> Deskew finished in {time.perf_counter() - t:.1f}s', flush=True)
    t = time.perf_counter()
    htr = hog_features(xtrain)
    hva = hog_features(xval)
    hhm = hog_features(xhand)
    print(f'HOG extracted once in {time.perf_counter() - t:.1f}s', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, by_writer, per_class, records, baseline_preds = [], [], [], {}, {}
    for weight in WEIGHTS:
        name = 'hog_only' if weight == 0.0 else f'pixel_{int(round(weight * 100)):02d}_hog_{int(round((1-weight)*100)):02d}'
        train_features = features(xtrain, htr, weight)
        val_features = features(xval, hva, weight)
        hand_features = features(xhand, hhm, weight)
        evaluate(name, weight, train_features, train_labels, val_features, yva,
                 hand_features, yhm, writers, indices, records, rows,
                 by_writer, per_class, baseline_preds)
        del train_features, val_features, hand_features
    eligible = [r for r in rows if r['official_guardrail_pass']]
    winner = max(eligible, key=lambda r: (r['handmade_macro_f1'],
                                         r['handmade_accuracy'],
                                         r['official_accuracy'])) if eligible else None
    payload = {
        'experiment': 'EXP-013', 'dataset_sha256': sha,
        'official_training_n': len(ytr), 'augmented_training_n': len(selected),
        'train_n_total': len(train_labels), 'validation_n': len(yva),
        'handmade_dev_n': len(yhm),
        'augmentation': {'fraction': FRACTION, 'stroke_strength': STROKE,
                         'size_stats': size_stats, 'stroke_stats': stroke_stats},
        'preprocessing': 'CoM -> Deskew',
        'knn': {'n_neighbors': 3, 'weights': 'distance', 'metric': 'euclidean'},
        'hog_dim': htr.shape[1], 'pixel_dim': xtrain.shape[1],
        'hybrid': 'Per-image L2 normalization of pixel and HOG, concatenated after sqrt-weighting',
        'weights_evaluated': list(WEIGHTS), 'candidate_results': rows,
        'selected_development_candidate': winner, 'results': records,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'sklearn': sklearn.__version__},
        'note': 'Hand-made has been reused for model selection; this is development validation, not a final unseen test. Official Test not used.',
    }
    (OUT / 'metrics.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nSELECTED DEVELOPMENT CANDIDATE:', winner['variant'] if winner else 'NONE', flush=True)
    print('Saved:', OUT, flush=True)


if __name__ == '__main__':
    main()
