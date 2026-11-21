#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-015: KNN k/weights sweep for the EXACT EXP-014 targeted-10 training set.

Run from repository root:
  python 05_model_optimize/KNN/Normal/scripts/EXP015_KNN_Normal_K_Weights_HOG.py

Only official Training is fitted. Validation and four Hand-made writers are
used for development evaluation. Official Test is never accessed.

Uses the exact EXP014 helper functions/seed/augmentation. A single 11-NN
query is reused across all k/weight variants. Supports k=1,3,5,7,9,11 and
uniform/distance. Results are development results, not held-out Test scores.
"""
from __future__ import annotations

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

from common import LABELS, fixed_split, scores
from EXP004_KNN_Normal_CoM import BatchedCoM, read_handmade, paired_stats, test_centering
from EXP007_KNN_Normal_Augmentation import (
    sample_balanced, size_jitter, stroke_jitter, test_augmentation,
    save_csv, save_cm, OFFICIAL_GUARDRAIL, SEED,
)
from EXP011_KNN_Normal_Deskew import deskew, test_deskew
from EXP012_KNN_Normal_HOG import hog_features, hybrid_features, test_hog
from EXP014_KNN_Normal_Targeted_Augmentation import (
    weak_classes_from_dev_predictions, select_extra, augmented_features,
    PRED_CSV, BASE_FRACTION, STROKE_STRENGTH, PIXEL_WEIGHT,
    CONTROL_CORRECT as EXP013_CONTROL_CORRECT,
)

OUT = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'EXP015_k_weights_hog'
K_VALUES = (1, 3, 5, 7, 9, 11)
WEIGHTS = ('uniform', 'distance')
EXPECTED_EXP014 = (18851, 2381)  # EXP014 targeted_10 official / Hand-made correct
TARGETED_EXTRA_N = 7680


def predict_from_neighbors(ix: np.ndarray, d: np.ndarray, train_y: np.ndarray,
                           k: int, weights: str) -> np.ndarray:
    """Exact sklearn vote semantics for 16 integer labels, including zero distances.

    Ties are resolved to smallest label (as in sklearn's weighted_mode/mode).
    """
    labels = train_y[ix[:, :k]]
    distances = d[:, :k]
    if weights == 'uniform':
        vote_weights = np.ones_like(distances, dtype=np.float64)
    elif weights == 'distance':
        zero = distances == 0
        vote_weights = np.zeros_like(distances, dtype=np.float64)
        np.divide(1.0, distances, out=vote_weights, where=~zero)
        vote_weights = np.where(zero.any(axis=1, keepdims=True),
                                zero.astype(np.float64), vote_weights)
    else:
        raise ValueError(weights)
    votes = np.zeros((len(labels), len(LABELS)), dtype=np.float64)
    for label in range(len(LABELS)):
        votes[:, label] = np.sum(vote_weights * (labels == label), axis=1)
    return np.argmax(votes, axis=1).astype(np.int64)


def voting_self_test():
    y = np.array([1, 2, 1, 2, 3], dtype=np.int64)
    i = np.array([[0, 1, 2], [1, 3, 4], [0, 1, 4]])
    d = np.array([[1., 2., 3.], [0., 0., 3.], [0., 0., 0.]])
    assert np.array_equal(predict_from_neighbors(i, d, y, 3, 'uniform'), [1, 2, 1])
    assert np.array_equal(predict_from_neighbors(i, d, y, 3, 'distance'), [1, 2, 1])
    print('VOTING SELF-TEST PASSED: uniform, inverse distance, zero-distance ties', flush=True)


def save_predictions(name, pred, yhm, writers, local_indices):
    save_csv(OUT / f'handmade_predictions_{name}.csv', [
        {'writer': int(writers[i]), 'index': int(local_indices[i]),
         'true': LABELS[int(yhm[i])], 'predicted': LABELS[int(pred[i])],
         'correct': int(pred[i] == yhm[i])} for i in range(len(yhm))
    ])


def main():
    test_centering()
    test_augmentation()
    test_deskew()
    test_hog()
    voting_self_test()
    xtr, ytr, xva, yva, sha = fixed_split()
    xhm, yhm, writers, local_indices = read_handmade()
    if (len(ytr), len(yva), len(yhm)) != (76800, 19200, 2560):
        raise RuntimeError('Unexpected data split')
    weak, counts, _ = weak_classes_from_dev_predictions(
        PRED_CSV, yhm, writers, local_indices)
    if [LABELS[i] for i in weak] != ['D', 'B', '5', '3']:
        raise RuntimeError('EXP014 weak-class selection unexpectedly changed')
    print('Selected targeted classes:', [LABELS[i] for i in weak], flush=True)
    print('Official Training=76800; Validation=19200; Hand-made=2560', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # Recreate EXACT EXP014 targeted_10 training features: order, seed, transforms.
    selected = sample_balanced(ytr, fraction=BASE_FRACTION, seed=SEED)
    sized, _ = size_jitter(xtr, selected, seed=SEED)
    strokes, _ = stroke_jitter(sized, strength=STROKE_STRENGTH)
    raw_base = np.concatenate([xtr, strokes], axis=0)
    labels_base = np.concatenate([ytr, ytr[selected]], axis=0)
    del sized, strokes

    com = BatchedCoM()
    t0 = time.perf_counter()
    base_img, _ = deskew(com.transform(raw_base))
    val_img, _ = deskew(com.transform(xva))
    hm_img, _ = deskew(com.transform(xhm))
    del raw_base
    print(f'CoM -> Deskew preprocessing: {time.perf_counter()-t0:.1f}s', flush=True)
    t0 = time.perf_counter()
    base_feat = hybrid_features(base_img, hog_features(base_img), pixel_weight=PIXEL_WEIGHT)
    val_feat = hybrid_features(val_img, hog_features(val_img), pixel_weight=PIXEL_WEIGHT)
    hm_feat = hybrid_features(hm_img, hog_features(hm_img), pixel_weight=PIXEL_WEIGHT)
    del base_img, val_img, hm_img
    print(f'Pixel:HOG 20:80 extraction: {time.perf_counter()-t0:.1f}s', flush=True)

    indices = select_extra(ytr, selected, TARGETED_EXTRA_N, weak, seed=SEED + 142)
    extra_feat, _ = augmented_features(xtr, indices, com, 'targeted_10')
    features = np.ascontiguousarray(np.concatenate([base_feat, extra_feat], axis=0), dtype=np.float32)
    labels = np.concatenate([labels_base, ytr[indices]]).astype(np.int64)
    del base_feat, extra_feat, labels_base
    if len(labels) != 107520 or features.shape != (107520, 1108):
        raise RuntimeError(f'Wrong EXP014 training shape: {features.shape}')
    print(f'EXACT EXP-014 targeted_10 training rows: {len(labels)}, features: {features.shape[1]}', flush=True)

    # sklearn's Euclidean (p=2) brute search; compute all 11 neighbors once.
    knn = KNeighborsClassifier(n_neighbors=max(K_VALUES), weights='distance',
                               metric='euclidean', algorithm='brute', n_jobs=-1)
    t0 = time.perf_counter()
    knn.fit(features, labels)
    fit_sec = time.perf_counter() - t0
    print('Querying 11 neighbors for official Validation (once)...', flush=True)
    t0 = time.perf_counter()
    dv, iv = knn.kneighbors(val_feat, n_neighbors=max(K_VALUES), return_distance=True)
    val_sec = time.perf_counter() - t0
    print(f'Official neighbors obtained in {val_sec:.1f}s', flush=True)
    print('Querying 11 neighbors for Hand-made (once)...', flush=True)
    t0 = time.perf_counter()
    dh, ih = knn.kneighbors(hm_feat, n_neighbors=max(K_VALUES), return_distance=True)
    hm_sec = time.perf_counter() - t0
    print(f'Hand-made neighbors obtained in {hm_sec:.1f}s', flush=True)

    # Check custom vote against sklearn itself on a subset before full comparison.
    for feature_array, indices_array, distances_array in ((val_feat, iv, dv), (hm_feat, ih, dh)):
        knn.set_params(n_neighbors=3, weights='distance')
        sklearn_pred = np.asarray(knn.predict(feature_array[:128]), dtype=np.int64)
        our_pred = predict_from_neighbors(indices_array[:128], distances_array[:128],
                                          labels, 3, 'distance')
        if not np.array_equal(sklearn_pred, our_pred):
            raise RuntimeError('Custom voting does not reproduce sklearn on a test batch')

    rows, writer_rows, class_rows, records = [], [], [], {}
    control_pv = control_ph = None
    for k in K_VALUES:
        for weights in WEIGHTS:
            name = f'k{k}_{weights}'
            pv = predict_from_neighbors(iv, dv, labels, k, weights)
            ph = predict_from_neighbors(ih, dh, labels, k, weights)
            ov, hh = scores(yva, pv), scores(yhm, ph)
            correct = (int(np.sum(pv == yva)), int(np.sum(ph == yhm)))
            if k == 3 and weights == 'distance':
                if correct != EXPECTED_EXP014:
                    raise RuntimeError(
                        f'EXP014 targeted_10 reproduction failed: {correct} != {EXPECTED_EXP014}')
                control_pv, control_ph = pv.copy(), ph.copy()
                print('EXP-014 targeted_10 k3_distance reproduced exactly', flush=True)
            row = {
                'variant': name, 'k': k, 'weights': weights, 'train_n': len(labels),
                'official_correct': correct[0], 'official_accuracy': ov['accuracy'],
                'official_macro_f1': ov['macro_f1'],
                'handmade_correct': correct[1], 'handmade_accuracy': hh['accuracy'],
                'handmade_macro_f1': hh['macro_f1'],
                'official_guardrail_pass': int(ov['accuracy'] >= OFFICIAL_GUARDRAIL),
            }
            rows.append(row)
            records[name] = {'row': row, 'official': ov, 'handmade': hh}
            for w in (41, 42, 43, 44):
                mask = writers == w
                s = scores(yhm[mask], ph[mask])
                writer_rows.append({'variant': name, 'writer': w,
                                    'accuracy': s['accuracy'], 'macro_f1': s['macro_f1']})
            for char in LABELS:
                s = hh['per_class'][char]
                class_rows.append({'variant': name, 'class': char,
                                   'support': s['support'], 'precision': s['precision'],
                                   'recall': s['recall'], 'f1': s['f1']})
            save_csv(OUT / 'candidate_results.csv', rows)
            save_csv(OUT / 'by_writer.csv', writer_rows)
            save_csv(OUT / 'per_class_handmade.csv', class_rows)
            save_cm(OUT / f'official_cm_{name}.csv', ov['confusion_matrix'])
            save_cm(OUT / f'handmade_cm_{name}.csv', hh['confusion_matrix'])
            save_predictions(name, ph, yhm, writers, local_indices)
            (OUT / 'metrics_partial.json').write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'{name:<14} Official={ov["accuracy"]:.6f} F1={ov["macro_f1"]:.6f} | '
                  f'Hand-made={hh["accuracy"]:.6f} F1={hh["macro_f1"]:.6f} | '
                  f'guardrail={"PASS" if row["official_guardrail_pass"] else "FAIL"}', flush=True)

    if control_pv is None:
        raise RuntimeError('Missing control predictions')
    for name, rec in records.items():
        if name == 'k3_distance':
            continue
        # Predictions can be reconstructed from saved nearest neighbor arrays here.
        k, weight = rec['row']['k'], rec['row']['weights']
        pv = predict_from_neighbors(iv, dv, labels, k, weight)
        ph = predict_from_neighbors(ih, dh, labels, k, weight)
        rec['paired_vs_exp014'] = {
            'official': paired_stats(yva, control_pv, pv),
            'handmade': paired_stats(yhm, control_ph, ph),
        }
    eligible = [r for r in rows if r['official_guardrail_pass']]
    if not eligible:
        raise RuntimeError('No eligible candidate')
    best = max(eligible, key=lambda r: (r['handmade_macro_f1'],
                                       r['handmade_accuracy'], r['official_accuracy']))
    summary = {
        'experiment': 'EXP-015', 'dataset_sha256': sha,
        'official_train_n': len(ytr), 'official_validation_n': len(yva),
        'handmade_development_n': len(yhm),
        'knn': {'metric': 'euclidean', 'k_values': list(K_VALUES), 'weights': list(WEIGHTS)},
        'preprocessing': 'CoM -> Deskew', 'pixel_weight': PIXEL_WEIGHT,
        'baseline_augmentation_fraction': BASE_FRACTION,
        'stroke_strength': STROKE_STRENGTH,
        'targeted_classes': [LABELS[c] for c in weak],
        'targeted_extra_n': TARGETED_EXTRA_N,
        'training_n': len(labels), 'feature_dimensions': features.shape[1],
        'neighbor_query_seconds': {'fit': fit_sec, 'official': val_sec, 'handmade': hm_sec},
        'candidates': rows, 'selected_development_candidate': best,
        'results': records,
        'environment': {'python': platform.python_version(),
                        'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'limitations': 'Hand-made development labels have already been used to choose '
                       'targeted classes and tune preprocessing. No independent Test used.',
    }
    (OUT / 'metrics.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nSELECTED DEVELOPMENT CANDIDATE:', best['variant'], flush=True)
    print(f'Official={best["official_accuracy"]:.6f}, Hand-made={best["handmade_accuracy"]:.6f}, '
          f'MacroF1={best["handmade_macro_f1"]:.6f}', flush=True)
    print('Saved:', OUT, flush=True)


if __name__ == '__main__':
    main()
