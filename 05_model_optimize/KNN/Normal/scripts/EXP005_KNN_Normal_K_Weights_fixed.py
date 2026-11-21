#!/usr/bin/env python3
"""EXP-005: CoM KNN Normal neighbor count and vote weights comparison.

Expected repository files:
- 03_experiments/model_optimization/common.py
- 05_model_optimize/MLP/Normal/scripts/Automatic_Centering_Transformer.py
- 05_model_optimize/KNN/Normal/scripts/EXP004_KNN_Normal_CoM.py

Train on the EXACT official training fold. Evaluate official Validation and
four dev Hand-made writers, never use official Original Test / final Test.
The 14 candidates reuse one 15-NN query. This is a development experiment,
not an independent final generalization estimate.
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / '03_experiments' / 'model_optimization'))
sys.path.insert(0, str(ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'))

from common import LABELS, fixed_split, pipeline, scores
from EXP004_KNN_Normal_CoM import BatchedCoM, read_handmade, test_centering

OUT = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'EXP005_k_weights'
K_VALUES = (1, 3, 5, 7, 9, 11, 15)
WEIGHTS = ('uniform', 'distance')
MIN_OFFICIAL_ACC = 0.9569791666666667  # CoM baseline 0.961979 - 0.005 (0.5%p)
BASE_OFFICIAL_ACC = 0.9619791666666667
BASE_OFFICIAL_F1 = 0.9619584413989763
BASE_HANDMADE_ACC = 0.778515625
BASE_HANDMADE_F1 = 0.7764856058056413


def csv_write(path: Path, rows: list[dict], columns=None):
    columns = columns or list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def predict_from_neighbors(indices, distances, training_labels, k: int, weights: str):
    labels = training_labels[indices[:, :k]]
    d = distances[:, :k]
    if weights == 'uniform':
        w = np.ones(d.shape, dtype=np.float64)
    else:
        zeros = d == 0
        w = np.zeros(d.shape, dtype=np.float64)
        np.divide(1.0, d, out=w, where=~zeros)
        w = np.where(zeros.any(axis=1, keepdims=True), zeros.astype(np.float64), w)
    votes = np.empty((len(labels), len(LABELS)), dtype=np.float64)
    for c in range(len(LABELS)):
        votes[:, c] = np.sum((labels == c) * w, axis=1)
    return np.argmax(votes, axis=1).astype(np.int64)


def transform_before_knn(fitted_model, X):
    """Apply already fitted preprocessing steps without slicing a Pipeline.

    A Pipeline slice ending in a stateless CoM/scaler can fail the
    scikit-learn fitted-state check despite the full pipeline being fitted.
    """
    transformed = X
    for name, transformer in fitted_model.steps[:-1]:
        transformed = transformer.transform(transformed)
    return transformed


def paired_counts(y, old, new):
    a, b = old == y, new == y
    return {'corrected': int(np.sum(~a & b)),
            'new_errors': int(np.sum(a & ~b)),
            'unchanged_correct': int(np.sum(a & b)),
            'unchanged_wrong': int(np.sum(~a & ~b))}


def main():
    test_centering()
    xtr, ytr, xva, yva, dataset_sha = fixed_split()
    xhm, yhm, writer_ids, writer_indices = read_handmade()
    print(f'Official Training/Validation: {len(ytr)}/{len(yva)}; Hand-made: {len(yhm)}', flush=True)

    knn = KNeighborsClassifier(n_neighbors=max(K_VALUES), weights='uniform', metric='minkowski', p=2)
    model = Pipeline([('center_of_mass', BatchedCoM()), *pipeline(knn, True).steps])
    start = time.perf_counter()
    print('Fitting KNN with unchanged CoM on official Training only...', flush=True)
    model.fit(xtr, ytr)
    fit_sec = time.perf_counter() - start
    model_knn = model.named_steps['model']

    print('Computing 15 nearest neighbors on official Validation...', flush=True)
    start = time.perf_counter()
    dva, iva = model_knn.kneighbors(transform_before_knn(model, xva), n_neighbors=max(K_VALUES), return_distance=True)
    query_va_sec = time.perf_counter() - start
    print('Computing 15 nearest neighbors on Hand-made...', flush=True)
    start = time.perf_counter()
    dhm, ihm = model_knn.kneighbors(transform_before_knn(model, xhm), n_neighbors=max(K_VALUES), return_distance=True)
    query_hm_sec = time.perf_counter() - start

    predictions = {}
    rows, writer_rows = [], []
    for k in K_VALUES:
        for weights in WEIGHTS:
            key = f'k{k}_{weights}'
            pred_va = predict_from_neighbors(iva, dva, ytr, k, weights)
            pred_hm = predict_from_neighbors(ihm, dhm, ytr, k, weights)
            eva, ehm = scores(yva, pred_va), scores(yhm, pred_hm)
            predictions[key] = (pred_va, pred_hm)
            row = {'candidate': key, 'k': k, 'weights': weights,
                   'official_accuracy': eva['accuracy'], 'official_macro_f1': eva['macro_f1'],
                   'handmade_accuracy': ehm['accuracy'], 'handmade_macro_f1': ehm['macro_f1'],
                   'official_delta_pp_vs_com_k5': 100 * (eva['accuracy'] - BASE_OFFICIAL_ACC),
                   'handmade_delta_pp_vs_com_k5': 100 * (ehm['accuracy'] - BASE_HANDMADE_ACC),
                   'official_guardrail_pass': int(eva['accuracy'] >= MIN_OFFICIAL_ACC)}
            rows.append(row)
            for w in (41, 42, 43, 44):
                mask = writer_ids == w
                e = scores(yhm[mask], pred_hm[mask])
                writer_rows.append({'candidate': key, 'writer': w, 'n': int(mask.sum()),
                                    'accuracy': e['accuracy'], 'macro_f1': e['macro_f1']})
            print(f"{key:13s} Official={eva['accuracy']:.6f} F1={eva['macro_f1']:.6f} | "
                  f"Hand-made={ehm['accuracy']:.6f} F1={ehm['macro_f1']:.6f} | "
                  f"guardrail={'PASS' if row['official_guardrail_pass'] else 'FAIL'}", flush=True)

    ref_va, ref_hm = predictions['k5_uniform']
    ref_va_s, ref_hm_s = scores(yva, ref_va), scores(yhm, ref_hm)
    if not (np.isclose(ref_va_s['accuracy'], BASE_OFFICIAL_ACC, atol=1e-12)
            and np.isclose(ref_va_s['macro_f1'], BASE_OFFICIAL_F1, atol=1e-12)
            and np.isclose(ref_hm_s['accuracy'], BASE_HANDMADE_ACC, atol=1e-12)
            and np.isclose(ref_hm_s['macro_f1'], BASE_HANDMADE_F1, atol=1e-12)):
        raise RuntimeError('CoM k=5 uniform reference differs from EXP-004. '
                           'Do not compare or record candidate results before investigating '
                           'version/tie handling and source data.')
    print('EXP-004 reference metrics successfully reproduced.', flush=True)

    eligible = [r for r in rows if r['official_guardrail_pass']]
    if not eligible:
        raise RuntimeError('No candidate satisfies official Validation guardrail.')
    chosen = max(eligible, key=lambda r: (r['handmade_macro_f1'], r['handmade_accuracy'],
                                          r['official_macro_f1'], -r['k']))
    best_key = chosen['candidate']
    _, best_hm = predictions[best_key]
    best_hm_scores = scores(yhm, best_hm)
    changes = paired_counts(yhm, ref_hm, best_hm)

    OUT.mkdir(parents=True, exist_ok=True)
    csv_write(OUT / 'candidate_results.csv', rows)
    csv_write(OUT / 'by_writer.csv', writer_rows)
    csv_write(OUT / 'handmade_best_predictions.csv', [
        {'writer': int(w), 'index': int(idx), 'true': LABELS[int(y)],
         'com_k5_prediction': LABELS[int(a)], 'candidate_prediction': LABELS[int(b)],
         'com_k5_correct': int(a == y), 'candidate_correct': int(b == y)}
        for w, idx, y, a, b in zip(writer_ids, writer_indices, yhm, ref_hm, best_hm)
    ])
    cm = best_hm_scores['confusion_matrix']
    csv_write(OUT / 'handmade_best_confusion_matrix.csv', [
        {'true/pred': LABELS[i], **{LABELS[j]: cm[i][j] for j in range(16)}}
        for i in range(16)
    ])
    summary = {'experiment': 'EXP-005 KNN Normal CoM neighbor count and vote weights',
               'official_dataset_sha256': dataset_sha,
               'train_n': len(ytr), 'validation_n': len(yva), 'handmade_n': len(yhm),
               'center_method': 'MLP AutomaticCentering mass, threshold=32, target=(13.5,13.5), integer shift',
               'k_values': list(K_VALUES), 'weights_values': list(WEIGHTS),
               'distance': 'Euclidean (Minkowski p=2)',
               'guardrail': f'official Validation accuracy >= {MIN_OFFICIAL_ACC:.9f}',
               'selection': 'among guardrail pass: highest hand-made macro F1; ties by accuracy, official macro F1, smaller k',
               'note': 'Development selection on Hand-made. These scores are NOT an independent final Test.',
               'reference_com_k5': {'official': ref_va_s, 'handmade': ref_hm_s},
               'candidates': rows, 'chosen': chosen,
               'chosen_vs_reference_handmade_paired': changes,
               'fit_seconds': fit_sec, 'query_official_seconds': query_va_sec,
               'query_handmade_seconds': query_hm_sec,
               'python': platform.python_version(), 'numpy': np.__version__,
               'sklearn': sklearn.__version__}
    (OUT / 'metrics.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nSELECTED CANDIDATE (development only): {best_key}', flush=True)
    print(f"Official={chosen['official_accuracy']:.6f}, Hand-made={chosen['handmade_accuracy']:.6f}, "
          f"MacroF1={chosen['handmade_macro_f1']:.6f}", flush=True)
    print('Paired Hand-made vs CoM k5:', changes, flush=True)
    print('Saved:', OUT, flush=True)


if __name__ == '__main__':
    main()
