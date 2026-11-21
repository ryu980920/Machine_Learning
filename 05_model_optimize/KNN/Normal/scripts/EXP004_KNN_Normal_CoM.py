#!/usr/bin/env python3
"""EXP-004: KNN Normal CoM centering vs unchanged baseline.

Uses identical CoM centering as the existing MLP Normal EXP유004 code:
threshold 32/255, weighted mass, target (13.5,13.5), integer shift, zero-fill.
No original Test data, final Test data, or Hand-made images are used for fitting.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / '03_experiments' / 'model_optimization'))
sys.path.insert(0, str(ROOT / '05_model_optimize' / 'MLP' / 'Normal' / 'scripts'))

from common import LABELS, fixed_split, pipeline, scores
from knn import build_baseline
from Automatic_Centering_Transformer import center_batch, test_centering

OUT = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'EXP004_com_centering'
HANDMADE = ROOT / '02_data' / 'Hand_made'
EXPECTED_HANDMADE_SHA = {
    41: '0a829172a72787c770139c73de34cc8b751181829ff6fe08ec260247b9be5a2f',
    42: 'f364fb453d05b4fe2a705c2b8a80683b404608c7c257d664e3ec7ac86932d4c5',
    43: '10db20997c727c1bc1f1299f8e89626fadd3ec9618468d6759c86b0d2ddc4a66',
    44: '0e266a36e7b29edfcca0514836b5c383d13939948b4e335d70f61bb40bdc084d',
}

class BatchedCoM(BaseEstimator, TransformerMixin):
    """Memory-bounded wrapper of the repo's existing MLP CoM implementation."""
    def __init__(self, batch_size=2048):
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        a = np.asarray(X, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 784:
            raise ValueError(f'Expected (N,784) images, got {a.shape}')
        out = np.empty_like(a)
        for start in range(0, len(a), self.batch_size):
            stop = min(start + self.batch_size, len(a))
            out[start:stop] = center_batch(a[start:stop], 'mass')
        return out


def write_rows(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_cm(path, cm):
    write_rows(path, [
        {'true/pred': LABELS[i], **{LABELS[j]: int(cm[i][j]) for j in range(16)}}
        for i in range(16)
    ])


def read_handmade():
    arrays, labs, writers, local_indices = [], [], [], []
    for w in (41, 42, 43, 44):
        path = HANDMADE / f'ml26_{w}.npz'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != EXPECTED_HANDMADE_SHA[w]:
            raise ValueError(f'Unexpected Hand-made SHA256 for writer {w}: {digest}')
        with np.load(path, allow_pickle=False) as d:
            if set(d.files) != {'img', 'label', 'label_id'}:
                raise ValueError(f'Unexpected Hand-made keys {d.files}')
            img = d['img']
            y = d['label_id'].astype(np.int64)
            label = d['label']
        if img.shape != (640, 28, 28) or img.dtype != np.uint8 or y.shape != (640,):
            raise ValueError(f'Unexpected images/labels for writer {w}')
        if not np.array_equal(y, np.array([LABELS.index(str(c)) for c in label], dtype=np.int64)):
            raise ValueError(f'Unexpected label mapping for writer {w}')
        if not np.array_equal(np.bincount(y, minlength=16), np.full(16, 40)):
            raise ValueError(f'Unexpected label counts for writer {w}')
        arrays.append(img.reshape(640, 784).astype(np.float32))
        labs.append(y)
        writers.extend([w] * 640)
        local_indices.extend(range(640))
    return np.concatenate(arrays), np.concatenate(labs), np.array(writers), np.array(local_indices)


def paired_stats(y, original, centered):
    old_correct = original == y
    new_correct = centered == y
    return {
        'corrected': int(np.sum(~old_correct & new_correct)),
        'new_errors': int(np.sum(old_correct & ~new_correct)),
        'unchanged_correct': int(np.sum(old_correct & new_correct)),
        'unchanged_wrong': int(np.sum(~old_correct & ~new_correct)),
    }


def main():
    test_centering()
    xtr, ytr, xva, yva, sha = fixed_split()
    xhm, yhm, writers, local_indices = read_handmade()
    print(f'Official Training/Validation: {len(ytr)}/{len(yva)}; Hand-made: {len(yhm)}', flush=True)
    raw_model, scaled = build_baseline()
    if not scaled:
        raise ValueError('KNN baseline is expected to use PixelScaler255')
    raw_pipe = pipeline(raw_model, scaled)
    candidate_model, _ = build_baseline()
    candidate_scaled_pipe = pipeline(candidate_model, True)
    centered_pipe = Pipeline([('center_of_mass', BatchedCoM()), *candidate_scaled_pipe.steps])

    print('Fitting unchanged KNN baseline...', flush=True)
    t = time.perf_counter()
    raw_pipe.fit(xtr, ytr)
    raw_fit = time.perf_counter() - t
    pred_raw_va = raw_pipe.predict(xva)
    pred_raw_hm = raw_pipe.predict(xhm)
    baseline_va = scores(yva, pred_raw_va)
    baseline_hm = scores(yhm, pred_raw_hm)
    print(f"Baseline Official={baseline_va['accuracy']:.6f}, Hand-made={baseline_hm['accuracy']:.6f}", flush=True)
    if (not np.isclose(baseline_va['accuracy'], 0.9629166666666666, atol=1e-12)
        or not np.isclose(baseline_hm['accuracy'], 0.58828125, atol=1e-12)
        or not np.isclose(baseline_hm['macro_f1'], 0.5664200025438473, atol=1e-12)):
        raise ValueError('EXP-001/002 baseline results were not reproduced; refusing comparison')

    print('Fitting KNN with CoM on official Training only...', flush=True)
    t = time.perf_counter()
    centered_pipe.fit(xtr, ytr)
    com_fit = time.perf_counter() - t
    pred_com_va = centered_pipe.predict(xva)
    pred_com_hm = centered_pipe.predict(xhm)
    com_va = scores(yva, pred_com_va)
    com_hm = scores(yhm, pred_com_hm)
    print(f"CoM Official={com_va['accuracy']:.6f}, MacroF1={com_va['macro_f1']:.6f}", flush=True)
    print(f"CoM Hand-made={com_hm['accuracy']:.6f}, MacroF1={com_hm['macro_f1']:.6f}", flush=True)
    print(f"Difference Official={100*(com_va['accuracy']-baseline_va['accuracy']):+.3f} percentage points", flush=True)
    print(f"Difference Hand-made={100*(com_hm['accuracy']-baseline_hm['accuracy']):+.3f} percentage points", flush=True)

    OUTPUT = OUT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_cm(OUTPUT / 'official_com_confusion_matrix.csv', com_va['confusion_matrix'])
    write_cm(OUTPUT / 'handmade_com_confusion_matrix.csv', com_hm['confusion_matrix'])

    summary_rows = []
    for name, base, com in [('official_validation', baseline_va, com_va), ('handmade_dev', baseline_hm, com_hm)]:
        summary_rows.append({'evaluation': name, 'n': len(yva) if name=='official_validation' else len(yhm),
                             'baseline_accuracy': base['accuracy'], 'com_accuracy': com['accuracy'],
                             'delta_accuracy_pp': 100*(com['accuracy']-base['accuracy']),
                             'baseline_macro_f1': base['macro_f1'], 'com_macro_f1': com['macro_f1'],
                             'delta_macro_f1': com['macro_f1']-base['macro_f1']})
    write_rows(OUTPUT / 'comparison.csv', summary_rows)

    writer_rows = []
    for w in (41,42,43,44):
        mask = writers == w
        old, new = scores(yhm[mask], pred_raw_hm[mask]), scores(yhm[mask], pred_com_hm[mask])
        writer_rows.append({'writer': w, 'n': int(mask.sum()), 'baseline_accuracy': old['accuracy'],
                            'com_accuracy': new['accuracy'], 'delta_pp': 100*(new['accuracy']-old['accuracy']),
                            'baseline_macro_f1': old['macro_f1'], 'com_macro_f1': new['macro_f1'],
                            **paired_stats(yhm[mask], pred_raw_hm[mask], pred_com_hm[mask])})
        print(f"Writer {w}: {old['accuracy']:.4f} -> {new['accuracy']:.4f}", flush=True)
    write_rows(OUTPUT / 'by_writer.csv', writer_rows)

    pc_rows = []
    for i, c in enumerate(LABELS):
        b, n = baseline_hm['per_class'][c], com_hm['per_class'][c]
        pc_rows.append({'class': c, 'n': b['support'], 'baseline_recall': b['recall'],
                        'com_recall': n['recall'], 'delta_recall_pp': 100*(n['recall']-b['recall']),
                        'baseline_f1': b['f1'], 'com_f1': n['f1']})
    write_rows(OUTPUT / 'per_class_handmade.csv', pc_rows)

    pred_rows = []
    for w, idx, y, a, b in zip(writers, local_indices, yhm, pred_raw_hm, pred_com_hm):
        pred_rows.append({'writer': int(w), 'index': int(idx), 'true': LABELS[int(y)],
                          'baseline_prediction': LABELS[int(a)], 'com_prediction': LABELS[int(b)],
                          'baseline_correct': int(a==y), 'com_correct': int(b==y)})
    write_rows(OUTPUT / 'handmade_predictions_comparison.csv', pred_rows)

    result = {
        'experiment':'EXP-004 KNN Normal CoM centering',
        'source_centering':'05_model_optimize/MLP/Normal/scripts/Automatic_Centering_Transformer.py',
        'center_method':'mass', 'threshold_raw':32, 'target_xy':[13.5,13.5],
        'shift':'np.rint integer translation, zero fill; no wrap/no interpolation',
        'training_n':len(ytr),'validation_n':len(yva),'handmade_n':len(yhm),
        'official_dataset_sha256':sha, 'handmade_sha256':{str(k):v for k,v in EXPECTED_HANDMADE_SHA.items()},
        'model_hyperparameters': {str(k):repr(v) for k,v in raw_model.get_params().items()},
        'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,
        'fit_seconds':{'baseline':raw_fit, 'com':com_fit},
        'official_validation':{'baseline':baseline_va,'com':com_va,'paired':paired_stats(yva,pred_raw_va,pred_com_va)},
        'handmade_dev':{'baseline':baseline_hm,'com':com_hm,'paired':paired_stats(yhm,pred_raw_hm,pred_com_hm)},
        'by_writer':writer_rows,
        'note':'CoM is fixed from MLP project parameters; no Hand-made labels used to fit/tune transforms. Original Test and final Test unused.'
    }
    (OUTPUT / 'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Paired official:', result['official_validation']['paired'], flush=True)
    print('Paired Hand-made:', result['handmade_dev']['paired'], flush=True)
    print('Saved:', OUTPUT, flush=True)

if __name__ == '__main__':
    main()
