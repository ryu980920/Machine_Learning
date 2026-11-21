#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-007: KNN Normal, train-only size and stroke augmentation.

Use beside EXP004_KNN_Normal_CoM.py and EXP006_KNN_Normal_BB_Size_fixed.py.
Run from repository root:
    python 05_model_optimize/KNN/Normal/scripts/EXP007_KNN_Normal_Augmentation.py

Control: EXP-005 CoM + KNN(k=3, weights=distance, Euclidean).
Variants append exactly 20% of the official Training fold as augmented examples.
Hand-made is development evaluation only, never a source of augmentation.
Original Test and final held-out Test are not loaded.
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.ndimage import grey_dilation, grey_erosion, label, binary_fill_holes
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'
sys.path.insert(0, str(ROOT / '03_experiments' / 'model_optimization'))
sys.path.insert(0, str(SCRIPTS))

from common import LABELS, fixed_split, pipeline, scores
from EXP004_KNN_Normal_CoM import BatchedCoM, read_handmade, paired_stats
from EXP006_KNN_Normal_BB_Size_fixed import apply_size, geometry
from Automatic_Centering_Transformer import test_centering

OUTPUT = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'EXP007_augmentation'
SEED = 20261121
AUG_FRACTION = 0.20
TARGET_LONG_SIDES = (16, 18, 22, 24)
STROKE_STRENGTH = 0.28
THRESHOLD_TOPOLOGY = 128
OFFICIAL_GUARDRAIL = 0.9569791666666667  # same as EXP-005/006
EXPECTED_OFFICIAL_CORRECT = 18529  # 0.96505208 * 19200
EXPECTED_HANDMADE_CORRECT = 2017  # 0.787890625 * 2560

# 4-connectivity cross, matching cv2.getStructuringElement(MORPH_CROSS, (3,3))
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
STRUCTURE8 = np.ones((3, 3), dtype=bool)


def save_csv(path, rows):
    if not rows:
        raise ValueError(f'Empty CSV: {path}')
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_cm(path, matrix):
    save_csv(path, [
        {'true/pred': LABELS[i], **{LABELS[j]: int(matrix[i][j]) for j in range(16)}}
        for i in range(16)
    ])


def sample_balanced(y, fraction=AUG_FRACTION, seed=SEED):
    """Class-stratified training indices; no validation/Hand-made sampling."""
    rng = np.random.default_rng(seed)
    groups = []
    for c in range(16):
        ids = np.flatnonzero(y == c)
        n = int(round(len(ids) * fraction))
        if n <= 0 or n >= len(ids):
            raise ValueError(f'Invalid augmentation count for class {c}')
        groups.append(np.sort(rng.choice(ids, size=n, replace=False)))
    selected = np.sort(np.concatenate(groups))
    assert len(np.unique(selected)) == len(selected)
    return selected


def topology_counts(image):
    """Number of connected foreground components and enclosed holes."""
    mask = np.asarray(image) >= THRESHOLD_TOPOLOGY
    ncomp = int(label(mask, structure=STRUCTURE8)[1])
    holes = binary_fill_holes(mask) & ~mask
    nholes = int(label(holes, structure=STRUCTURE8)[1])
    return ncomp, nholes


def stroke_jitter(batch, strength=STROKE_STRENGTH):
    """MLP-style gray morphology blend, cross kernel, topology guarded.

    Deterministic alternation: even images erode, odd images dilate.
    Unlike random pixel noise, this directly tests stroke thickness.
    """
    src = np.asarray(batch, dtype=np.float32).reshape(-1, 28, 28)
    result = np.empty_like(src)
    accepted = rejected = 0
    for i, img in enumerate(src):
        target = (grey_erosion(img, footprint=CROSS, mode='constant', cval=0)
                  if i % 2 == 0 else
                  grey_dilation(img, footprint=CROSS, mode='constant', cval=0))
        proposed = np.clip((1 - strength) * img + strength * target, 0, 255)
        if topology_counts(proposed) == topology_counts(img):
            result[i] = proposed
            accepted += 1
        else:
            result[i] = img
            rejected += 1
    return result.reshape(-1, 784), {'changed': accepted, 'topology_rejected': rejected}


def size_jitter(x, selected, seed=SEED):
    """Balanced across sizes within each class, with no fixed global target."""
    chosen = np.asarray(x[selected], dtype=np.float32)
    out = np.empty_like(chosen)
    rng = np.random.default_rng(seed + 17)
    targets = np.tile(np.asarray(TARGET_LONG_SIDES, dtype=np.int32),
                      (len(selected) + len(TARGET_LONG_SIDES)-1) // len(TARGET_LONG_SIDES))[:len(selected)]
    rng.shuffle(targets)
    for t in TARGET_LONG_SIDES:
        mask = targets == t
        if np.any(mask):
            out[mask], _ = apply_size(chosen[mask], int(t))
    return out, {str(t): int(np.sum(targets == t)) for t in TARGET_LONG_SIDES}


def test_augmentation():
    test_centering()
    x = np.zeros((2, 28, 28), dtype=np.float32)
    x[0, 7:21, 10:18] = 255
    x[1, 6:22, 8:20] = 255
    test_scaled, _ = apply_size(x, 18)
    assert test_scaled.shape == (2, 784)
    assert test_scaled.min() >= 0 and test_scaled.max() <= 255
    altered, record = stroke_jitter(x, strength=STROKE_STRENGTH)
    assert altered.shape == (2, 784)
    assert record['changed'] + record['topology_rejected'] == 2
    assert np.all(np.isfinite(altered)) and 0 <= altered.min() <= altered.max() <= 255
    assert len(geometry(x)[0]) == 2
    print('AUGMENTATION SELF-TEST PASSED: size, stroke topology, range', flush=True)


def knn_model():
    return pipeline(KNeighborsClassifier(n_neighbors=3, weights='distance',
                                         metric='minkowski', p=2), True)


def main():
    test_augmentation()
    xtr, ytr, xva, yva, dataset_sha = fixed_split()
    xhm, yhm, writers, indices = read_handmade()
    if (len(ytr), len(yva), len(yhm)) != (76800, 19200, 2560):
        raise RuntimeError('Dataset sizes changed; refusing to compare')
    print(f'Official train/val={len(ytr)}/{len(yva)}; Hand-made={len(yhm)}', flush=True)

    com = BatchedCoM()
    xtr_com = com.transform(xtr)
    xva_com = com.transform(xva)
    xhm_com = com.transform(xhm)

    selected = sample_balanced(ytr)
    print(f'Augmented samples: {len(selected)} extra (original official Training remains intact)', flush=True)
    size_aug, size_counts = size_jitter(xtr, selected)
    stroke_aug, stroke_stats = stroke_jitter(xtr[selected])
    both_aug, both_stats = stroke_jitter(size_aug)
    augmentations = {
        'control_com': None,
        'size_only': size_aug,
        'stroke_only': stroke_aug,
        'size_plus_stroke': both_aug,
    }
    # All augmented examples, and all evaluation samples, receive the same CoM rule.
    rows = []
    writer_rows = []
    class_rows = []
    predict_rows = []
    records = {}
    predictions = {}
    baseline_official = baseline_handmade = None

    for variant, aug in augmentations.items():
        print(f'\n[{variant}] Preparing official Training examples...', flush=True)
        if aug is None:
            train = xtr_com
            target = ytr
        else:
            aug_com = com.transform(aug)
            train = np.concatenate([xtr_com, aug_com], axis=0)
            target = np.concatenate([ytr, ytr[selected]])
        print(f'  Fit rows={len(target)}; augmented={len(target)-len(ytr)}', flush=True)
        model = knn_model()
        start = time.perf_counter()
        model.fit(train, target)
        fit_time = time.perf_counter() - start
        start = time.perf_counter()
        off_pred = np.asarray(model.predict(xva_com), dtype=np.int64)
        off_predict_seconds = time.perf_counter() - start
        start = time.perf_counter()
        hm_pred = np.asarray(model.predict(xhm_com), dtype=np.int64)
        hm_predict_seconds = time.perf_counter() - start
        off = scores(yva, off_pred)
        hm = scores(yhm, hm_pred)
        if variant == 'control_com':
            if (int(np.count_nonzero(off_pred == yva)) != EXPECTED_OFFICIAL_CORRECT or
                    int(np.count_nonzero(hm_pred == yhm)) != EXPECTED_HANDMADE_CORRECT):
                raise RuntimeError('EXP-005 baseline mismatch: stop and check dependencies/data')
            baseline_official, baseline_handmade = off_pred.copy(), hm_pred.copy()
            print('  EXP-005 control reproduced exactly (correct counts)', flush=True)
        records[variant] = {'official': off, 'handmade': hm,
                            'fit_seconds': fit_time, 'official_predict_seconds': off_predict_seconds,
                            'handmade_predict_seconds': hm_predict_seconds}
        predictions[variant] = {'official': off_pred, 'handmade': hm_pred}
        guard = off['accuracy'] >= OFFICIAL_GUARDRAIL
        rows.append({
            'variant': variant, 'training_n': len(target), 'augmented_n': len(target) - len(ytr),
            'official_accuracy': off['accuracy'], 'official_macro_f1': off['macro_f1'],
            'handmade_accuracy': hm['accuracy'], 'handmade_macro_f1': hm['macro_f1'],
            'official_delta_pp_vs_com': 100 * (off['accuracy'] - EXPECTED_OFFICIAL_CORRECT / len(yva)),
            'handmade_delta_pp_vs_com': 100 * (hm['accuracy'] - EXPECTED_HANDMADE_CORRECT / len(yhm)),
            'official_guardrail_pass': int(guard),
            'fit_seconds': fit_time, 'official_predict_seconds': off_predict_seconds,
            'handmade_predict_seconds': hm_predict_seconds,
        })
        for w in (41, 42, 43, 44):
            mask = writers == w
            ss = scores(yhm[mask], hm_pred[mask])
            writer_rows.append({'variant': variant, 'writer': w, 'n': int(mask.sum()),
                                'accuracy': ss['accuracy'], 'macro_f1': ss['macro_f1']})
        for klass in LABELS:
            item = hm['per_class'][klass]
            class_rows.append({'variant': variant, 'class': klass, 'n': item['support'],
                               'precision': item['precision'], 'recall': item['recall'], 'f1': item['f1']})
        print(f"  Official={off['accuracy']:.6f} F1={off['macro_f1']:.6f} | "
              f"Hand-made={hm['accuracy']:.6f} F1={hm['macro_f1']:.6f} | "
              f"guardrail={'PASS' if guard else 'FAIL'}", flush=True)
        del model, train, target

    valid = [r for r in rows if r['official_guardrail_pass']]
    selected_candidate = max(valid, key=lambda r: (r['handmade_macro_f1'], r['handmade_accuracy'],
                                                   r['official_macro_f1'])) if valid else None

    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_csv(OUTPUT / 'candidate_results.csv', rows)
    save_csv(OUTPUT / 'by_writer.csv', writer_rows)
    save_csv(OUTPUT / 'per_class_handmade.csv', class_rows)
    for variant in augmentations:
        save_cm(OUTPUT / f'official_cm_{variant}.csv', records[variant]['official']['confusion_matrix'])
        save_cm(OUTPUT / f'handmade_cm_{variant}.csv', records[variant]['handmade']['confusion_matrix'])
        pred = predictions[variant]['handmade']
        for i in range(len(yhm)):
            predict_rows.append({
                'variant': variant, 'writer': int(writers[i]), 'index': int(indices[i]),
                'true': LABELS[int(yhm[i])], 'predicted': LABELS[int(pred[i])],
                'correct': int(pred[i] == yhm[i])
            })
    save_csv(OUTPUT / 'handmade_predictions.csv', predict_rows)
    transitions = {
        variant: {
            'official': paired_stats(yva, baseline_official, predictions[variant]['official']),
            'handmade': paired_stats(yhm, baseline_handmade, predictions[variant]['handmade']),
        } for variant in ('size_only', 'stroke_only', 'size_plus_stroke')
    }
    summary = {
        'experiment': 'EXP-007 KNN Normal train-only size/stroke augmentation',
        'reference': 'EXP-005 CoM k=3 distance',
        'official_dataset_sha256': dataset_sha,
        'handmade_n': len(yhm), 'official_train_n': len(ytr), 'official_validation_n': len(yva),
        'random_seed': SEED, 'augmentation_fraction': AUG_FRACTION,
        'augmentation_design': 'Append 20% of class-stratified official train rows; no validation augmentation',
        'target_long_sides': list(TARGET_LONG_SIDES), 'target_size_counts': size_counts,
        'stroke_strength': STROKE_STRENGTH, 'stroke_method': 'cross-kernel gray erosion/dilation blend',
        'stroke_topology': 'preserve connected components and hole counts at threshold 128',
        'stroke_only_stats': stroke_stats, 'size_plus_stroke_stats': both_stats,
        'knn': {'n_neighbors': 3, 'weights': 'distance', 'metric': 'minkowski', 'p': 2},
        'centering': 'MLP Normal CoM, threshold=32, integer shift zero-fill',
        'guardrail_min_official_accuracy': OFFICIAL_GUARDRAIL,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
        'candidates': rows, 'paired_vs_com': transitions,
        'selected_candidate_development_only': selected_candidate,
        'note': 'Hand-made is development-only; Original Test and final Test untouched.',
    }
    (OUTPUT / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nPAIRED vs CoM:')
    for name, val in transitions.items():
        print(name, val, flush=True)
    print('\nSELECTED DEVELOPMENT CANDIDATE:',
          selected_candidate['variant'] if selected_candidate else 'NONE', flush=True)
    if selected_candidate:
        print(f"Official={selected_candidate['official_accuracy']:.6f}, "
              f"Hand-made={selected_candidate['handmade_accuracy']:.6f}, "
              f"MacroF1={selected_candidate['handmade_macro_f1']:.6f}", flush=True)
    print('Saved:', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
