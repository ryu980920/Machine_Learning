#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-005 SVC Normal: CoM+Deskew with pixel, HOG and pixel:HOG (20:80).

Run from the Machine_Learning repository root:
  python 05_model_optimize/SVC/Normal/scripts/EXP_005_SVC_Normal_HOG_SafePilot.py

Safe development pilot: 8,000 official train / 3,200 official validation,
all 2,560 Hand-made for development, and no official Test data.
Reuses EXP-004 preprocessing and EXP-012 HOG implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(ROOT / '05_model_optimize' / 'SVC' / 'Normal' / 'scripts'),
    str(ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'),
    str(ROOT / '03_experiments' / 'model_optimization'),
]
from EXP_001_SVC_Normal_Baseline import WRITERS, baseline, load_writer, report
from EXP004_KNN_Normal_CoM import BatchedCoM
from EXP011_KNN_Normal_Deskew import deskew, test_deskew
from EXP012_KNN_Normal_HOG import hog_features, hybrid_features, test_hog
from common import fixed_split, pipeline

RESULTS = ROOT / '05_model_optimize' / 'SVC' / 'Normal' / 'results'


def say(*parts):
    print(*parts, flush=True)


def select_stratified(x, y, n, seed=42):
    if n > len(y):
        raise ValueError(f'Requested {n} rows but only {len(y)} exist')
    if n == len(y):
        return x, y
    ix, _ = train_test_split(np.arange(len(y)), train_size=n,
                             stratify=y, random_state=seed)
    return x[ix], y[ix]


def predict_batches(model, x, name, batch_size=256):
    preds = []
    start = time.perf_counter()
    for pos in range(0, len(x), batch_size):
        end = min(pos + batch_size, len(x))
        preds.append(np.asarray(model.predict(x[pos:end]), dtype=np.int64))
        if pos == 0 or end == len(x) or (pos // batch_size + 1) % 8 == 0:
            say(f'  {name}: {end}/{len(x)} predicted')
    return np.concatenate(preds), round(time.perf_counter() - start, 2)


def run(args):
    svc, scaling, source = baseline()
    if not scaling:
        raise RuntimeError('Original SVC baseline unexpectedly has scaling=False')
    say('SVC baseline:', source)
    say('SVC parameters:', svc.get_params())
    say('EXP-005 SAFE PILOT: pixel vs HOG vs pixel:HOG 20:80; official Test NOT loaded')
    test_deskew()
    test_hog()

    t = time.perf_counter()
    xtr, ytr, xval, yval, sha = fixed_split()
    say(f'Official split loaded in {time.perf_counter()-t:.2f}s: {len(ytr)} / {len(yval)}')
    xtr, ytr = select_stratified(xtr, ytr, args.train_size)
    xval, yval = select_stratified(xval, yval, args.val_size)
    writers = {w: load_writer(w) for w in WRITERS}
    say(f'Pilot train={len(ytr)} val={len(yval)} handmade={sum(len(v[1]) for v in writers.values())}')

    t = time.perf_counter()
    com = BatchedCoM()
    centered_train = com.transform(xtr)
    centered_val = com.transform(xval)
    centered_writers = {w: (com.transform(writers[w][0]), writers[w][1]) for w in WRITERS}
    say(f'CoM preprocessing completed in {time.perf_counter()-t:.2f}s')
    t = time.perf_counter()
    pixel_train, _ = deskew(centered_train)
    pixel_val, _ = deskew(centered_val)
    pixel_writers = {w: (deskew(centered_writers[w][0])[0], centered_writers[w][1])
                     for w in WRITERS}
    say(f'Deskew preprocessing completed in {time.perf_counter()-t:.2f}s')

    t = time.perf_counter()
    hog_train = hog_features(pixel_train)
    hog_val = hog_features(pixel_val)
    hog_writers = {w: (hog_features(pixel_writers[w][0]), pixel_writers[w][1])
                   for w in WRITERS}
    say(f'HOG features extracted in {time.perf_counter()-t:.2f}s')
    t = time.perf_counter()
    mix_train = hybrid_features(pixel_train, hog_train, pixel_weight=0.20)
    mix_val = hybrid_features(pixel_val, hog_val, pixel_weight=0.20)
    mix_writers = {w: (hybrid_features(pixel_writers[w][0], hog_writers[w][0], pixel_weight=0.20),
                       pixel_writers[w][1]) for w in WRITERS}
    say(f'Pixel:HOG 20:80 features prepared in {time.perf_counter()-t:.2f}s')

    # The project pipeline applies 0..255 -> 0..1 scaling. This is only
    # appropriate for raw pixel features. HOG and hybrid are already normalized,
    # so we pass them directly to the same original SVC estimator.
    candidates = [
        ('pixel_com_deskew', pixel_train, pixel_val, pixel_writers, True),
        ('hog_com_deskew', hog_train, hog_val, hog_writers, False),
        ('pixel20_hog80_com_deskew', mix_train, mix_val, mix_writers, False),
    ]
    output_dir = RESULTS / f'EXP_005_SVC_Normal_HOG_SAFE_PILOT_train{len(ytr)}'
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    summary_rows = []

    for name, train, val, hw, use_scaler in candidates:
        model = pipeline(clone(svc), True) if use_scaler else clone(svc)
        assert train.shape[1] in (784, 324, 1108)
        assert np.isfinite(train).all() and np.isfinite(val).all()
        say(f'\n[{name}] fit START; rows={len(ytr)} dims={train.shape[1]} scaled_by_project_pipeline={use_scaler}')
        t = time.perf_counter()
        model.fit(train, ytr)
        fit_s = round(time.perf_counter() - t, 2)
        say(f'[{name}] fit DONE {fit_s}s')
        official_pred, official_pred_s = predict_batches(model, val, f'{name} official', args.batch_size)
        official_result = report(yval, official_pred)
        say(f'[{name}] Official Validation accuracy={official_result["accuracy"]:.6f} '
            f'F1={official_result["macro_f1"]:.6f}; predict={official_pred_s}s')
        by_writer, y_all, p_all = [], [], []
        for w in WRITERS:
            wx, wy = hw[w]
            wp, wp_s = predict_batches(model, wx, f'{name} writer {w}', args.batch_size)
            scores = report(wy, wp)
            by_writer.append({'writer': w, 'accuracy': scores['accuracy'],
                              'macro_f1': scores['macro_f1'], 'n': len(wy),
                              'predict_seconds': wp_s})
            y_all.append(wy)
            p_all.append(wp)
            say(f'[{name}] writer {w} accuracy={scores["accuracy"]:.6f}')
        hand_result = report(np.concatenate(y_all), np.concatenate(p_all))
        say(f'[{name}] Hand-made development accuracy={hand_result["accuracy"]:.6f} '
            f'F1={hand_result["macro_f1"]:.6f}')
        records[name] = {
            'feature_dimensions': int(train.shape[1]),
            'scaling': 'project pixel/255 pipeline' if use_scaler else 'pre-normalized features; no pixel/255 scaler',
            'fit_seconds': fit_s, 'official_predict_seconds': official_pred_s,
            'official_validation': official_result, 'handmade_development': hand_result,
            'writers': by_writer,
        }
        summary_rows.append({
            'model': name, 'features': int(train.shape[1]), 'train_size': len(ytr),
            'fit_seconds': fit_s, 'official_accuracy': official_result['accuracy'],
            'official_macro_f1': official_result['macro_f1'],
            'handmade_accuracy': hand_result['accuracy'],
            'handmade_macro_f1': hand_result['macro_f1'],
        })
        (output_dir / 'metrics_partial.json').write_text(
            json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        del model

    control = records['pixel_com_deskew']
    # This pilot fixes the same stratified subsets as EXP-004 and should
    # reproduce the saved control accuracy for 8,000/3,200.
    if args.train_size == 8000 and args.val_size == 3200:
        expect = (0.959375, 0.76796875)
        found = (control['official_validation']['accuracy'],
                 control['handmade_development']['accuracy'])
        if not np.allclose(found, expect, atol=1e-9):
            raise RuntimeError(f'EXP-004 control not reproduced: found={found}, expected={expect}')
        say('EXP-004 pixel control EXACTLY reproduced')
    best = max(summary_rows, key=lambda r: (r['handmade_macro_f1'], r['official_accuracy']))
    result = {
        'experiment': 'EXP-005 SVC Normal HOG safe pilot',
        'train_size': len(ytr), 'validation_size': len(yval), 'official_test_used': False,
        'model_source': source, 'model_parameters': svc.get_params(),
        'official_dataset_sha256': sha,
        'results': records, 'selected_development_candidate': best['model'],
        'note': 'Subset pilot; repeated-use Hand-made development set is not an independent final test',
    }
    (output_dir / 'metrics.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    with (output_dir / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    say('\nEXP-005 SAFE PILOT SUMMARY')
    for r in summary_rows:
        say(f'  {r["model"]:<27} official={r["official_accuracy"]:.6f} '
            f'handmade={r["handmade_accuracy"]:.6f} F1={r["handmade_macro_f1"]:.6f}')
    say('SELECTED DEVELOPMENT CANDIDATE:', best['model'])
    say('Saved:', output_dir)
    say('Official Test was NOT loaded.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-size', type=int, default=8000, choices=(2000, 4000, 8000, 16000))
    parser.add_argument('--val-size', type=int, default=3200, choices=(800, 1600, 3200, 6400))
    parser.add_argument('--batch-size', type=int, default=256, choices=(64, 128, 256, 512))
    run(parser.parse_args())
