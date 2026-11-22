#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVC Normal EXP-003 safety pilot: fair baseline-versus-CoM comparison.

Run from repo root:
  python 05_model_optimize/SVC/Normal/scripts/EXP_003_SVC_Normal_CoM_SafePilot.py
  python 05_model_optimize/SVC/Normal/scripts/EXP_003_SVC_Normal_CoM_SafePilot.py --train-size 16000

This is a reduced-data development pilot, NOT a full 76,800-row experiment.
Never reads official Test. Does not overwrite EXP-001/002/003 result folders.
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
from common import fixed_split, pipeline

OUT_BASE = ROOT / '05_model_optimize' / 'SVC' / 'Normal' / 'results'


def say(*items):
    print(*items, flush=True)


def select_stratified(x, y, n, seed=42):
    if n >= len(y):
        return x, y
    ix, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return x[ix], y[ix]


def batched_predict(model, x, name, batch=256):
    output = []
    start = time.perf_counter()
    for i in range(0, len(x), batch):
        p = model.predict(x[i:i + batch]).astype(np.int64)
        output.append(p)
        if (i // batch) % 8 == 0:
            say(f'  {name}: {min(i + batch, len(x))}/{len(x)} predicted')
    return np.concatenate(output), round(time.perf_counter() - start, 2)


def run(args):
    svc, scaling, source = baseline()
    if not scaling:
        raise RuntimeError('Original project SVC baseline must use scaling')
    say('SVC baseline:', source)
    say('SVC params:', svc.get_params())
    say('SAFE PILOT ONLY. Official Test not loaded.')
    t = time.perf_counter()
    xtr, ytr, xval, yval, sha = fixed_split()
    say(f'Official split loaded in {time.perf_counter() - t:.2f} s: {len(ytr)} / {len(yval)}')
    xtr, ytr = select_stratified(xtr, ytr, args.train_size)
    xval, yval = select_stratified(xval, yval, args.val_size)
    say(f'Pilot train={len(ytr)} val={len(yval)}; X dtype={xtr.dtype}')
    hm = {w: load_writer(w) for w in WRITERS}
    com = BatchedCoM()
    t = time.perf_counter()
    centered_train = com.transform(xtr)
    centered_val = com.transform(xval)
    centered_hm = {w: (com.transform(hm[w][0]), hm[w][1]) for w in WRITERS}
    com_seconds = round(time.perf_counter() - t, 2)
    say('CoM preprocessed in', com_seconds, 'sec')
    if not np.isfinite(centered_train).all():
        raise RuntimeError('Nonfinite CoM transformed train')

    records = {}
    for label, train, val, hw in [
        ('baseline_subset', xtr, xval, {w: (hm[w][0], hm[w][1]) for w in WRITERS}),
        ('com_subset', centered_train, centered_val, centered_hm),
    ]:
        clf = pipeline(clone(svc), scaling)
        say(f'[{label}] fit START')
        t = time.perf_counter()
        clf.fit(train, ytr)
        fit_seconds = round(time.perf_counter() - t, 2)
        say(f'[{label}] fit DONE {fit_seconds}s')
        pv, predict_seconds = batched_predict(clf, val, label + ' official', args.batch_size)
        official = report(yval, pv)
        say(f'[{label}] official={official["accuracy"]:.6f}, F1={official["macro_f1"]:.6f} predict={predict_seconds}s')
        writers = []
        truths, preds = [], []
        for w in WRITERS:
            x, y = hw[w]
            p, pred_sec = batched_predict(clf, x, label + f' writer {w}', args.batch_size)
            m = report(y, p)
            writers.append({'writer': w, **m, 'predict_seconds': pred_sec})
            truths.append(y)
            preds.append(p)
            say(f'[{label}] writer {w}: {m["accuracy"]:.6f}')
        handmade = report(np.concatenate(truths), np.concatenate(preds))
        say(f'[{label}] Hand-made={handmade["accuracy"]:.6f} F1={handmade["macro_f1"]:.6f}')
        records[label] = {'fit_seconds': fit_seconds, 'official_validation': official,
                          'official_predict_seconds': predict_seconds,
                          'handmade_development': handmade, 'writers': writers}

    out = OUT_BASE / f'EXP_003_SVC_Normal_CoM_SAFE_PILOT_train{len(ytr)}'
    out.mkdir(parents=True, exist_ok=True)
    summary = {'experiment': 'EXP-003 CoM SAFE PILOT (subset, not comparable to EXP-001 full)',
               'train_size': len(ytr), 'validation_size': len(yval), 'com_preprocessing_seconds': com_seconds,
               'official_test_used': False, 'model_source': source, 'hyperparameters': svc.get_params(),
               'official_dataset_sha256': sha, 'results': records,
               'handmade_delta_pp': 100 * (records['com_subset']['handmade_development']['accuracy'] - records['baseline_subset']['handmade_development']['accuracy']),
               'official_delta_pp': 100 * (records['com_subset']['official_validation']['accuracy'] - records['baseline_subset']['official_validation']['accuracy'])}
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    with (out / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'fit_seconds', 'official_accuracy', 'official_macro_f1', 'handmade_accuracy', 'handmade_macro_f1'])
        writer.writeheader()
        for name, r in records.items():
            writer.writerow({'model': name, 'fit_seconds': r['fit_seconds'],
                             'official_accuracy': r['official_validation']['accuracy'],
                             'official_macro_f1': r['official_validation']['macro_f1'],
                             'handmade_accuracy': r['handmade_development']['accuracy'],
                             'handmade_macro_f1': r['handmade_development']['macro_f1']})
    say('PILOT COMPARISON: CoM minus baseline official:', f'{summary["official_delta_pp"]:+.4f}pp',
        'Hand-made:', f'{summary["handmade_delta_pp"]:+.4f}pp')
    say('Saved:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-size', type=int, default=8000, choices=(2000, 4000, 8000, 16000, 24000))
    parser.add_argument('--val-size', type=int, default=3200, choices=(800, 1600, 3200, 6400))
    parser.add_argument('--batch-size', type=int, default=256, choices=(64, 128, 256, 512))
    run(parser.parse_args())
