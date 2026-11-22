#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVC Normal EXP-004 safety pilot: CoM versus CoM followed by Deskew.

Run from repo root:
  python 05_model_optimize/SVC/Normal/scripts/EXP_004_SVC_Normal_Deskew_SafePilot.py
  python 05_model_optimize/SVC/Normal/scripts/EXP_004_SVC_Normal_Deskew_SafePilot.py --train-size 16000

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
from EXP011_KNN_Normal_Deskew import deskew, test_deskew
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
    say('EXP-004 SAFE PILOT. CoM vs CoM->Deskew. Official Test not loaded.')
    test_deskew()
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
    t = time.perf_counter()
    deskew_train, deskew_train_stats = deskew(centered_train)
    deskew_val, deskew_val_stats = deskew(centered_val)
    deskew_hm = {}
    for w in WRITERS:
        adjusted, _ = deskew(centered_hm[w][0])
        deskew_hm[w] = (adjusted, centered_hm[w][1])
    deskew_seconds = round(time.perf_counter() - t, 2)
    say('Deskew after CoM preprocessed in', deskew_seconds, 'sec')
    if not np.isfinite(deskew_train).all() or deskew_train.shape != centered_train.shape:
        raise RuntimeError('Deskew train invalid')
    if not np.isfinite(centered_train).all():
        raise RuntimeError('Nonfinite CoM transformed train')

    records = {}
    for label, train, val, hw in [
        ('com_subset', centered_train, centered_val, centered_hm),
        ('com_deskew_subset', deskew_train, deskew_val, deskew_hm),
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

    out = OUT_BASE / f'EXP_004_SVC_Normal_Deskew_SAFE_PILOT_train{len(ytr)}'
    out.mkdir(parents=True, exist_ok=True)
    comparison = records['com_deskew_subset']['handmade_development']['accuracy'] - records['com_subset']['handmade_development']['accuracy']
    off_delta = records['com_deskew_subset']['official_validation']['accuracy'] - records['com_subset']['official_validation']['accuracy']
    summary = {
        'experiment': 'EXP-004 SVC CoM -> Deskew SAFE PILOT (subset only)',
        'train_size': len(ytr), 'validation_size': len(yval),
        'com_preprocessing_seconds': com_seconds, 'deskew_preprocessing_seconds': deskew_seconds,
        'deskew_train_diagnostics': deskew_train_stats,
        'deskew_validation_diagnostics': deskew_val_stats,
        'official_test_used': False,
        'model_source': source, 'hyperparameters': svc.get_params(),
        'official_dataset_sha256': sha, 'results': records,
        'handmade_delta_pp': 100 * comparison,
        'official_delta_pp': 100 * off_delta,
        'note': '8,000-row development pilot; do not directly compare to 76,800-row full baseline',
    }
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
    say('PILOT COMPARISON: CoM+Deskew minus CoM official:', f'{summary["official_delta_pp"]:+.4f}pp',
        'Hand-made:', f'{summary["handmade_delta_pp"]:+.4f}pp')
    say('Saved:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-size', type=int, default=8000, choices=(2000, 4000, 8000, 16000))
    parser.add_argument('--val-size', type=int, default=3200, choices=(800, 1600, 3200, 6400))
    parser.add_argument('--batch-size', type=int, default=256, choices=(64, 128, 256, 512))
    run(parser.parse_args())
