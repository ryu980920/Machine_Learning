#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-006 SVC Normal: RBF C/gamma grid on frozen CoM->Deskew->HOG features.

Run from repository root:
  python 05_model_optimize/SVC/Normal/scripts/EXP_006_SVC_Normal_C_Gamma_SafePilot.py

Safe pilot: 8,000 official train / 3,200 official validation, all 2,560
Hand-made development images. The official Test data are never loaded.
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

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(ROOT / '05_model_optimize' / 'SVC' / 'Normal' / 'scripts'),
    str(ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'),
    str(ROOT / '03_experiments' / 'model_optimization'),
]
from EXP_001_SVC_Normal_Baseline import WRITERS, baseline, load_writer, report
from EXP_005_SVC_Normal_HOG_SafePilot import predict_batches, select_stratified
from EXP004_KNN_Normal_CoM import BatchedCoM
from EXP011_KNN_Normal_Deskew import deskew, test_deskew
from EXP012_KNN_Normal_HOG import hog_features, test_hog
from common import fixed_split

RESULTS = ROOT / '05_model_optimize' / 'SVC' / 'Normal' / 'results'


def say(*args):
    print(*args, flush=True)


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def run(args):
    svc, scaling, source = baseline()
    if not scaling or svc.kernel != 'rbf':
        raise RuntimeError('Expected the original project SVC with RBF kernel and pixel scaling')
    say('SVC baseline:', source)
    say('Baseline C/gamma:', svc.C, svc.gamma)
    say('EXP-006 SAFE PILOT: CoM -> Deskew -> HOG, C/gamma search')
    say('Official Test NOT loaded. Hand-made is repeated-use DEVELOPMENT data.')
    test_deskew()
    test_hog()

    t = time.perf_counter()
    xtr, ytr, xval, yval, sha = fixed_split()
    say(f'Official split loaded in {time.perf_counter() - t:.2f}s: {len(ytr)} / {len(yval)}')
    xtr, ytr = select_stratified(xtr, ytr, args.train_size)
    xval, yval = select_stratified(xval, yval, args.val_size)
    writers = {w: load_writer(w) for w in WRITERS}
    say(f'Pilot train={len(ytr)}, validation={len(yval)}, handmade={sum(len(writers[w][1]) for w in WRITERS)}')

    t = time.perf_counter()
    com = BatchedCoM()
    centered_train = com.transform(xtr)
    centered_val = com.transform(xval)
    centered_writers = {w: com.transform(writers[w][0]) for w in WRITERS}
    say(f'CoM preprocessed in {time.perf_counter() - t:.2f}s')

    t = time.perf_counter()
    pixel_train, _ = deskew(centered_train)
    pixel_val, _ = deskew(centered_val)
    pixel_writers = {w: deskew(centered_writers[w])[0] for w in WRITERS}
    say(f'Deskew preprocessed in {time.perf_counter() - t:.2f}s')

    t = time.perf_counter()
    hog_train = np.ascontiguousarray(hog_features(pixel_train), dtype=np.float32)
    hog_val = np.ascontiguousarray(hog_features(pixel_val), dtype=np.float32)
    hog_writers = {w: np.ascontiguousarray(hog_features(pixel_writers[w]), dtype=np.float32) for w in WRITERS}
    if hog_train.shape != (len(ytr), 324) or not np.isfinite(hog_train).all():
        raise RuntimeError(f'Unexpected HOG features: {hog_train.shape}')
    say(f'HOG features prepared in {time.perf_counter() - t:.2f}s; dims={hog_train.shape[1]}')

    output_dir = RESULTS / f'EXP_006_SVC_Normal_C_Gamma_SAFE_PILOT_train{len(ytr)}'
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    records = {}
    c_values = [0.1, 1.0, 10.0]
    gamma_values = ['scale', 0.01, 0.1]

    for c in c_values:
        for gamma in gamma_values:
            name = f'C_{c:g}_gamma_{gamma}'
            model = clone(svc).set_params(C=c, gamma=gamma, cache_size=300)
            say(f'\n[{name}] fit START rows={len(ytr)} dims=324')
            t = time.perf_counter()
            model.fit(hog_train, ytr)
            fit_sec = round(time.perf_counter() - t, 2)
            say(f'[{name}] fit DONE {fit_sec}s')
            official_pred, official_sec = predict_batches(model, hog_val, f'{name} official', args.batch_size)
            official = report(yval, official_pred)
            say(f'[{name}] official={official["accuracy"]:.6f} F1={official["macro_f1"]:.6f} predict={official_sec}s')
            ys, ps, writer_rows = [], [], []
            for w in WRITERS:
                pred, sec = predict_batches(model, hog_writers[w], f'{name} writer {w}', args.batch_size)
                target = writers[w][1]
                wr = report(target, pred)
                writer_rows.append({'writer': w, 'accuracy': wr['accuracy'], 'macro_f1': wr['macro_f1'], 'n': len(target), 'predict_seconds': sec})
                ys.append(target)
                ps.append(pred)
                say(f'[{name}] writer {w} accuracy={wr["accuracy"]:.6f}')
            handmade = report(np.concatenate(ys), np.concatenate(ps))
            say(f'[{name}] Hand-made={handmade["accuracy"]:.6f} F1={handmade["macro_f1"]:.6f}')
            rec = {'C': c, 'gamma': gamma, 'fit_seconds': fit_sec,
                   'official_predict_seconds': official_sec, 'official_validation': official,
                   'handmade_development': handmade, 'writers': writer_rows}
            records[name] = rec
            rows.append({'candidate': name, 'C': c, 'gamma': gamma, 'train_size': len(ytr),
                         'fit_seconds': fit_sec, 'official_accuracy': official['accuracy'],
                         'official_macro_f1': official['macro_f1'],
                         'handmade_accuracy': handmade['accuracy'],
                         'handmade_macro_f1': handmade['macro_f1']})
            save_json(output_dir / 'metrics_partial.json', records)
            with (output_dir / 'comparison_partial.csv').open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            del model

    control = records['C_1_gamma_scale']
    if args.train_size == 8000 and args.val_size == 3200:
        expected = (0.9753125, 0.89375)
        found = (control['official_validation']['accuracy'], control['handmade_development']['accuracy'])
        if not np.allclose(found, expected, atol=1e-9):
            raise RuntimeError(f'EXP-005 HOG control mismatch: found={found}; expected={expected}')
        say('EXP-005 HOG CONTROL EXACTLY REPRODUCED')

    # Development choice prioritizes Hand-made macro F1; use official accuracy to break ties.
    best = max(rows, key=lambda r: (r['handmade_macro_f1'], r['official_accuracy']))
    result = {'experiment': 'EXP-006 SVC Normal RBF C/gamma safe pilot',
              'train_size': len(ytr), 'validation_size': len(yval), 'official_test_used': False,
              'model_source': source, 'baseline_params': svc.get_params(),
              'official_dataset_sha256': sha, 'preprocessing': 'CoM -> Deskew -> HOG(324)',
              'feature_scaling': 'HOG pre-normalized; no pixel/255 scaling',
              'results': records, 'selected_development_candidate': best['candidate'],
              'note': 'Hand-made data used repeatedly for development, not an independent test.'}
    save_json(output_dir / 'metrics.json', result)
    with (output_dir / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    say('\nEXP-006 SAFE PILOT SUMMARY')
    for row in rows:
        say(f'  {row["candidate"]:<24} official={row["official_accuracy"]:.6f} '
            f'handmade={row["handmade_accuracy"]:.6f} F1={row["handmade_macro_f1"]:.6f}')
    say('SELECTED DEVELOPMENT CANDIDATE:', best['candidate'])
    say('Saved:', output_dir)
    say('Official Test was NOT loaded.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-size', type=int, default=8000, choices=(2000, 4000, 8000, 16000))
    parser.add_argument('--val-size', type=int, default=3200, choices=(800, 1600, 3200, 6400))
    parser.add_argument('--batch-size', type=int, default=256, choices=(64, 128, 256, 512))
    run(parser.parse_args())
