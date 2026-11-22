#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVC Normal EXP-007: full official Training, HOG or optional Pixel:HOG.

From the repository root:
  python 05_model_optimize/SVC/Normal/scripts/EXP_007_SVC_Normal_Full_Validation.py
  python 05_model_optimize/SVC/Normal/scripts/EXP_007_SVC_Normal_Full_Validation.py --feature hybrid

Only official Training/Validation, never Test. This script does not alter prior results.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
import gc
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(ROOT / '05_model_optimize' / 'SVC' / 'Normal' / 'scripts'),
    str(ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'),
    str(ROOT / '03_experiments' / 'model_optimization'),
]
from EXP_001_SVC_Normal_Baseline import WRITERS, baseline, load_writer, report
from EXP_005_SVC_Normal_HOG_SafePilot import predict_batches
from EXP004_KNN_Normal_CoM import BatchedCoM
from EXP011_KNN_Normal_Deskew import deskew, test_deskew
from EXP012_KNN_Normal_HOG import hog_features, hybrid_features, test_hog
from common import fixed_split, LABELS

OUT = ROOT / '05_model_optimize' / 'SVC' / 'Normal' / 'results' / 'EXP_007_SVC_Normal_FULL_76800'


def say(*items):
    print(*items, flush=True)


def save_json(path, content):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(content, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(temp, path)


def save_csv(path, records, columns):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def features(x, feature, title):
    start = time.perf_counter()
    com = BatchedCoM().transform(x)
    say(f'[{title}] CoM {time.perf_counter() - start:.2f}s')
    start = time.perf_counter()
    pixel, _ = deskew(com)
    del com
    say(f'[{title}] Deskew {time.perf_counter() - start:.2f}s')
    start = time.perf_counter()
    hog = hog_features(pixel)
    say(f'[{title}] HOG {time.perf_counter() - start:.2f}s')
    if feature == 'hybrid':
        start = time.perf_counter()
        feat = hybrid_features(pixel, hog, pixel_weight=0.20)
        say(f'[{title}] Hybrid 20:80 {time.perf_counter() - start:.2f}s')
    else:
        feat = np.ascontiguousarray(hog, dtype=np.float32)
    del pixel, hog
    gc.collect()
    if feat.dtype != np.float32 or not np.isfinite(feat).all():
        raise RuntimeError(f'{title}: invalid features')
    return feat


def run(args):
    svc, scaling, source = baseline()
    if not scaling or svc.kernel != 'rbf':
        raise RuntimeError('Unexpected team baseline: expected pixel-scaled RBF SVC')
    test_deskew()
    test_hog()
    say('EXP-007: FULL official training 76800 + validation 19200')
    say('Feature:', args.feature, '| SVC: C=1, gamma=scale, kernel=rbf')
    say('Official Test is NOT loaded. Hand-made is DEVELOPMENT only.')
    say('Original SVC source:', source)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{args.feature}_metrics.json'
    if path.exists() and not args.overwrite:
        raise FileExistsError(f'Results already exist: {path}. Use --overwrite to rerun intentionally.')
    t = time.perf_counter()
    xtr, ytr, xval, yval, sha = fixed_split()
    if (len(ytr), len(yval)) != (76800, 19200):
        raise RuntimeError('Official split mismatch; refusing partial experiment')
    writers = {w: load_writer(w) for w in WRITERS}
    say(f'Data loaded in {time.perf_counter()-t:.2f}s; {len(ytr)} / {len(yval)} + handmade 2560')
    status = {'experiment': 'EXP-007 SVC Normal full training', 'feature': args.feature,
              'official_training': len(ytr), 'official_validation_size': len(yval),
              'official_test_used': False, 'dataset_sha256': sha,
              'baseline_source': source, 'parameters': {'kernel':'rbf','C':1.0,'gamma':'scale',
                  'cache_size_mb': args.cache_size}, 'status':'preprocessing',
              'note': 'Hand-made evaluation is repeated-use development data.'}
    save_json(OUT / f'{args.feature}_status.json', status)
    train = features(xtr, args.feature, 'Train 76800')
    del xtr
    gc.collect()
    val = features(xval, args.feature, 'Validation 19200')
    del xval
    hw = {}
    for w in WRITERS:
        x, y, _ = writers[w]
        hw[w] = (features(x, args.feature, f'Writer {w}'), y)
    del writers
    gc.collect()
    say(f'Prepared train={train.shape}, val={val.shape}, dtype={train.dtype}')
    status['status'] = 'fitting'
    status['feature_dimensions'] = int(train.shape[1])
    save_json(OUT / f'{args.feature}_status.json', status)
    model = clone(svc).set_params(C=1.0, gamma='scale', cache_size=args.cache_size)
    say('SVC fit START (full dataset; RBF optimization may take substantial CPU/RAM)')
    t = time.perf_counter()
    model.fit(train, ytr)
    fit_seconds = round(time.perf_counter()-t, 2)
    say(f'SVC fit DONE: {fit_seconds:.2f}s; support_vectors={len(model.support_)}')
    status.update(status='trained', fit_seconds=fit_seconds,
                  support_vectors=int(len(model.support_)))
    save_json(OUT / f'{args.feature}_status.json', status)
    del train
    gc.collect()
    pred, val_seconds = predict_batches(model, val, 'Official Validation', args.batch_size)
    official = report(yval, pred)
    say(f'Official Validation: accuracy={official["accuracy"]:.6f} F1={official["macro_f1"]:.6f}')
    status.update(status='official_evaluated', official_validation=official,
                  validation_predict_seconds=val_seconds)
    save_json(OUT / f'{args.feature}_status.json', status)
    del val
    gc.collect()
    yt, yp, writer_rows, pred_rows = [], [], [], []
    for w in WRITERS:
        x, y = hw[w]
        hp, htime = predict_batches(model, x, f'Writer {w}', args.batch_size)
        stats = report(y, hp)
        writer_rows.append({'writer':w, **stats, 'predict_seconds': htime})
        say(f'Writer {w}: {stats["correct"]}/640 accuracy={stats["accuracy"]:.6f}')
        pred_rows.extend({'writer':w,'index':i,'true':LABELS[int(a)],
                          'predicted': LABELS[int(b)], 'correct':int(a==b)}
                         for i,(a,b) in enumerate(zip(y,hp)))
        yt.append(y)
        yp.append(hp)
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)
    handmade = report(yt,yp)
    cm = confusion_matrix(yt,yp,labels=np.arange(16))
    pr, re, f1, support = precision_recall_fscore_support(
        yt,yp,labels=np.arange(16),zero_division=0)
    save_csv(OUT / f'{args.feature}_writers.csv',writer_rows,
             ['writer','n','correct','accuracy','macro_f1','predict_seconds'])
    save_csv(OUT / f'{args.feature}_handmade_predictions.csv',pred_rows,
             ['writer','index','true','predicted','correct'])
    save_csv(OUT / f'{args.feature}_handmade_confusion_matrix.csv',
             [{'true': LABELS[i], **{LABELS[j]:int(cm[i,j]) for j in range(16)}} for i in range(16)],
             ['true',*LABELS])
    save_csv(OUT / f'{args.feature}_per_class.csv',
             [{'class':LABELS[i],'precision':float(pr[i]),'recall':float(re[i]),
               'f1':float(f1[i]),'support':int(support[i])} for i in range(16)],
             ['class','precision','recall','f1','support'])
    status.update(status='complete', handmade_development=handmade, writers=writer_rows)
    save_json(path,status)
    save_json(OUT / f'{args.feature}_status.json',status)
    say(f'FULL EXP-007 {args.feature}: official={official["accuracy"]:.6f} '
        f'handmade={handmade["accuracy"]:.6f} macro_f1={handmade["macro_f1"]:.6f} '
        f'({handmade["correct"]}/2560)')
    say('Results saved:',OUT)
    say('Official Test was NOT loaded.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--feature',choices=['hog','hybrid'],default='hog')
    parser.add_argument('--batch-size',type=int,default=256,choices=[64,128,256,512])
    parser.add_argument('--cache-size',type=int,default=200,help='LibSVM cache size in MB')
    parser.add_argument('--overwrite',action='store_true')
    run(parser.parse_args())
