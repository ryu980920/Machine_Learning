#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVC CoM diagnostics; pilot only, never fits full 76,800 rows."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[4]
SVC_SCRIPTS = ROOT / '05_model_optimize' / 'SVC' / 'Normal' / 'scripts'
KNN_SCRIPTS = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'scripts'
sys.path[:0] = [str(SVC_SCRIPTS), str(KNN_SCRIPTS), str(ROOT / '03_experiments' / 'model_optimization')]

from EXP_001_SVC_Normal_Baseline import baseline, report
from EXP004_KNN_Normal_CoM import BatchedCoM
from common import fixed_split, pipeline


def log(*args):
    print(*args, flush=True)


def run(train_n, val_n):
    model, scaled, source = baseline()
    if not scaled:
        raise RuntimeError('SVC baseline scaling must be True')
    log('SVC baseline source:', source)
    log('SVC parameters:', model.get_params())
    log('Pilot training:', train_n, 'validation:', val_n)
    log('Official Test is NOT loaded')
    t = time.perf_counter()
    xtr, ytr, xva, yva, sha = fixed_split()
    log('Data loaded in seconds:', round(time.perf_counter() - t, 2))
    trids, _ = train_test_split(np.arange(len(ytr)), train_size=train_n, stratify=ytr, random_state=42)
    vaids, _ = train_test_split(np.arange(len(yva)), train_size=val_n, stratify=yva, random_state=42)
    xt, yt, xv, yv = xtr[trids], ytr[trids], xva[vaids], yva[vaids]
    del xtr, ytr, xva, yva
    log('Pilot shapes:', xt.shape, xv.shape, 'dtype:', xt.dtype)
    t = time.perf_counter()
    com = BatchedCoM()
    centered_train = com.transform(xt)
    log('CoM train seconds:', round(time.perf_counter() - t, 2))
    t = time.perf_counter()
    centered_val = com.transform(xv)
    log('CoM validation seconds:', round(time.perf_counter() - t, 2))
    log('CoM train range:', float(centered_train.min()), float(centered_train.max()))
    log('CoM train finite:', bool(np.isfinite(centered_train).all()))
    log('CoM train zero rows:', int(np.count_nonzero(np.max(np.abs(centered_train), axis=1) == 0)))
    log('CoM train dtype:', centered_train.dtype)
    if not np.isfinite(centered_train).all():
        raise RuntimeError('Non-finite transformed values')
    clf = pipeline(model, scaled)
    log('SVC pilot fit start...')
    t = time.perf_counter()
    clf.fit(centered_train, yt)
    log('SVC fit seconds:', round(time.perf_counter() - t, 2))
    log('Support vectors:', len(clf.named_steps['svc'].support_) if 'svc' in clf.named_steps else 'see pipeline steps below')
    log('Pipeline steps:', list(clf.named_steps))
    t = time.perf_counter()
    pred = clf.predict(centered_val)
    log('SVC predict seconds:', round(time.perf_counter() - t, 2))
    log('Pilot official Validation:', report(yv, pred))
    log('DIAGNOSTIC COMPLETE; no full training and no experiment results overwritten')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=2000)
    parser.add_argument('--val', type=int, default=800)
    args = parser.parse_args()
    if not 100 <= args.train <= 8000 or not 100 <= args.val <= 3200:
        parser.error('diagnostics restricted to 100..8000 train and 100..3200 validation')
    run(args.train, args.val)
