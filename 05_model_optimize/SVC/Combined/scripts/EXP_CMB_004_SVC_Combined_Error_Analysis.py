#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze saved SVC Combined writer-heldout predictions without retraining."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / '05_model_optimize' / 'SVC' / 'Combined' / 'results'
AUG = BASE / 'EXP_CMB_003_SVC_Hybrid_Handmade_Augmentation'
OUT = BASE / 'EXP_CMB_004_SVC_Combined_Error_Analysis'
WRITERS = (41, 42, 43, 44)
RATIOS = (25, 50, 100)
EXPECTED = {0: 2403, 25: 2414, 50: 2424, 100: 2430}


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing experiment result: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_predictions(ratio):
    path = AUG / f'ratio_{ratio}_oof_predictions.csv'
    rows = read_csv(path)
    result = {}
    for r in rows:
        key = (int(r['writer']), int(r['index']))
        if key in result:
            raise ValueError(f'Duplicate sample in ratio {ratio}: {key}')
        if key[0] not in WRITERS or not (0 <= key[1] < 640):
            raise ValueError(f'Unexpected writer/index: {key}')
        result[key] = {'true': r['true'], 'control': r['control_pred'], 'pred': r['augmented_pred']}
    expected_keys = {(w, i) for w in WRITERS for i in range(640)}
    if set(result) != expected_keys:
        raise ValueError(f'Ratio {ratio} has missing or unexpected samples')
    correct = sum(r['true'] == r['pred'] for r in result.values())
    control_correct = sum(r['true'] == r['control'] for r in result.values())
    if correct != EXPECTED[ratio] or control_correct != EXPECTED[0]:
        raise ValueError(f'Ratio {ratio} results changed: {correct} / control {control_correct}')
    return result


def analyze():
    data = {ratio: load_predictions(ratio) for ratio in RATIOS}
    reference = data[100]
    keys = sorted(reference)
    for ratio in RATIOS:
        for key in keys:
            if data[ratio][key]['true'] != reference[key]['true'] or data[ratio][key]['control'] != reference[key]['control']:
                raise ValueError(f'Inconsistent truth/control predictions for {key} at ratio {ratio}')
    labels = list('0123456789ABCDEF')
    if set(r['true'] for r in reference.values()) != set(labels):
        raise ValueError('Expected all 16 MNIST/EMNIST classes')
    for ratio in RATIOS:
        if not set(r['pred'] for r in data[ratio].values()) <= set(labels):
            raise ValueError(f'Unknown prediction label at ratio {ratio}')
    OUT.mkdir(parents=True, exist_ok=True)
    y = [reference[k]['true'] for k in keys]
    control = [reference[k]['control'] for k in keys]
    pred = [reference[k]['pred'] for k in keys]
    cm = confusion_matrix(y, pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = []
    for i, lab in enumerate(labels):
        baseline_n = sum(reference[k]['true'] == lab and reference[k]['control'] == lab for k in keys)
        per_class.append({'class': lab, 'support': int(support[i]), 'correct': int(cm[i, i]),
                          'errors': int(support[i] - cm[i, i]), 'precision': float(precision[i]),
                          'recall': float(recall[i]), 'f1': float(f1[i]),
                          'control_correct': baseline_n, 'gain_vs_control': int(cm[i, i]) - baseline_n})
    save_csv(OUT / 'per_class.csv', per_class,
             ['class', 'support', 'correct', 'errors', 'precision', 'recall', 'f1', 'control_correct', 'gain_vs_control'])
    save_csv(OUT / 'confusion_matrix.csv',
             [{'true': lab, **{p: int(cm[i, j]) for j, p in enumerate(labels)}} for i, lab in enumerate(labels)],
             ['true', *labels])
    confusions = sorted(({'true': labels[i], 'predicted': labels[j], 'count': int(cm[i,j])}
                         for i in range(16) for j in range(16) if i != j and cm[i,j] > 0),
                        key=lambda z: (-z['count'], z['true'], z['predicted']))
    save_csv(OUT / 'top_confusions.csv', confusions, ['true','predicted','count'])
    writer_rows = []
    for w in WRITERS:
        samples = [k for k in keys if k[0] == w]
        cnt = sum(reference[k]['pred'] == reference[k]['true'] for k in samples)
        corrected = sum(reference[k]['control'] != reference[k]['true'] and reference[k]['pred'] == reference[k]['true'] for k in samples)
        regressed = sum(reference[k]['control'] == reference[k]['true'] and reference[k]['pred'] != reference[k]['true'] for k in samples)
        writer_rows.append({'writer': w, 'correct': cnt, 'incorrect': 640-cnt, 'accuracy': cnt/640,
                            'corrected_from_control': corrected, 'new_errors_vs_control': regressed})
    save_csv(OUT / 'per_writer.csv', writer_rows, ['writer','correct','incorrect','accuracy','corrected_from_control','new_errors_vs_control'])
    all_rows = []
    for w,i in keys:
        row = reference[(w,i)]
        status = ('corrected' if row['control'] != row['true'] and row['pred'] == row['true'] else
                  'new_error' if row['control'] == row['true'] and row['pred'] != row['true'] else
                  'still_wrong' if row['pred'] != row['true'] else 'still_correct')
        all_rows.append({'writer': w, 'index': i, 'true': row['true'], 'control_pred': row['control'],
                         'aug25_pred': data[25][(w,i)]['pred'], 'aug50_pred': data[50][(w,i)]['pred'],
                         'aug100_pred': row['pred'], 'control_correct': int(row['control'] == row['true']),
                         'aug100_correct': int(row['pred'] == row['true']), 'change_from_control': status})
    fields = list(all_rows[0])
    save_csv(OUT / 'all_samples.csv', all_rows, fields)
    save_csv(OUT / 'remaining_errors_100.csv', [r for r in all_rows if r['aug100_correct'] == 0], fields)
    save_csv(OUT / 'corrected_from_control.csv', [r for r in all_rows if r['change_from_control']=='corrected'], fields)
    save_csv(OUT / 'new_errors_vs_control.csv', [r for r in all_rows if r['change_from_control']=='new_error'], fields)
    ratio_rows = []
    for ratio in (0, *RATIOS):
        n = EXPECTED[ratio]
        ratio_rows.append({'augmentation_percent': ratio, 'correct': n, 'errors': 2560-n,
                           'accuracy': n/2560, 'delta_vs_control_pp': 100*(n-EXPECTED[0])/2560})
    save_csv(OUT / 'augmentation_comparison.csv', ratio_rows,
             ['augmentation_percent','correct','errors','accuracy','delta_vs_control_pp'])
    persistent = [r for r in all_rows if all(r[f'aug{x}_pred'] != r['true'] for x in RATIOS)]
    save_csv(OUT / 'errors_persistent_25_50_100.csv', persistent, fields)
    counts = Counter(r['change_from_control'] for r in all_rows)
    if len([r for r in all_rows if not r['aug100_correct']]) != 130 or counts['corrected'] != 28 or counts['new_error'] != 1:
        raise RuntimeError('100% augmentation summary does not match saved EXP-CMB-003 metrics')
    summary = {'experiment': 'EXP-CMB-004 SVC Combined saved prediction error analysis',
               'source': str(AUG.relative_to(ROOT)), 'official_test_used': False,
               'note': 'Repeated-use writer-heldout DEVELOPMENT samples, not independent final test',
               'n': 2560, 'correct': 2430, 'errors': 130, 'accuracy': 2430/2560,
               'paired_vs_control': dict(counts), 'persistent_errors_across_25_50_100': len(persistent),
               'top_confusions': confusions[:12], 'weakest_recall_classes': sorted(per_class,key=lambda r:r['recall'])[:8],
               'by_writer': writer_rows, 'augmentation': ratio_rows}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nEXP-CMB-004: saved predictions only; NO SVC training / NO official Test', flush=True)
    for r in ratio_rows:
        print(f"Aug {r['augmentation_percent']:3d}%: {r['correct']}/2560 accuracy={r['accuracy']:.6f}",flush=True)
    print('100% augmented remaining errors:', 130,flush=True)
    print('Paired changes vs 0%:', dict(counts),flush=True)
    print('Top confusions:', [(r['true']+'->'+r['predicted'],r['count']) for r in confusions[:10]],flush=True)
    print('Weakest recalls:', [(r['class'],round(r['recall'],4)) for r in sorted(per_class,key=lambda r:r['recall'])[:6]],flush=True)
    print('Persistent errors across 25%, 50%, 100%:',len(persistent),flush=True)
    print('Saved:',OUT,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    analyze()
