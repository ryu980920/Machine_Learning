#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVC Normal EXP-002: inspect saved EXP-001 predictions, without retraining or Test access."""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / '05_model_optimize' / 'SVC' / 'Normal' / 'results'
SRC = BASE / 'EXP_001_SVC_Normal_Baseline'
OUT = BASE / 'EXP_002_SVC_Normal_Error_Analysis'
LABELS = tuple('0123456789ABCDEF')


def write_csv(path, rows, cols):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    path = SRC / 'handmade_predictions.csv'
    if not path.exists():
        raise FileNotFoundError(f'EXP-001 result missing: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 2560:
        raise ValueError(f'Expected 2560 Hand-made predictions, found {len(rows)}')
    counter = Counter()
    writer_counter = Counter()
    label_total = Counter()
    label_errors = Counter()
    by_writer = {}
    for row in rows:
        w = int(row['writer'])
        t, p = row['true'], row['prediction']
        if w not in (41, 42, 43, 44) or t not in LABELS or p not in LABELS:
            raise ValueError(f'Invalid prediction row: {row}')
        label_total[t] += 1
        by_writer.setdefault(w, [0, 0])
        by_writer[w][0] += 1
        if t != p:
            counter[(t, p)] += 1
            writer_counter[w] += 1
            label_errors[t] += 1
        else:
            by_writer[w][1] += 1
    if sorted(by_writer) != [41, 42, 43, 44] or any(n != 640 for n, _ in by_writer.values()):
        raise ValueError('Writer counts are inconsistent')
    if any(label_total[c] != 160 for c in LABELS):
        raise ValueError('Each class must have exactly 160 Hand-made examples')
    pairs = [{'true': t, 'predicted': p, 'count': n}
             for (t, p), n in sorted(counter.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))]
    classes = [{'class': c, 'total': label_total[c], 'errors': label_errors[c],
                'correct': label_total[c] - label_errors[c],
                'recall': (label_total[c] - label_errors[c]) / label_total[c]}
               for c in LABELS]
    writers = [{'writer': w, 'total': n, 'correct': ok,
                'errors': writer_counter[w], 'accuracy': ok / n}
               for w, (n, ok) in sorted(by_writer.items())]
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'confusion_pairs.csv', pairs, ['true', 'predicted', 'count'])
    write_csv(OUT / 'per_class_error.csv', classes, ['class', 'total', 'errors', 'correct', 'recall'])
    write_csv(OUT / 'writer_error.csv', writers, ['writer', 'total', 'correct', 'errors', 'accuracy'])
    summary = {'experiment': 'EXP-002 SVC Normal error analysis', 'source': str(path.relative_to(ROOT)),
               'handmade_n': len(rows), 'correct': sum(r['correct'] for r in writers),
               'errors': sum(r['errors'] for r in writers), 'top_confusions': pairs[:12],
               'weakest_classes': sorted(classes, key=lambda x: (x['recall'], x['class']))[:8],
               'writers': writers, 'model_retrained': False, 'official_test_used': False,
               'evaluation_role': 'Repeated-use Hand-made development; not independent test'}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SVC Normal EXP-002 — Error analysis (no retraining)', flush=True)
    print(f"Hand-made={len(rows)}, correct={summary['correct']}, errors={summary['errors']}")
    print('TOP CONFUSIONS (true -> predicted):')
    for r in pairs[:10]:
        print(f"  {r['true']} -> {r['predicted']}: {r['count']}")
    print('WEAKEST CLASSES:')
    for r in summary['weakest_classes'][:8]:
        print(f"  {r['class']}: recall={r['recall']:.4f}, errors={r['errors']}/160")
    print('Saved:', OUT)


if __name__ == '__main__':
    main()
