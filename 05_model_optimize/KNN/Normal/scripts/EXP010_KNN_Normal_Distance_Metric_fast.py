#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-010 fast continuation: reuse completed EXP-010 CSV results, do not rerun distances.

Run from the repository root:
  python 05_model_optimize/KNN/Normal/scripts/EXP010_KNN_Normal_Distance_Metric_fast.py

The original EXP-010 writes pilot_results.csv after each pilot and
full_results.csv after every completed full candidate. This script only
reads those files. An interrupted p=1.5 full computation cannot be resumed;
its result is NEVER inferred from its pilot score.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ["EXP010_PROJECT_ROOT"]).resolve() if "EXP010_PROJECT_ROOT" in os.environ else Path(__file__).resolve().parents[4]
RESULTS = ROOT / '05_model_optimize' / 'KNN' / 'Normal' / 'results' / 'EXP010_distance_metrics'
EXPECTED_P2 = (18381, 2153)
MIN_OFFICIAL_ACC = 0.957


def load_rows(path):
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def numeric(row, name):
    return float(row[name])


def get_p(row):
    return float(row['p'])


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    pilot_path = RESULTS / 'pilot_results.csv'
    full_path = RESULTS / 'full_results.csv'
    pilot = load_rows(pilot_path)
    full = load_rows(full_path)
    if not pilot and not full:
        print('No completed EXP-010 CSV found in:', RESULTS)
        print('The original run did not save completed results here. Do not rerun the slow FULL p=1.5 command.')
        return 1
    print(f'Cached results: PILOT={len(pilot)} row(s); FULL={len(full)} row(s)')
    print('NO KNN training or distance computation will be performed.\n')

    unique_pilot = {get_p(row): row for row in pilot}
    unique_full = {get_p(row): row for row in full}
    if 2.0 in unique_full:
        control = unique_full[2.0]
        correct_pair = (int(control['official_correct']), int(control['handmade_correct']))
        if correct_pair != EXPECTED_P2:
            raise RuntimeError(f'EXP-009 baseline mismatch: {correct_pair} != {EXPECTED_P2}')
        print('FULL p=2 EXACT REPRODUCTION VERIFIED.')
    else:
        print('FULL p=2 not present; this continuation cannot verify the EXP-009 baseline.')

    print('\nPILOT (reduced train AND evaluation sets; not full-comparable):')
    pilot_ranked = sorted(unique_pilot.values(), key=lambda r: (
        numeric(r, 'handmade_macro_f1'), numeric(r, 'handmade_accuracy'),
        numeric(r, 'official_macro_f1')), reverse=True)
    for row in sorted(unique_pilot.values(), key=get_p):
        print(f" p={get_p(row):g}: official={numeric(row, 'official_accuracy'):.6f}, "
              f"handmade={numeric(row, 'handmade_accuracy'):.6f}, "
              f"handmade_F1={numeric(row, 'handmade_macro_f1'):.6f}")
    if pilot_ranked:
        print(f" Pilot leader: p={get_p(pilot_ranked[0]):g} (PILOT ONLY)")

    print('\nFULL (only finished, saved models):')
    for row in sorted(unique_full.values(), key=get_p):
        print(f" p={get_p(row):g}: official={numeric(row, 'official_accuracy'):.6f}, "
              f"handmade={numeric(row, 'handmade_accuracy'):.6f}, "
              f"handmade_F1={numeric(row, 'handmade_macro_f1'):.6f}")

    eligible = [r for r in unique_full.values()
                if numeric(r, 'official_accuracy') >= MIN_OFFICIAL_ACC]
    selected = max(eligible, key=lambda r: (
        numeric(r, 'handmade_macro_f1'), numeric(r, 'handmade_accuracy'),
        numeric(r, 'official_macro_f1'))) if eligible else None
    candidate = selected and get_p(selected)
    if candidate is None:
        print('\nNo FULL-evaluated candidate met the official guardrail.')
    else:
        print(f'\nSelected among FULL-evaluated models only: p={candidate:g}')
    if len(unique_full) < 4:
        print('INCOMPLETE SEARCH: other p values were not fully evaluated. No claim of global best p.')

    rows = []
    for p in (1.0, 1.5, 2.0, 3.0):
        a = unique_pilot.get(p)
        b = unique_full.get(p)
        rows.append({
            'p': p,
            'pilot_official_accuracy': a['official_accuracy'] if a else '',
            'pilot_handmade_accuracy': a['handmade_accuracy'] if a else '',
            'pilot_handmade_macro_f1': a['handmade_macro_f1'] if a else '',
            'full_status': 'COMPLETED' if b else 'NOT_COMPLETED',
            'full_official_accuracy': b['official_accuracy'] if b else '',
            'full_handmade_accuracy': b['handmade_accuracy'] if b else '',
            'full_handmade_macro_f1': b['handmade_macro_f1'] if b else '',
            'full_guardrail_pass': (numeric(b, 'official_accuracy') >= MIN_OFFICIAL_ACC) if b else '',
        })
    fields = list(rows[0].keys())
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / 'fast_comparison.csv', rows, fields)
    report = {
        'experiment': 'EXP-010 fast continuation (cached results only)',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'source_files': [str(pilot_path), str(full_path)],
        'pilot_rows_count': len(pilot),
        'full_rows_count': len(full),
        'official_guardrail': MIN_OFFICIAL_ACC,
        'full_reproduced_exp009': 2.0 in unique_full,
        'selected_from_completed_full_only_p': candidate,
        'exhaustive_distance_search': len(unique_full) == 4,
        'comparison': rows,
        'limitations': [
            'Pilot has a much smaller training and evaluation set; pilot scores must not be compared with full scores.',
            'An interrupted FULL p=1.5 result is not available and cannot be resumed from partial computation.',
            'Incomplete full search cannot establish that p=2 (or any other p) is globally optimal.',
            'No original final Test data were accessed by this continuation script.',
        ],
    }
    (RESULTS / 'fast_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (RESULTS / 'EXP010_fast_summary.md').open('w', encoding='utf-8') as f:
        f.write('# EXP-010 거리 함수 비교 — 중단 후 캐시 결과 정리\n\n')
        f.write('완료된 CSV만 읽었습니다. 추가 KNN 계산은 수행하지 않았습니다.\n\n')
        f.write('| p | PILOT 손글씨 정확도 | FULL 손글씨 정확도 | FULL 공식 정확도 | FULL 상태 |\n')
        f.write('|---:|---:|---:|---:|---|\n')
        for r in rows:
            pil = f"{float(r['pilot_handmade_accuracy']):.2%}" if r['pilot_handmade_accuracy'] != '' else '—'
            hm = f"{float(r['full_handmade_accuracy']):.2%}" if r['full_handmade_accuracy'] != '' else '—'
            off = f"{float(r['full_official_accuracy']):.2%}" if r['full_official_accuracy'] != '' else '—'
            f.write(f"| {r['p']:g} | {pil} | {hm} | {off} | {r['full_status']} |\n")
        f.write('\n**해석:** PILOT과 FULL 성능은 직접 비교할 수 없으며, FULL 미완료 조건에 대한 결론을 내릴 수 없습니다.\n')
        f.write(f'\n**FULL 완료 조건 중 선택:** {f"p={candidate:g}" if candidate is not None else "없음"}\n')
    print('Saved:', RESULTS / 'fast_comparison.csv')
    print('Saved:', RESULTS / 'fast_summary.json')
    print('Saved:', RESULTS / 'EXP010_fast_summary.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
