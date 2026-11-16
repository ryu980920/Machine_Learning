#!/usr/bin/env python3
"""Aggregate EXP유004 real 15 fitted MLP outputs. Do not select a Test-tuned policy."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

CONDITIONS = ["C0_CoM", "C1_Stroke015", "C2_Stroke030",
              "C3_GlyphWidth015", "C4_Stroke015_GlyphWidth015"]
SEEDS = [42, 43, 44]
TARGETS = {"Handmade/letters": .80, "Handmade/digits": .85,
           "OfficialVal/all": .973}

def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def save_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main(root):
    records = []
    all_pairs, all_groups = [], []
    for condition in CONDITIONS:
        folder = root / ("normal-a004-" + condition)
        manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest["condition"] != condition or manifest["official_original_test_used"] or manifest["professor_test_used"]:
            raise RuntimeError("Manifest failed for " + condition)
        if sorted(int(r["seed"]) for r in manifest["seed_results"]) != SEEDS:
            raise RuntimeError("Missing fitted seed(s) in " + condition)
        groups = read_csv(folder / "group_metrics.csv")
        pairs = read_csv(folder / "pair_errors.csv")
        if len(pairs) != len(SEEDS) * 6:
            raise RuntimeError("Missing confusion pairs")
        all_groups.extend(groups)
        all_pairs.extend(pairs)
        for seed in SEEDS:
            def find(dataset, group):
                matches = [r for r in groups if int(r["seed"]) == seed and
                           r["dataset"] == dataset and r["group"] == group]
                if len(matches) != 1:
                    raise RuntimeError(f"Ambiguous group {condition} {seed} {dataset}/{group}")
                return float(matches[0]["accuracy"])
            letters = find("Handmade", "letters")
            digits = find("Handmade", "digits")
            official = find("OfficialVal", "all")
            error_sum = sum(int(p["count"]) for p in pairs if int(p["seed"]) == seed)
            records.append({"condition": condition, "seed": seed, "letters": letters,
                            "digits": digits, "official_val": official,
                            "six_directional_errors": error_sum,
                            "pass_all_three": (letters >= .80 and digits >= .85 and official >= .973)})
    save_csv(root / "all_group_metrics.csv", all_groups)
    save_csv(root / "all_pair_errors.csv", all_pairs)
    save_csv(root / "pass_by_seed.csv", records)
    averages = []
    for condition in CONDITIONS:
        rows = [r for r in records if r["condition"] == condition]
        m = {k: mean(r[k] for r in rows) for k in ("letters", "digits", "official_val", "six_directional_errors")}
        s = {k + "_sd": pstdev(r[k] for r in rows) for k in ("letters", "digits", "official_val")}
        passed = (m["letters"] >= .80 and m["digits"] >= .85 and m["official_val"] >= .973)
        consistent = sum(bool(r["pass_all_three"]) for r in rows) >= 2
        averages.append({"condition": condition, **m, **s,
                         "seeds_individual_pass": sum(bool(r["pass_all_three"]) for r in rows),
                         "development_pass": bool(passed and consistent)})
    save_csv(root / "method_summary.csv", averages)
    lines = [
        "# 가설 A · EXP유004 — 공식 Training 재학습 실측 (검토용)",
        "",
        "**모델:** MLPClassifier 100 ReLU hidden nodes, Adam, 공식 Training 76,800장만 학습.",
        "**전처리:** 공식 Training에서만 굵기/가로폭 기준을 fit. 클래스·작성자·정답 사용 없음.",
        "**반복:** 조건별 seed 42, 43, 44. 5조건 × 3회 = 15개 MLP.",
        "**Test:** 공식 Original Test와 교수님 Test는 미사용. Hand-made는 개발용 평가에만 사용.",
        "",
        "통과 판정은 **3개 seed 평균이 문자≥80%, 숫자≥85%, 공식 Validation≥97.30%**이고, **최소 2개 seed가 세 지표를 동시에 충족**해야 한다.",
        "이는 개발 통과이며, 미관찰 작성자 또는 최종 Test 성능을 의미하지 않는다.",
        "",
        "| 조건 | 문자 평균 | 숫자 평균 | 공식 Val 평균 | 6방향 오류 평균 | 개별 통과 seed | 개발 통과 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in averages:
        lines.append(f'| {r["condition"]} | {r["letters"]*100:.3f}% | '
                     f'{r["digits"]*100:.3f}% | {r["official_val"]*100:.3f}% | '
                     f'{r["six_directional_errors"]:.1f} | '
                     f'{r["seeds_individual_pass"]}/3 | '
                     f'{"PASS" if r["development_pass"] else "FAIL"} |')
    lines += ["", "### 해석 범위",
              "- 수치가 높은 후보를 단일 seed로 사후 선택하지 않는다.",
              "- 각 방법의 실제 원본/변형 손글씨 PNG, 개별 seed 체크포인트 및 예측은 Actions Artifact에 보존한다.",
              "- 모델과 학습/추론 전처리 버전은 함께 보관해야 동일 결과를 재현할 수 있다.",
              "- 기존 Hand-made 4명에 대한 반복 분석이므로 최종 독립 Test의 통과는 별도로 필요하다."]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "run_index.json").write_text(json.dumps({
        "conditions": CONDITIONS, "seeds": SEEDS, "trained_models": len(records),
        "official_original_test_used": False, "professor_test_used": False,
        "gate": TARGETS, "results": averages
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines), flush=True)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    main(p.parse_args().root)
