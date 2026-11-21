# EXP-009 | KNN Normal — 획 강도 탐색

> **실험 목적:** 복합 증강 30% 고정 상태에서 획 변형 강도를 비교한다.
> **핵심 결론:** 강도 0.44에서 84.10%; 공식 95.73%로 guardrail 충족.

| 항목 | 내용 |
|---|---|
| ID / 날짜 | EXP-009 / 2026-11-21 |
| 구분 | KNN Normal, 개발용 실험 |
| 실험 코드 | [`EXP009_KNN_Normal_Stroke_Strength.py`](../scripts/EXP009_KNN_Normal_Stroke_Strength.py) |
| 결과 폴더 | [`EXP009_stroke_strength`](../results/EXP009_stroke_strength/) |

## 1. 문제 및 가설

복합 증강 30% 고정 상태에서 획 변형 강도를 비교한다.

## 2. 데이터·실험 조건

Training 76,800 / Validation 19,200; Hand-made development 2,560 (writers 41–44). Original Test 미사용.

- 모든 파생 학습 이미지는 공식 Training에서만 생성하며 Validation 및 Hand-made는 학습에 직접 포함하지 않는다.
- 결과의 공식 Validation / Hand-made는 개발 중 반복 비교된 수치이며 별도 독립 Test 결과가 아니다.
- 아래 수치는 실행 로그를 전사한 기록이며, EXP-005~015의 원본 전체 CSV·개별 예측 파일은 사용자 PC에 보관되어 있다.

- 기본 증강은 EXP-007의 크기·획 변형을 사용하며, 실험 단계에 따라 비율과 강도가 달라진다.

## 3. 실측 결과

| 조건 | 공식 Validation Accuracy | Hand-made Accuracy | Hand-made Macro F1 |
|---|---:|---:|---:|
| `control_com` | 96.5052% | 78.7891% | 0.786540 |
| `size_30` | 96.1198% | 82.2266% | 0.822570 |
| `combined_strength_12` | 95.9427% | 82.5391% | 0.825624 |
| `combined_strength_20` | 95.8698% | 83.2422% | 0.832685 |
| `combined_strength_28` | 95.8073% | 83.0859% | 0.830861 |
| `combined_strength_36` | 95.6979% | 83.2422% | 0.831968 |
| `combined_strength_44` | 95.7344% | 84.1016% | 0.840871 |

![EXP-009 조건별 성능](../results/EXP009_stroke_strength/comparison.png)

**선정 후보:** `combined_strength_44` — 공식 95.7344%, Hand-made 84.1016%, Hand-made Macro F1 0.840871.

## 4. 해석 및 채택 판단

강도 0.44에서 84.10%; 공식 95.73%로 guardrail 충족.

강도 0.44는 공식 95.7344%로 95.70% 기준에 매우 근접한다.

## 5. 재현 및 원본 보존

```bat
python 05_model_optimize/KNN/Normal/scripts/EXP009_KNN_Normal_Stroke_Strength.py
```

- 콘솔 전사 지표: [`console_metrics_transcribed.csv`](../results/EXP009_stroke_strength/console_metrics_transcribed.csv)
- 비교 그래프: [`comparison.png`](../results/EXP009_stroke_strength/comparison.png)
- **추가 첨부 필요:** 사용자 PC의 같은 `results/` 폴더에 생성된 `metrics.json`, `candidate_results.csv`, `by_writer.csv`, `per_class_handmade.csv`, 예측 CSV, 혼동행렬 CSV 등(해당 실험에서 실제 생성된 것만).
- 본 패키지는 전달받지 못한 세부 결과를 합성하지 않았다.
