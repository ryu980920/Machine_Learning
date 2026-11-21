# EXP-011 | KNN Normal — Deskew 기울기 보정

> **실험 목적:** 기울기 보정 전처리 순서가 성능에 주는 영향을 비교한다.
> **핵심 결론:** CoM→Deskew가 공식 96.72%, 손글씨 86.52%로 동시 개선.

| 항목 | 내용 |
|---|---|
| ID / 날짜 | EXP-011 / 2026-11-21 |
| 구분 | KNN Normal, 개발용 실험 |
| 실험 코드 | [`EXP011_KNN_Normal_Deskew.py`](../scripts/EXP011_KNN_Normal_Deskew.py) |
| 결과 폴더 | [`EXP011_deskew`](../results/EXP011_deskew/) |

## 1. 문제 및 가설

기울기 보정 전처리 순서가 성능에 주는 영향을 비교한다.

## 2. 데이터·실험 조건

Training 76,800 / Validation 19,200; Hand-made development 2,560 (writers 41–44). Original Test 미사용.

- 모든 파생 학습 이미지는 공식 Training에서만 생성하며 Validation 및 Hand-made는 학습에 직접 포함하지 않는다.
- 결과의 공식 Validation / Hand-made는 개발 중 반복 비교된 수치이며 별도 독립 Test 결과가 아니다.
- 아래 수치는 실행 로그를 전사한 기록이며, EXP-005~015의 원본 전체 CSV·개별 예측 파일은 사용자 PC에 보관되어 있다.

- 기본 증강은 EXP-007의 크기·획 변형을 사용하며, 실험 단계에 따라 비율과 강도가 달라진다.
- 전처리는 CoM 및 Deskew이며, EXP-011에서 적용 순서를 비교했다.

## 3. 실측 결과

| 조건 | 공식 Validation Accuracy | Hand-made Accuracy | Hand-made Macro F1 |
|---|---:|---:|---:|
| `com_control` | 95.7344% | 84.1016% | 0.840871 |
| `deskew_then_com` | 96.7135% | 86.4062% | 0.863137 |
| `com_then_deskew` | 96.7240% | 86.5234% | 0.864277 |

![EXP-011 조건별 성능](../results/EXP011_deskew/comparison.png)

**선정 후보:** `com_then_deskew` — 공식 96.7240%, Hand-made 86.5234%, Hand-made Macro F1 0.864277.

## 4. 해석 및 채택 판단

CoM→Deskew가 공식 96.72%, 손글씨 86.52%로 동시 개선.

같은 손글씨 예측에서 기존 오답→정답 152, 정답→오답 90, 순증가 62장(최종 CoM→Deskew).

## 5. 재현 및 원본 보존

```bat
python 05_model_optimize/KNN/Normal/scripts/EXP011_KNN_Normal_Deskew.py
```

- 콘솔 전사 지표: [`console_metrics_transcribed.csv`](../results/EXP011_deskew/console_metrics_transcribed.csv)
- 비교 그래프: [`comparison.png`](../results/EXP011_deskew/comparison.png)
- **추가 첨부 필요:** 사용자 PC의 같은 `results/` 폴더에 생성된 `metrics.json`, `candidate_results.csv`, `by_writer.csv`, `per_class_handmade.csv`, 예측 CSV, 혼동행렬 CSV 등(해당 실험에서 실제 생성된 것만).
- 본 패키지는 전달받지 못한 세부 결과를 합성하지 않았다.
