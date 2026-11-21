# EXP-006 | KNN Normal — BB 크기 정규화

> **실험 목적:** 글자의 BB 긴 변을 훈련셋 기준 20픽셀로 정규화하면 유리한가?
> **핵심 결론:** BB 단독은 손글씨 급락; BB+CoM은 공식 정확도 저하. CoM 유지.

| 항목 | 내용 |
|---|---|
| ID / 날짜 | EXP-006 / 2026-11-21 |
| 구분 | KNN Normal, 개발용 실험 |
| 실험 코드 | [`EXP006_KNN_Normal_BB_Size_fixed.py`](../scripts/EXP006_KNN_Normal_BB_Size_fixed.py) |
| 결과 폴더 | [`EXP006_bb_size`](../results/EXP006_bb_size/) |

## 1. 문제 및 가설

글자의 BB 긴 변을 훈련셋 기준 20픽셀로 정규화하면 유리한가?

## 2. 데이터·실험 조건

Training 76,800 / Validation 19,200; Hand-made development 2,560 (writers 41–44). Original Test 미사용.

- 모든 파생 학습 이미지는 공식 Training에서만 생성하며 Validation 및 Hand-made는 학습에 직접 포함하지 않는다.
- 결과의 공식 Validation / Hand-made는 개발 중 반복 비교된 수치이며 별도 독립 Test 결과가 아니다.
- 아래 수치는 실행 로그를 전사한 기록이며, EXP-005~015의 원본 전체 CSV·개별 예측 파일은 사용자 PC에 보관되어 있다.


## 3. 실측 결과

| 조건 | 공식 Validation Accuracy | Hand-made Accuracy | Hand-made Macro F1 |
|---|---:|---:|---:|
| `com` | 96.5052% | 78.7891% | 0.786540 |
| `bbox_size` | 95.1510% | 56.6406% | 0.529048 |
| `size_then_com` | 94.3750% | 79.0625% | 0.785646 |

![EXP-006 조건별 성능](../results/EXP006_bb_size/comparison.png)

**선정 후보:** `com` — 공식 96.5052%, Hand-made 78.7891%, Hand-made Macro F1 0.786540.

## 4. 해석 및 채택 판단

BB 단독은 손글씨 급락; BB+CoM은 공식 정확도 저하. CoM 유지.

BB + CoM이 손글씨 79.06%로 소폭 올랐지만 공식 94.38%로 기준 95.70%를 위반하여 제외했다.

## 5. 재현 및 원본 보존

```bat
python 05_model_optimize/KNN/Normal/scripts/EXP006_KNN_Normal_BB_Size_fixed.py
```

- 콘솔 전사 지표: [`console_metrics_transcribed.csv`](../results/EXP006_bb_size/console_metrics_transcribed.csv)
- 비교 그래프: [`comparison.png`](../results/EXP006_bb_size/comparison.png)
- **추가 첨부 필요:** 사용자 PC의 같은 `results/` 폴더에 생성된 `metrics.json`, `candidate_results.csv`, `by_writer.csv`, `per_class_handmade.csv`, 예측 CSV, 혼동행렬 CSV 등(해당 실험에서 실제 생성된 것만).
- 본 패키지는 전달받지 못한 세부 결과를 합성하지 않았다.
