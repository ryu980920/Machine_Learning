# EXP-015 | KNN Normal — HOG 최종 k·가중치 재검증

> **실험 목적:** 최종 특징·증강에 맞는 k 및 가중치를 재탐색한다.
> **핵심 결론:** 12조합 중 k3_distance가 손글씨 Accuracy/F1 최고. EXP-014 채택 설정 유지.

| 항목 | 내용 |
|---|---|
| ID / 날짜 | EXP-015 / 2026-11-21 |
| 구분 | KNN Normal, 개발용 실험 |
| 실험 코드 | [`EXP015_KNN_Normal_K_Weights_HOG.py`](../scripts/EXP015_KNN_Normal_K_Weights_HOG.py) |
| 결과 폴더 | [`EXP015_k_weights_hog`](../results/EXP015_k_weights_hog/) |

## 1. 문제 및 가설

최종 특징·증강에 맞는 k 및 가중치를 재탐색한다.

## 2. 데이터·실험 조건

Training 76,800 / Validation 19,200; Hand-made development 2,560 (writers 41–44). Original Test 미사용.

- 모든 파생 학습 이미지는 공식 Training에서만 생성하며 Validation 및 Hand-made는 학습에 직접 포함하지 않는다.
- 결과의 공식 Validation / Hand-made는 개발 중 반복 비교된 수치이며 별도 독립 Test 결과가 아니다.
- 아래 수치는 실행 로그를 전사한 기록이며, EXP-005~015의 원본 전체 CSV·개별 예측 파일은 사용자 PC에 보관되어 있다.

- 기본 증강은 EXP-007의 크기·획 변형을 사용하며, 실험 단계에 따라 비율과 강도가 달라진다.
- 전처리는 CoM 및 Deskew이며, EXP-011에서 적용 순서를 비교했다.
- 특징은 Pixel / HOG / 혼합이며, HOG는 324차원, Pixel은 784차원이다.
- D/B/5/3 집중 증강 대상은 EXP-013의 Hand-made 개발 데이터 오류 수(D 48, B 41, 5 14, 3 13)에서 선정했다.

## 3. 실측 결과

| 조건 | 공식 Validation Accuracy | Hand-made Accuracy | Hand-made Macro F1 |
|---|---:|---:|---:|
| `k1_uniform` | 97.9062% | 92.2656% | 0.922612 |
| `k1_distance` | 97.9062% | 92.2656% | 0.922612 |
| `k3_uniform` | 98.1510% | 92.7734% | 0.927790 |
| `k3_distance` | 98.1823% | 93.0078% | 0.930121 |
| `k5_uniform` | 98.2760% | 92.7344% | 0.927481 |
| `k5_distance` | 98.2969% | 92.7734% | 0.927879 |
| `k7_uniform` | 98.2135% | 92.5781% | 0.925878 |
| `k7_distance` | 98.2240% | 92.7344% | 0.927419 |
| `k9_uniform` | 98.1510% | 92.8516% | 0.928620 |
| `k9_distance` | 98.1458% | 92.8516% | 0.928648 |
| `k11_uniform` | 98.1146% | 92.6172% | 0.926258 |
| `k11_distance` | 98.1458% | 92.8516% | 0.928695 |

![EXP-015 조건별 성능](../results/EXP015_k_weights_hog/comparison.png)

**선정 후보:** `k3_distance` — 공식 98.1823%, Hand-made 93.0078%, Hand-made Macro F1 0.930121.

## 4. 해석 및 채택 판단

12조합 중 k3_distance가 손글씨 Accuracy/F1 최고. EXP-014 채택 설정 유지.

최종 채택은 EXP-014의 `targeted_10` + k=3 / distance이다. EXP-015는 그 설정을 재현했으며 별도 향상은 없다.

## 5. 재현 및 원본 보존

```bat
python 05_model_optimize/KNN/Normal/scripts/EXP015_KNN_Normal_K_Weights_HOG.py
```

- 콘솔 전사 지표: [`console_metrics_transcribed.csv`](../results/EXP015_k_weights_hog/console_metrics_transcribed.csv)
- 비교 그래프: [`comparison.png`](../results/EXP015_k_weights_hog/comparison.png)
- **추가 첨부 필요:** 사용자 PC의 같은 `results/` 폴더에 생성된 `metrics.json`, `candidate_results.csv`, `by_writer.csv`, `per_class_handmade.csv`, 예측 CSV, 혼동행렬 CSV 등(해당 실험에서 실제 생성된 것만).
- 본 패키지는 전달받지 못한 세부 결과를 합성하지 않았다.
