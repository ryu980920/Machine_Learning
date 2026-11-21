# EXP-012 | KNN Normal — HOG 특징 추출

> **실험 목적:** 픽셀 784D, HOG 324D, 50:50 혼합 1108D를 비교한다.
> **핵심 결론:** HOG와 혼합이 손글씨 91.45%; HOG 단독은 더 빠르고 Macro F1 소폭 우수.

| 항목 | 내용 |
|---|---|
| ID / 날짜 | EXP-012 / 2026-11-21 |
| 구분 | KNN Normal, 개발용 실험 |
| 실험 코드 | [`EXP012_KNN_Normal_HOG.py`](../scripts/EXP012_KNN_Normal_HOG.py) |
| 결과 폴더 | [`EXP012_hog`](../results/EXP012_hog/) |

## 1. 문제 및 가설

픽셀 784D, HOG 324D, 50:50 혼합 1108D를 비교한다.

## 2. 데이터·실험 조건

Training 76,800 / Validation 19,200; Hand-made development 2,560 (writers 41–44). Original Test 미사용.

- 모든 파생 학습 이미지는 공식 Training에서만 생성하며 Validation 및 Hand-made는 학습에 직접 포함하지 않는다.
- 결과의 공식 Validation / Hand-made는 개발 중 반복 비교된 수치이며 별도 독립 Test 결과가 아니다.
- 아래 수치는 실행 로그를 전사한 기록이며, EXP-005~015의 원본 전체 CSV·개별 예측 파일은 사용자 PC에 보관되어 있다.

- 기본 증강은 EXP-007의 크기·획 변형을 사용하며, 실험 단계에 따라 비율과 강도가 달라진다.
- 전처리는 CoM 및 Deskew이며, EXP-011에서 적용 순서를 비교했다.
- 특징은 Pixel / HOG / 혼합이며, HOG는 324차원, Pixel은 784차원이다.

## 3. 실측 결과

| 조건 | 공식 Validation Accuracy | Hand-made Accuracy | Hand-made Macro F1 |
|---|---:|---:|---:|
| `pixel_control` | 96.7240% | 86.5234% | 0.864277 |
| `hog_only` | 97.2240% | 91.4453% | 0.914183 |
| `pixel_hog_50_50` | 98.1979% | 91.4453% | 0.913995 |

![EXP-012 조건별 성능](../results/EXP012_hog/comparison.png)

**선정 후보:** `hog_only` — 공식 97.2240%, Hand-made 91.4453%, Hand-made Macro F1 0.914183.

## 4. 해석 및 채택 판단

HOG와 혼합이 손글씨 91.45%; HOG 단독은 더 빠르고 Macro F1 소폭 우수.

HOG 단독 예측시간: 공식 12.0초 + 손글씨 1.7초, 50:50 혼합은 공식 38.9초 + 손글씨 5.2초. 기기·실행 조건에 종속적인 관측치다.

## 5. 재현 및 원본 보존

```bat
python 05_model_optimize/KNN/Normal/scripts/EXP012_KNN_Normal_HOG.py
```

- 콘솔 전사 지표: [`console_metrics_transcribed.csv`](../results/EXP012_hog/console_metrics_transcribed.csv)
- 비교 그래프: [`comparison.png`](../results/EXP012_hog/comparison.png)
- **추가 첨부 필요:** 사용자 PC의 같은 `results/` 폴더에 생성된 `metrics.json`, `candidate_results.csv`, `by_writer.csv`, `per_class_handmade.csv`, 예측 CSV, 혼동행렬 CSV 등(해당 실험에서 실제 생성된 것만).
- 본 패키지는 전달받지 못한 세부 결과를 합성하지 않았다.
