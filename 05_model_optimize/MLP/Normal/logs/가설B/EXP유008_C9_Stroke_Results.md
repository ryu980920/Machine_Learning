# Normal 가설 B · EXP유008 — 가설 A C9 원본 재현 및 C9→Stroke 결합 실측

> **상태: 9개 MLP 학습·검증·집계 완료 / 최종 Pipeline 미채택**
>
> 실행: https://github.com/ryu980920/Machine_Learning/actions/runs/43790407107
> 실험 브랜치: exp/normal-b-exp008-c9-stroke

## 1. 연구 질문 및 실험 경계

가설 A EXP유004 Stage2의 **C9(BBox 긴 변 20px, 위상 보존, CoM 정렬)** 코드를 직접 재사용하여 Stroke(0.30/0.60) 추가의 효과를 분리한다.

- A: C9 원본 그대로.
- B: C9→Stroke 0.30→CoM 재정렬.
- C: C9→Stroke 0.60→CoM 재정렬.
- 조건별 Seed 42·43·44. 총 **9개 MLP**, 공식 Train 76,800장만 학습, Dev Validation 19,200장과 Hand-made 2,560장 개발 평가.
- **원래 가설 A의 모델 설정** 고정: hidden 100 ReLU, Adam, alpha 0.0001, max_iter=200, tol=0, n_iter_no_change=201, batch_size=auto. EXP유007의 MLP default 기반 학습과 설정이 다르므로 두 실험 간 정확도 차이를 전처리만의 효과로 간주하지 않는다.
- **Official Original Test 16,000장·교수님 Test 사용하지 않음.** Hand-made 학습 사용하지 않음.
- C9 원본 파일 3종(exp_a004_followup.py, exp_a004_official_mlp.py, width_normalization.py)을 가설 A 브랜치에서 **같은 바이트로 복제**하여 소스 차이를 제거.
- C9: threshold 64, 공식 Train 긴 변 중앙값 20px, 등방성 OpenCV bilinear 워프, 배율 0.75~1.25, topology 변화시 최대 5회 강도 완화, CoM 목표 (13.5,13.5), 정수 이동.
- Stroke: 기존 StrokeWidthNormalizer(strength=0.30 또는 0.60, threshold=64, q20/q80, preserve_topology=True). 기준값을 **C9 처리된 공식 Training에만 fit**: q20≈1.82843, q80=3.0px. Stroke 적용 후 CoM 재정렬.
- 세 조건 모두 **전처리 synthetic self-test 성공**.

## 2. 실측 정확도 — Seed별 (%)

| 조건 | Seed | 공식 Validation | Hand-made 전체 | 숫자 | 문자 |
|---|---:|---:|---:|---:|---:|
| C9 원본 | 42 | 96.0469 | 88.3594 | 85.7500 | 92.7083 |
| C9 원본 | 43 | 95.8958 | 88.7891 | 85.9375 | 93.5417 |
| C9 원본 | 44 | 95.9844 | 88.7500 | 86.8125 | 91.9792 |
| C9 + Stroke 0.30 | 42 | 96.0521 | 89.3359 | 86.3125 | 94.3750 |
| C9 + Stroke 0.30 | 43 | 95.8854 | 88.2422 | 85.3125 | 93.1250 |
| C9 + Stroke 0.30 | 44 | 95.7552 | 89.3750 | 87.5625 | 92.3958 |
| C9 + Stroke 0.60 | 42 | 95.9479 | 89.6875 | 88.0000 | 92.5000 |
| C9 + Stroke 0.60 | 43 | 95.6927 | 88.0859 | 85.1875 | 92.9167 |
| C9 + Stroke 0.60 | 44 | 95.9479 | 89.5703 | 87.3750 | 93.2292 |

### 3-seed 평균

| 조건 | 공식 Val | Handmade 전체 | 숫자 | 문자 |
|---|---:|---:|---:|---:|
| **C9 원본** | **95.9757%** | 88.6328% | 86.1667% | 92.7431% |
| **C9 + Stroke 0.30** | 95.8976% | 88.9844% | 86.3958% | **93.2986%** |
| **C9 + Stroke 0.60** | 95.8628% | **89.1146%** | **86.8542%** | 92.8819% |

**같은 실행의 C9 대조군 대비**
- Stroke 0.30: Hand-made 전체 **+0.3516%p**, 숫자 **+0.2292%p**, 문자 **+0.5556%p**, 공식 Val **−0.0781%p**.
- Stroke 0.60: Hand-made 전체 **+0.4818%p**, 숫자 **+0.6875%p**, 문자 **+0.1389%p**, 공식 Val **−0.1128%p**.
- Hand-made 전체 순증가가 모든 Seed에서 일치하는 것은 아니다. Seed43에서는 Stroke0.30이 **−14장**, Stroke0.60이 **−18장**이고, 나머지 42·44에서는 각각 개선됐다.
- 세 방법 모두 공식 Validation 목표 **97.30% 미달**.

## 3. 같은 이미지 전후 대조: 수정과 신규 오답

| 조건 | Seed42 정답 순증 | Seed43 | Seed44 |
|---|---:|---:|---:|
| Stroke0.30 vs C9 | +25 | **−14** | +16 |
| Stroke0.60 vs C9 | +34 | **−18** | +21 |

이는 동일한 손글씨 2,560장에 대한 대응 비교다.

### 주요 혼동 방향 (3-seed 평균 오류 건수)

| 실제→예측 | C9 | C9+Stroke0.30 | C9+Stroke0.60 |
|---|---:|---:|---:|
| B→8 | 7.33 | 8.67 | 7.67 |
| 8→B | 27.67 | 29.67 | 31.33 |
| D→0 | 14.67 | **8.67** | 15.33 |
| 0→D | 11.00 | 14.00 | **9.00** |
| 5→F | 11.00 | 13.33 | 12.67 |
| F→5 | 0.00 | 0.00 | 0.00 |

핵심 6방향 오답 합계는 C9 **71.67건**, Stroke0.30 **74.33건**, Stroke0.60 **76.00건**이다. 전체 Hand-made 정확도는 소폭 개선됐지만, 주요 혼동쌍은 합계 기준 감소하지 않았다. 각 개선·악화 방향을 혼동하지 않는다.

## 4. 작성자별 반례 (전체 정확도, 3-seed 평균 %)

| 작성자 | C9 | Stroke0.30 | Stroke0.60 |
|---|---:|---:|---:|
| 41 | 82.969 | 82.917 | **83.646** |
| 42 | **91.042** | 90.833 | 90.625 |
| 43 | 92.031 | **92.812** | 92.083 |
| 44 | 88.490 | 89.375 | **90.104** |

Writer42는 Stroke 추가 후 성능이 감소한다. 다른 작성자에서 향상되더라도 모든 필체에 일관된 개선이라고 주장하지 않는다.

## 5. 중요한 재현성 확인 및 해석 제한

- 가설 A C9 원본 코드를 그대로 사용했고 동일 과학 패키지 버전을 고정했지만, **원본 C9 역사적 Seed별 정답 수는 완전히 재현되지 않았다**. 같은 코드를 재학습해도 이전 학습과 완전히 같은 예측은 나오지 않았다. 구체적 원인은 이 실험에서 분리하지 못했다.
- 역사적 C9 결과 대비 새로운 C9 정답 수 차이:
  - Seed42: 공식 Val **−16장**, Hand-made **+1장**.
  - Seed43: 공식 Val **+201장**, Hand-made **+59장**.
  - Seed44: 공식 Val **+49장**, Hand-made **+30장**.
- 따라서 **과거 C9 평균 95.5694% 공식 Val / 문자 90.9722%**를 이번 새 Stroke 모델과 직접 빼서 추가 효과라고 주장하지 않는다. 새로 학습한 같은 실행 내 C9 원본을 대조군으로 사용한다.
- Hand-made는 네 작성자 41~44를 반복 연구에 사용한 **개발 데이터**이며 독립 일반화 성능이 아니다.
- 이번 연구의 결과는 **C9+Stroke를 적용·학습할 수 있다는 것과 일부 개발 데이터에서 소폭 이득을 관찰한 것**이다. 공식 Val 하락과 Seed43 역전 때문에 최종 Pipeline 채택은 보류한다.

## 6. 실측 원본과 재현 위치

- 실행: https://github.com/ryu980920/Machine_Learning/actions/runs/43790407107
- 연구 코드: 05_model_optimize/MLP/Normal/scripts/exp008_c9_exact_stroke.py
- 원본 소스 복제: exp_a004_followup.py, exp_a004_official_mlp.py, width_normalization.py
- 3개 조건별 Actions Artifact: normal-exp008-C9_Original, normal-exp008-C9_Stroke030, normal-exp008-C9_Stroke060.
- 종합 Artifact: Normal-HypB-EXP008-complete-review. SUMMARY.md, all_metrics.csv, paired_vs_c9.csv, c9_reference_reproduction.csv.
- 각 조건 Artifact: MLP 3개 checkpoint, run_manifest.json, group_metrics.csv, pair_errors.csv, handmade_predictions.csv, real_handmade_examples.png, fit_metrics.csv.
- Artifact 보존기간 30일. 실험 브랜치에서만 기록하고, 사용자 추가 승인 전에 main·기존 최종 모델은 변경하지 않는다.
