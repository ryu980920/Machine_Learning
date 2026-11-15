# 가설 B · EXP유005 — CoM 기반 ±1px 위치 증강

> **검증 목표:** CoM 중심 정렬 후 학습 이미지를 ±1px 이동해 모델이 위치 변화에 더 견고해지고, 특히 CoM으로 하락한 문자 정확도를 회복하는지 확인한다.
>
> **상태:** 실행·검증 완료. 본 문서는 정식 로그이며, 상세 CSV/학습 후보 모델은 별도 검토용 산출물로 보관한다. **추론 모델 교체 없음.**

## 1. 선행 가설

[EXP유004](EXP유004_Automatic_Centering.md)에서 CoM 학습 MLP는 Hand-made 숫자 **83.1875%**, 문자 **66.9792%**를 기록했다. 위치 변화에 대한 학습을 늘리면 문자 인식 악화를 줄일 것으로 예상했다.

## 2. 통제 조건

| 조건 | 학습 | 평가 |
|---|---|---|
| A: EXP유004 CoM | 기존 학습 모델 고정 | CoM 입력 |
| B: 동일 5회 반복 | CoM 정렬 공식 Train 76,800장 × 5회 = **384,000장** | CoM 입력 |
| C: ±1px 증강 | CoM 정렬 공식 Train 원본 및 좌/우/상/하 1px = **384,000장** | CoM 입력 |

- 동일 16-class MLP(은닉층 100, ReLU/Adam, random_state=42, 픽셀 0~1).
- CoM: 28×28, 전경 임계값 32, 목표 좌표 (13.5,13.5), 정수 이동·zero fill.
- 공식 Training **76,800장**, Validation **19,200장**, Hand-made **2,560장**(작성자 41~44).
- C의 위치 증강은 **Training에서 CoM 적용 이후에만** 사용. 평가 입력에는 증강하지 않음.
- Hand-made 학습·하이퍼파라미터 선택 없음. 공식 Original Test, 교수님 Test 미사용.
- 기존 체크포인트 SHA와 EXP유004 **21,760개 예측** 재현 검증 완료.

## 3. 실측 결과

| 대상 | A: CoM | B: 반복5 | C: ±1px |
|---|---:|---:|---:|
| Official Validation 전체 | 96.9167% | 97.0677% | **97.7865%** |
| Hand-made 전체 | 77.1094% | 78.8281% | **79.5313%** |
| Hand-made 숫자 | 83.1875% | 84.4375% | **87.0000%** |
| Hand-made 문자 | 66.9792% | **69.4792%** | 67.0833% |

B 대비 C에서 Hand-made **전체 +0.7031%p**(178장 교정·160장 신규 오답), **숫자 +2.5625%p**(112장 교정·71장 신규 오답), **문자 −2.3958%p**(66장 교정·89장 신규 오답).

A 대비 C의 숫자 **+3.8125%p**, 문자 **+0.1042%p**(960장 중 정답 1장 증가).

## 4. 대표 오분류

| 실제→예측 | A: CoM | B: 반복5 | C: ±1px |
|---|---:|---:|---:|
| 5→F | 12 | 13 | **7** |
| 4→F | 13 | 9 | **2** |
| B→8 | 79 | 68 | **83** |
| 0→D | 20 | 23 | 26 |

## 5. 실험 한계 및 결론

B/C의 총 학습 샘플 수는 같지만 B **33 epochs·85.13초**, C **108 epochs·274.90초**로 실제 최적화 과정이 달랐다. 단일 seed, 4명 Hand-made 반복 평가, 이미지 가장자리 일부 픽셀 손실 가능성도 한계다.

**결론:** 위치 증강은 숫자와 전체 정확도를 개선했지만 문자는 반복 대조군보다 **2.40%p 낮았다**. 위치 민감도만 해결한다고 문자 오분류가 해결되지는 않는다. 후속 분류 기준 분석은 [가설 A EXP유001](../가설A/EXP유001_픽셀판단근거_분석.md)에서 진행한다.

## 6. 원본·재현 근거

- [실행 코드](../../scripts/CoM_Translation_Augmentation_Experiment.py)
- [실험 성공 실행](https://github.com/ryu980920/Machine_Learning/actions/runs/43691538540) · [워크플로](../../../../../.github/workflows/mlp_exp005_com_translation_augmentation_review.yml)
- 별도 보관: `SUMMARY.md`, `exp005_group_metrics.csv`, `exp005_confusion_matrices.csv`, `exp005_confusion_pairs.csv`, `exp005_paired_transitions.csv`, `exp005_handmade_predictions.csv.gz`, `exp005_learning_curves.csv`, `exp005_metadata.json`, 후보 모델 2종. 이번 GitHub 변경에서는 **산출물 본체를 업로드하지 않았다**.
