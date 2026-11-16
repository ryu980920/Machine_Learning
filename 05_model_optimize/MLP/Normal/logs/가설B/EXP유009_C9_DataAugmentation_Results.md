# Normal 가설 B · EXP유009 — C9 Data Augmentation 실측 결과

> **실행 완료: 5조건 × 3 Seed = 15개 MLP 실학습·평가 성공**
>
> Run: https://github.com/ryu980920/Machine_Learning/actions/runs/43796627489
>
> 실험 브랜치에만 기록 / 최종 Pipeline 및 main 미변경.

## 1. 연구 질문과 통제

가설 A EXP유004의 C9(BBox 긴 변 20px Size 정규화 → CoM 정렬)를 그대로 사용했다. C9 자체가 Size→CoM이므로, 이전 계획의 Data Augmentation은 **C9 두 번 적용이 아닌 원본+C9 표현을 함께 학습하는 실험**으로 정의했다.

- 학습은 공식 Train 76,800개 고정 원천 이미지, Dev Val 19,200, Hand-made 2,560장(작성자41~44). Official Original Test 및 교수님 Test 미사용, Hand-made 학습 미사용.
- 5조건 × Seed42·43·44 = 총 15개 MLP; hidden 100 ReLU, Adam, alpha 1e-4, lr 1e-3, tol 0, n_iter_no_change 201, batch size auto=200.
- 전처리 C9를 원본 공식 Train에만 fit, 평가에서는 (1) 원본(raw) (2) C9를 **각각 별도 입력으로 측정**.
- 동일 학습 업데이트 예산 통제: 76,800장 200 epochs, 153,600장 100 epochs. 두 경우 모두 명목상 76,800 minibatch steps. 153,600장은 같은 공식 Train 원본을 두 번 표현/반복한 것이지 독립 새 이미지가 아니다.
- 50:50의 원본/C9 선택은 StratifiedShuffleSplit seed=20261116으로 고정해 모든 모델 Seed에 동일 조건 유지.

| 실험군 | 공식 Train 구성 | 학습행 | max_iter |
|---|---|---:|---:|
| A C9-only | C9 정규화만 | 76,800 | 200 |
| B Raw-only | 원본만 | 76,800 | 200 |
| C Raw/C9 50:50 | 각 원본에 원본 또는 C9 중 하나 | 76,800 | 200 |
| D Raw+C9 증강 | 모든 원본과 해당 C9 처리본 | 153,600 | 100 |
| E Raw×2 반복 | 원본 두 번 | 153,600 | 100 |

## 2. C9 전처리한 입력에서의 성능 (3-Seed 평균, %)

| Training | 공식 Val | Handmade 전체 | 숫자 | 문자 |
|---|---:|---:|---:|---:|
| C9-only | 95.5694 | 87.4609 | 85.3542 | 90.9722 |
| Raw-only | 81.6181 | 73.6979 | 87.6875 | 50.3819 |
| Raw/C9 50:50 | 95.2986 | 88.4635 | 86.6875 | 91.4236 |
| **Raw+C9** | **96.0712** | **88.5547** | **86.8333** | **91.4236** |
| Raw×2 | 81.7222 | 73.0078 | 87.6875 | 48.5417 |

D Raw+C9 vs A C9-only:
- 공식 Validation **+0.5018%p**.
- Hand-made 전체 **+1.0938%p**, 숫자 **+1.4791%p**, 문자 **+0.4514%p**.
- 공식 Validation 97.30% 목표까지 여전히 **1.2288%p 부족**.

C 50:50 vs A C9-only:
- 공식 Validation **−0.2708%p**.
- Hand-made 전체 **+1.0026%p**, 숫자 **+1.3333%p**, 문자 **+0.4514%p**.
- 원본을 일부 섞기만 해도 Hand-made는 상승했지만 공식 Val은 하락. D 전체 증강과 구별해야 한다.

## 3. 원본(raw) 입력에서의 성능 (3-Seed 평균, %)

| Training | 공식 Val | Handmade 전체 | 숫자 | 문자 |
|---|---:|---:|---:|---:|
| C9-only | 88.8316 | 73.8151 | 70.6875 | 79.0278 |
| Raw-only | **97.3767** | 62.5781 | 56.1458 | 73.2986 |
| Raw/C9 50:50 | 96.5365 | 74.0755 | 65.2917 | 88.7153 |
| Raw+C9 | 97.0955 | **74.8958** | **65.4583** | **90.6250** |
| Raw×2 | 97.2691 | 62.8385 | 56.8542 | 72.8125 |

같은 153,600 학습행, 명목상 업데이트 수가 같은 D(Raw+C9) vs E(Raw×2)에서 raw 평가 Handmade 전체 +12.0573%p, 숫자 +8.6041%p, 문자 +17.8125%p. 단순 반복보다 입력 다양성을 추가하는 방식이 유리했으나, Raw+C9 모델의 raw 평가 숫자는 65.4583%로 85% 목표에는 크게 부족하다.

Raw-only 모델은 raw 공식 Validation 97.3767%로 공식 목표만 넘지만 Hand-made 숫자 56.1458%, 문자 73.2986%로 동시 통과 실패. C9로 추론 입력을 바꾸면 같은 Raw-only 모델의 공식 Val 81.6181%, 문자 50.3819%로 떨어지는 문제도 확인.

## 4. C9 평가 입력의 문자별 Recall

| 문자 | C9-only | Raw/C9 50:50 | Raw+C9 | D-A 변화 |
|---|---:|---:|---:|---:|
| A | 95.62 | 96.88 | 96.88 | +1.25%p |
| B | 86.67 | 83.96 | 86.04 | −0.63%p |
| C | 92.50 | 94.58 | 95.00 | +2.50%p |
| D | 80.42 | 82.71 | 82.92 | +2.50%p |
| E | 95.21 | 95.00 | 94.17 | −1.04%p |
| F | 95.42 | 95.42 | 93.54 | −1.88%p |

Raw+C9는 C·D에서 개선되었지만 B·E·F에서 악화됐다. 문자 전체 +0.4514%p의 이득은 모든 클래스에서 균일하지 않다.

동일한 Hand-made 이미지(입력 C9)에 대한 C9-only 대 Raw+C9 대응 비교:
- 전체 1 Seed당 평균 교정 **134.33장**, 신규 오답 **106.33장**, 순증 **28장**.
- 숫자 평균 정답 순증 **23.67장**, 문자 순증 **4.33장**.
- 기존 B↔8, D↔0 오분류를 모두 해결했다고 말할 수 없다.

## 5. 작성자별 C9 평가: C9-only → Raw+C9

| Writer | 숫자 | 문자 |
|---|---|---|
| 41 | 78.17→79.00 (+0.83%p) | 89.17→90.56 (+1.39%p) |
| 42 | 89.58→90.58 (+1.00%p) | 90.56→91.81 (+1.25%p) |
| 43 | 90.58→93.08 (+2.50%p) | 91.11→91.39 (+0.28%p) |
| 44 | 83.08→84.67 (+1.59%p) | 93.06→91.94 (−1.12%p) |

숫자는 작성자 41~44 모두 소폭 개선, 문자는 작성자 44가 하락. 반복 사용된 작성자 4명의 개발용 자료이므로 새로운 필체에 대한 일반화 주장 금지.

## 6. 재현성 및 최종 판정

**이번 EXP유009의 C9-only는 가설 A EXP유004 역사적 C9 Seed별 공식 Val·Hand-made 정답 수를 완전히 재현**했다.

| Seed | Official Val Correct /19200 | Handmade Correct /2560 |
|---|---:|---:|
| 42 | 18,457 | 2,261 |
| 43 | 18,211 | 2,214 |
| 44 | 18,380 | 2,242 |

이는 별도 EXP유008에서 기록된 C9 재현 불일치의 원인을 해결했다는 의미는 아니다. 각각의 실험 실행 상태·MLP 입력 구성 차이에 대한 별도 확인이 필요하다.

**최종 판정:** Raw+C9 증강은 C9-only 대비 공식 Val과 Hand-made 숫자·문자 모두 개선했지만, 공식 Val 97.30% 목표 미달. 5조건×2추론 입력 중 3개 지표 동시 통과 조합 없음. **최종 모델 교체는 하지 않고 후속 후보로 보관.**

## 7. 검증 원본

- 성공 Actions: https://github.com/ryu980920/Machine_Learning/actions/runs/43796627489
- 코드: 05_model_optimize/MLP/Normal/scripts/exp009_c9_augmentation.py
- 계획: 05_model_optimize/MLP/Normal/logs/가설B/EXP유009_C9_DataAugmentation_PLAN.md
- 결과 ZIP Artifact: Normal-HypB-EXP009-augmentation-review (SUMMARY.md / three_seed_summary.csv / all_group_metrics.csv / paired_handmade_vs_c9.csv).
- 5개 조건별 Artifact: 각 Seed별 모델 checkpoint, run_manifest.json, group_metrics.csv, pair_errors.csv, handmade_predictions.csv, fit_metrics.csv. 30일 보존.
- GitHub 기록은 exp/normal-b-exp009-c9-data-augmentation 실험 브랜치에만 반영. 추가 승인 전 main 미변경.
