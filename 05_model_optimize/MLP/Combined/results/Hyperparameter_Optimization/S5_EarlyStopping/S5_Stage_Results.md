# EXP-CMB-008 — S5 Early Stopping 실측·결정

> **단계: S5_early_stopping · 판단: 4방향 5% 목표 미달**
> 기존 기준 False / 선택 False; 이는 개발용 OOF 기반 선택이며 최종 독립 Test의 성능이 아니다.

## 이전 실험 → 이번 실험: 후보를 정한 이유
S4에서 실제 선정된 LR을 고정하고 Training Fold 내부 10%만 사용하는 Early Stopping을 적용했다. 같은 LR의 ES=False 원본 모델을 재사용해 조기 종료 효과 및 과소·과적합을 비교했다.
고정: Official Train의 Size30%/Stroke28% 증강, CoM, 단층 (256,), alpha=1e-4, ReLU/Adam, 3 Seed × 4 Writer CV, 데이터 SHA. S4에서 선정한 LR을 고정하고 ES=False/True만 변경.

## 전체 정확도와 중요 혼동 방향 (실제 클래스당 480회 예측)
| 후보 | OOF 정확도 | OOF 오분류율 | OfficialVal | D→0 | B→8 | 8→B | 0→D | 네 방향 <5% | 비악화·일반화 Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| False | 92.214% | 7.786% | 97.044% | 67 (13.958%) | 43 (8.958%) | 38 (7.917%) | 14 (2.917%) | 미달 | 통과 |
| True | 91.771% | 8.229% | 97.025% | 77 (16.042%) | 49 (10.208%) | 49 (10.208%) | 14 (2.917%) | 미달 | oof_accuracy_at_least_baseline;writer_drop_le_1pp;class_0_8_B_D_error_no_increase;D_to_0_no_increase;B_to_8_no_increase;8_to_B_no_increase |

## 오버피팅·클래스 오류 및 수렴 증거
| 후보 | 실제0 오류 | 실제8 오류 | 실제B 오류 | 실제D 오류 | 최저 Writer OOF | Seed 표준편차 | 최대 OOF Flip | 최대 OfficialVal Flip | 반복횟수 | 경고 Fold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| False | 3.750% | 11.875% | 15.625% | 17.708% | 88.698% | 0.279% | 22.708% | 16.583% | 63.4 | 0 |
| True | 3.750% | 14.167% | 15.833% | 19.167% | 88.594% | 0.622% | 21.042% | 16.236% | 47.2 | 0 |

### 4명의 보류 Writer별 OOF 정확도
| 후보 | W41 | W42 | W43 | W44 | Training loss 초기 평균 | Training loss 마지막 평균 | 평균 Fold 시간(초) |
|---|---:|---:|---:|---:|---:|---:|---:|
| False | 88.698% | 92.135% | 95.312% | 92.708% | 0.594789 | 0.007023 | 65.1 |
| True | 88.594% | 92.396% | 94.219% | 91.875% | 0.619425 | 0.008318 | 43.2 |

### 같은 이미지: 기존 오답 교정과 새 오답
| 후보 | 기존 오답 교정 | 기존 정답 신규 오답 | 순개선 |
|---|---:|---:|---:|
| True | 164 | 198 | -34 |

## 채택·탈락의 명시적 근거
- **이번 단계 채택값: False**.
- **4방향 각각 5% 미만 및 Gate 통과:** 없음. 목표는 아직 달성하지 못했다면 이를 우회해 성공으로 표기하지 않는다.
- **단계별 후보 판정:**
  - **False:** 비교 기준선. D→0 67건·OOF 92.214%·OfficialVal 97.044%.
  - **True:** oof_accuracy_at_least_baseline;writer_drop_le_1pp;class_0_8_B_D_error_no_increase;D_to_0_no_increase;B_to_8_no_increase;8_to_B_no_increase. D→0 77건·OOF 91.771%·OfficialVal 97.025%.

- **S1~S5 이후:** Size30/Stroke28/CoM·(256,)·alpha=1e-4·이번 LR 및 ES 선택값을 잠정 동결. 전체 실험의 개발용 결과로 정리하며 최종 Test는 여전히 미사용.
- **과적합 주의:** 같은 4 Writer를 반복 비교해 선택 편향 가능성이 남는다. Internal Early Stopping과 ConvergenceWarning 무발생만으로 과적합 부재를 증명하지 않는다.
- **데이터 경계:** Original Test·교수님 Test·신규 Hand-made 480장 모두 HP 선택 과정에서 미사용.
