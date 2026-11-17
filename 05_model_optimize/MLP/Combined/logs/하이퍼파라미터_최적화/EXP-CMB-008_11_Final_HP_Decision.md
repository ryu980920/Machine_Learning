# EXP-CMB-008 · 11 S1~S5 최종 HP 연구 결론

> **개발용 OOF/OfficialVal 기준 잠정 선택. Original Test·교수님 Test·신규 손글씨 480장 미사용. 최종 배포 체크포인트 미변경.**

## 단계별 가설 → 실측 → 결정 → 다음 질문

| 단계 | 비교와 선정 이유 | 결정 | 다음으로 넘어간 이유 |
|---|---|---|---|
| S1 너비 | (64,)/(100,)/(128,)/(256,)을 비교 | (256,)의 OOF 92.214%와 주요 오류 개선 | 표현 용량을 확인한 후 깊이 검증 |
| S2 깊이 | (256,128)/(256,256)으로 모델 복잡도 증가 | 두 층에서 8→B 등의 오류 악화 → 단층 (256,) 유지 | 구조를 동결하고 L2 강도 확인 |
| S3 L2 | alpha 1e-5/1e-4/1e-3/1e-2 | 강한 규제에서 D→0 악화 → alpha=1e-4 유지 | L2를 고정하고 최적화 속도 효과 분리 |
| S4 LR | 0.0005 / 0.001 / 0.002 | **0.001 선택** (BASELINE_RETAINED) | LR 고정 후 Early Stopping 효과 검증 |
| S5 종료 | Early Stopping False vs True (Train 내부 10%) | **False 선택** (BASELINE_RETAINED) | HP 잠정 동결 |

## S4 Learning Rate 실측·선정
| 조건 | OOF 정확도 | OfficialVal | D→0 | B→8 | 8→B | 0→D | 4방향 <5% | 악화 Gate |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0.0005 | 92.474% | 97.013% | 63 (13.125%) | 44 (9.167%) | 39 (8.125%) | 16 (3.333%) | 미달 | class_0_8_B_D_error_no_increase, B_to_8_no_increase, 8_to_B_no_increase, 0_to_D_no_increase |
| 0.001 | 92.214% | 97.044% | 67 (13.958%) | 43 (8.958%) | 38 (7.917%) | 14 (2.917%) | 미달 | 없음 |
| 0.002 | 91.771% | 97.027% | 82 (17.083%) | 55 (11.458%) | 37 (7.708%) | 21 (4.375%) | 미달 | oof_accuracy_at_least_baseline, writer_drop_le_1pp, class_0_8_B_D_error_no_increase, D_to_0_no_increase, B_to_8_no_increase, 0_to_D_no_increase |

## S5 Early Stopping 실측·선정
| 조건 | OOF 정확도 | OfficialVal | D→0 | B→8 | 8→B | 0→D | 4방향 <5% | 악화 Gate |
|---|---:|---:|---:|---:|---:|---:|---|---|
| False | 92.214% | 97.044% | 67 (13.958%) | 43 (8.958%) | 38 (7.917%) | 14 (2.917%) | 미달 | 없음 |
| True | 91.771% | 97.025% | 77 (16.042%) | 49 (10.208%) | 49 (10.208%) | 14 (2.917%) | 미달 | oof_accuracy_at_least_baseline, writer_drop_le_1pp, class_0_8_B_D_error_no_increase, D_to_0_no_increase, B_to_8_no_increase, 8_to_B_no_increase |

## 현재 동결할 학습 조건

| 파라미터 | 선정값 |
|---|---|
| Train augmentation | Size 30% × Stroke 강도 28% (Official Train 일부 1:1 대체) |
| Center | CoM |
| Hidden layers | (256,) |
| Solver/activation | Adam / ReLU |
| alpha | 0.0001 |
| learning_rate_init | **0.001** |
| early_stopping | **False** |
| max_iter | 200 |
| OOF Accuracy | **92.214%** |
| OfficialVal Accuracy | **97.044%** |
| D→0 | 67/480 = 13.958% |
| B→8 | 43/480 = 8.958% |
| 8→B | 38/480 = 7.917% |
| 0→D | 14/480 = 2.917% |

**네 방향 각 5% 미만의 목표 미달.** 오분류 비증가와 작성자/Val/Flip 안전 Gate를 통과한 최선의 개발용 대안을 선택했다.
**과적합 한계:** 보류 Writer 4명을 반복 탐색에 사용해 OOF 자체의 선택 편향 가능성은 남는다. ES 내부 10%의 검증은 Training Fold만 사용하며 독립 Test를 대체하지 않는다.

## 추적 가능한 원본

- [S1 너비](EXP-CMB-008_02_Stage1_Width_Results.md)
- [S2 깊이](EXP-CMB-008_04_Stage2_Depth_Results.md)
- [S3 L2](EXP-CMB-008_06_Stage3_Alpha_Results.md)
- [S4 사전등록](EXP-CMB-008_07_Plan_S4_LearningRate.md) → [S4 실측](EXP-CMB-008_08_Stage4_LearningRate_Results.md)
- [S5 사전등록](EXP-CMB-008_09_Plan_S5_EarlyStopping.md) → [S5 실측](EXP-CMB-008_10_Stage5_EarlyStopping_Results.md)
- 원본 실측 CSV/JSON: ../../results/Hyperparameter_Optimization/S4_LearningRate/ 및 S5_EarlyStopping/
