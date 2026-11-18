# Combined MLP — EXP-CMB-008 단계별 하이퍼파라미터 연구

[Combined 전체 연구 흐름](../README.md) · [전처리 30/28 선정](../가설B/EXP-CMB-007_03_Final_Size60_100.md)

## 선정된 값과 현재 실측 상태

- **전처리/학습 데이터 고정:** Official Train Size30% / Stroke 강도28%, CoM. Seed 42/43/44 × 4 Writer OOF. Original Test·교수님 Test·신규 손글씨 480장 미사용.
- **S1 너비 실측 완료:** 기본 (100,) OOF 90.716%에서 (256,) OOF **92.214%**, OfficialVal **97.044%**. 취약 혼동 방향도 개선.
- **S2 깊이 실측 완료:** (256,128)/(256,256)은 8→B 등 주요 오분류 악화 → **단층 (256,) 유지**.
- **S3 L2 alpha 실측 완료:** alpha 증가 시 전체 정확도 소폭 증가했지만 D→0 **67→80/86건** 악화 → **alpha=0.0001 유지**.
- **S4/S5 연속 실험 사전등록:** S4 학습률 선택을 마친 후 **실제 선택된 LR**을 S5에 고정 전달하여 Training-only Early Stopping=False/True를 비교하도록 실행 설정. 신규 S4/S5 결과는 학습 완료 전까지 미확정.

## 연구 기록을 읽는 순서

| 번호 | 단계와 목적 | 상태 | 기록 |
|---:|---|---|---|
| 01 | EXP-CMB-008 전체 사전등록과 5% 기준 수정 이력 | 완료 | [전체 연구 계획](EXP-CMB-008_01_Plan_30_28_Frozen_HP.md) |
| 02 | S1 은닉층 너비를 왜 256으로 채택했는가? | 실측 완료 | [S1 후보·탈락 이유](EXP-CMB-008_02_Stage1_Width_Results.md) |
| 03 | S2 두 번째 층 후보를 왜 지정했는가? | 사전등록 | [S2 가설](EXP-CMB-008_03_Plan_S2_Depth.md) |
| 04 | S2 추가 층은 왜 채택하지 않았는가? | 실측 완료 | [S2 결과](EXP-CMB-008_04_Stage2_Depth_Results.md) |
| 05 | S3 L2 alpha를 약·강으로 비교한 이유 | 사전등록 | [S3 가설](EXP-CMB-008_05_Plan_S3_Alpha.md) |
| 06 | S3 규제 강도를 왜 1e-4로 유지했는가? | 실측 완료 | [S3 결과](EXP-CMB-008_06_Stage3_Alpha_Results.md) |
| 07 | S4 Learning Rate 0.0005/0.001/0.002 | 사전등록 | [S4 가설](EXP-CMB-008_07_Plan_S4_LearningRate.md) |
| 08 | S4 정확도·4방향 혼동·과적합 → LR 선정 | 실측 후 생성 예정 | S4 결과 |
| 09 | S5 왜 Early Stopping을 마지막에 비교하는가? | 사전등록 | [S5 가설](EXP-CMB-008_09_Plan_S5_EarlyStopping.md) |
| 10 | S4 선정 LR 고정 → Early Stopping 선택 | 실측 후 생성 예정 | S5 결과 |
| 11 | S1~S5 전체 선정·탈락 근거와 최종 HP | S5 검증 후 생성 예정 | 최종 보고 |

## 통과 기준 — 정확도 우선, 주요 클래스 악화 금지

- **1차 목표:** D→0, B→8, 8→B, 0→D의 실제 클래스 기준 오분류율을 **각각 5% 미만**(실제 클래스당 480회 예측 중 23건 이하).
- 불가능하면 해당 단계의 기준선보다 **OOF Accuracy 비하락, 네 방향별 오류 비증가, 클래스 0/8/B/D 전체 오분류율 비증가, Writer별·OfficialVal·Family Flip 안전 Gate**를 모두 만족하는 최고 정확도 후보. 해당 후보가 없다면 직전 선택값 유지.
- 모든 결과에는 Seed·Writer별 성능, Loss, n_iter, 수렴 경고, 수정/유발 오답을 기록. 같은 4 Writer 반복 선택에 따른 개발용 OOF 과적합 위험을 명시.

[연속 S4→S5 실행 워크플로](../../../../../.github/workflows/exp-cmb-008-s4-s5-chained.yml) · [S4 학습 코드](../../scripts/exp_cmb008_s4_learning_rate.py) · [S5 학습 코드](../../scripts/exp_cmb008_s5_early_stopping.py) · [통합 집계 및 최종 의사결정](../../scripts/exp_cmb008_s4_s5_finalize.py).

<!-- CMB008_S4S5_FINAL_START -->
**S1~S5 완료:** 단층 (256,), alpha=1e-4, LR **0.001**, Early Stopping **False**, OOF **92.214%**, OfficialVal **97.044%**. [S4 결과](EXP-CMB-008_08_Stage4_LearningRate_Results.md) → [S5 결과](EXP-CMB-008_10_Stage5_EarlyStopping_Results.md) → [최종 연구결정](EXP-CMB-008_11_Final_HP_Decision.md). 최종 Test·배포 미변경.
<!-- CMB008_S4S5_FINAL_END -->

## S5 보완: Early Stopping Learning Curve 원인 진단

- **동기:** Early Stopping=True의 OOF가 91.771%로 떨어진 사실만으로 종료 위치 적절성을 판단할 수 없다. 기존 ES=False 결정은 진단 전까지 **잠정**이다.
- [**12 사전등록 — Stop/Best Epoch와 90/100% Training 비교**](EXP-CMB-008_12_Plan_S5_LearningCurves.md)
- [**13 실측 — 5개 실제 Learning Curve PNG/CSV와 Fold별 재현성 감사**](EXP-CMB-008_13_S5_LearningCurves_Results.md) (**36개 학습·그림 생성 완료, 원본 S5 완전 재현 불성립**) 
- [Epoch별 원본 S5 재현 코드](../../scripts/exp_cmb008_s5_learning_curve_trace.py) · [곡선/재현성 검증 코드](../../scripts/exp_cmb008_s5_learning_curve_aggregate.py)
- Stop Epoch와 Validation 최고/복원 Epoch를 구분하고, Epoch별 OOF는 진단에만 사용한다. Original Test·교수님 Test는 계속 봉인한다.

- [**12A 원본 S5 재현성 불일치 기록**](EXP-CMB-008_12A_S5_Reproducibility_Deviation.md): 3/12 Fold만 Stop+예측 정확히 일치, 개별 원본 확인
- [**14 Learning Curve 정량 해석·ES 최종 판단 보류 이유**](EXP-CMB-008_14_S5_LearningCurve_Interpretation.md): H1/H2/H3 가설 결과, D→0 및 4방향 오류 상충, 반복 학습 과적합 위험
- **최종 결론:** ES=False는 개발용 잠정값 유지. 새 재측정 ES10과 기존 S5에 불일치가 있으므로 Early Stopping의 정확한 종료 효과를 인과적으로 단정하지 않는다.

## Same-fit Learning Curve (최신 실측)

- [15 학습 전 실험 계획](EXP-CMB-008_15_Plan_S5_SameFit_TrainVal.md) → [16 실제 Train/Validation Accuracy·Loss 그래프 6개](EXP-CMB-008_16_S5_SameFit_TrainVal_Results.md) → [17 과적합 및 실제 ES 위치 정량 해석](EXP-CMB-008_17_S5_SameFit_Overfitting_Interpretation.md)
- **훈련 Accuracy는 실제 70,848개 전체 Train**, Validation Accuracy는 내부 7,872개를 **같은 모델·동일 Epoch**에 측정했다.
- 학습 120 Epoch에서 Training Accuracy **99.969%**, Validation Accuracy **96.527%**. Validation Log-loss는 10 Epoch 최소 **0.140814** 이후 Epoch 120 **0.219020**으로 악화.
- 해당 동일 실행에서 patience=10이라면 평균 **42.17 Epoch**에 멈추고 Best Validation Epoch 평균 **31.17**. **12/12 Fold에서** 중단 후 더 높은 내부 Validation Accuracy Epoch가 존재했다. 모델 채택은 아직 잠정이며 독립 Test는 사용하지 않았다.

## 발표용 최종 정리 — Early Stopping=False 유지

- [**18 발표용 Learning Curve (16:9 고해상도 Accuracy / Loss 그림)와 ES=False 유지 결정**](EXP-CMB-008_18_S5_Presentation_LearningCurve_ESFalse.md)
- **최종 사용 설정(개발용):** Size30% / Stroke28% / CoM, MLP (256,), alpha=1e-4, LR=.001, **early_stopping=False**, max_iter=200.
- **선정 근거:** 기존 S5에서 False의 Writer OOF **92.214%**가 True **91.771%**보다 높고, D→0은 **67 vs 77건**으로 적었다. 0→D는 양쪽 14건.
- **Learning Curve 해석:** 동일 모델·Epoch의 Training Accuracy 99.969% / 내부 Validation Accuracy 96.527%(120 Epoch), 내부 Validation Loss 증가 → **과적합 징후는 존재**. 이 그림은 별도의 내부 검증 분할(90/10) 진단 실험이므로 ES=False를 학습한 곡선으로 혼동하지 않는다.
- **결정 범위:** Early Stopping의 세부 Patience 재최적화는 진행하지 않고 사용자 요청대로 False를 유지한다. 독립 Test 및 모델 배포는 수행하지 않았다.

## S6 — Batch Size 200/128/64 (09B 강의자료 연계)

- [19 **Batch Size 실험 사전등록**](EXP-CMB-008_19_Plan_S6_Batch_Size.md) → [20 **36개 실측 비교·채택 판단**](EXP-CMB-008_20_Stage6_Batch_Size_Results.md).
- **실측 OOF:** 200=**91.810%**, 128=**92.409%** (순수 정확도 1위), 64=**92.240%**. Batch 128과 64는 주요 혼동·취약 클래스 비악화 Gate 실패 → **기준 Batch 200 유지 추천**.
- **재현성 주의:** 기존 S5 Batch 200 OOF 92.214%와 이번 Batch 200 재실행 OOF 91.810%는 일치하지 않음. 동일 S6 36개 재실행 내에서만 직접 비교하며, 원인 미확정·독립 Test 사용 없음.
- 기존 최종 HP와 모델은 수정하지 않음. Batch Size 변경은 사용자 승인 전 보류.

## S6 후속 — Batch 128 우선 후보 (2026-11-18)

[21번 실측 후속 로그](EXP-CMB-008_21_Stage6_External438_Followup_Results.md): 사용자 품질 판정을 통과한 438장에 대해 Batch 200/128 각각 12모델을 재학습하여 비교. 평균 Accuracy **86.301% vs 87.519%**, Macro F1 **0.8567 vs 0.8690**. Writer OOF 및 OfficialVal도 새 재실행 내에서 Batch 128 우세. 따라서 Batch 128을 **정확도 중심 개발용 우선 후보**로 상향한다.

단, **20번의 S6 사전 가드레일 미통과 기록과 원래 Batch 200 추천은 그대로 보존한다.** 438장은 사후 품질 검토 및 기존 모델 분석에 활용됐으므로 독립 Test가 아니며 새로운 후보 판단은 사후적이다. 추가 전처리는 [CMB-009 72모델 후속 결과](../가설B/EXP-CMB-009_01_Sequential_Preprocessing_Results.md)에서 모두 조합 가드레일 미통과 → Size30/Stroke28/CoM 유지. 최종 배포 모델 변경 없음.
