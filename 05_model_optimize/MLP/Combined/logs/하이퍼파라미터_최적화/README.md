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
- [**13 결과 — 5개 Learning Curve PNG/CSV 및 재현성 검증**](EXP-CMB-008_13_S5_LearningCurves_Results.md) (**실험이 검증된 경우 생성**) 
- [Epoch별 원본 S5 재현 코드](../../scripts/exp_cmb008_s5_learning_curve_trace.py) · [곡선/재현성 검증 코드](../../scripts/exp_cmb008_s5_learning_curve_aggregate.py)
- Stop Epoch와 Validation 최고/복원 Epoch를 구분하고, Epoch별 OOF는 진단에만 사용한다. Original Test·교수님 Test는 계속 봉인한다.
