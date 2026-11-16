# Combined MLP — EXP-CMB-008 하이퍼파라미터 최적화 연구 흐름

[전체 Combined 연구 순서](../README.md) · [전처리 연구 CMB-007](../가설B/EXP-CMB-007_03_Final_Size60_100.md)

## 0. 확정된 변경 조건과 현재 단계

- **훈련 증강 고정:** Size30% / Stroke 강도28%, CoM. Original Test·교수님 Test·신규 손글씨 480장 접근 금지.
- **S1 너비(실측 완료):** (64,), (100,), (128,), (256,) 비교. **(256,) OOF 92.214%, OfficialVal 97.044%** 선정.
- **S2 깊이(실측 완료):** (256,128)/(256,256)은 8→B 등 일부 핵심 오분류 악화. 첫 층 256을 고정하고 **단층 (256,) 유지**.
- **S3 L2 alpha(신규 실험 설계 완료):** (256,)의 alpha=1e-4 기준과 alpha=1e-5/1e-3/1e-2 비교. **학습·판정 전** 후보 이유와 안전 기준 사전등록.
- **S4~S5:** S3에서 alpha를 실제 결정한 뒤 Learning Rate → Early Stopping 검증. 아직 후보 미선정.

## 1. 읽는 순서

| 번호 | 단계 | 검증 이유 및 결과 | 상태 |
|---|---|---|---|
| [01](EXP-CMB-008_01_Plan_30_28_Frozen_HP.md) | HP 전체 계획 | 데이터·평가 기준과 4방향 5% 미만 목표의 수정 이력 보존 | 완료 |
| [02](EXP-CMB-008_02_Stage1_Width_Results.md) | S1 은닉층 너비 | 100→256, OOF 90.716→92.214%. 네 혼동 방향 비악화, Writer 전체 개선 | 검증·선정 완료 |
| [03](EXP-CMB-008_03_Plan_S2_Depth.md) | S2 가설 | 단층 vs 추가 층 128/256, 용량 증가의 이득·위험 비교 | 사전등록 완료 |
| [04](EXP-CMB-008_04_Stage2_Depth_Results.md) | S2 결과 | 24개 신규 MLP 실측, 추가 깊이는 핵심 오류 악화 → **단층 유지** | 검증·선정 완료 |
| [05](EXP-CMB-008_05_Plan_S3_Alpha.md) | S3 L2 계획 | 규제를 약하게·강하게 했을 때 D→0 및 과적합 위험 변화 검증 | 사전등록 완료 |
| [06](EXP-CMB-008_06_Stage3_Alpha_Results.md) | S3 L2 결과 | 신규 36개 MLP, OOF·4방향 오류·Writer/Seed·과적합 수렴 비교 | **학습 성공 후 생성 예정** |

## 2. 실제 채택 규칙

**1차 목표:** D→0, B→8, 8→B, 0→D 각각 실제 클래스 내 OOF 오분류율 **5% 미만**. 3 Seed에 실제 클래스당 160장×3=480회 예측이므로 최대 23건.

**불가능한 경우:** 비교 기준선보다 Hand-made OOF 정확도가 떨어지지 않고 **네 방향 오분류 건수가 하나도 증가하지 않으며**, 실제 0/8/B/D 클래스 전체 오류율, Writer별 정확도, OfficialVal, Family Flip이 사전 가드레일을 만족하는 모델 중 최고 정확도 선택. 새 후보가 모두 탈락하면 기존 선택값을 유지한다.

**과적합 감시:** OfficialVal, Seed 분산·Writer별 OOF, 수렴 경고와 Loss, 학습 반복횟수, 동일 이미지 오류 교정/신규 오류를 기록한다. 같은 4 Writer의 OOF를 반복 탐색한 선택 편향은 남아 있으며 아직 최종 Test에 접근하지 않는다.

[원본 S1 CSV/JSON](../../results/Hyperparameter_Optimization/S1_Width/) · [S2 CSV/JSON](../../results/Hyperparameter_Optimization/S2_Depth/) · [S3 재현 코드](../../scripts/exp_cmb008_s3_alpha.py) · [S3 집계 코드](../../scripts/exp_cmb008_s3_aggregate.py).
