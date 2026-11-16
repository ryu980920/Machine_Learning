# 가설 B — 입력 위치 의존성과 CoM 정렬·위치 증강

## 검증 목적

MLP가 글자의 **절대 픽셀 위치와 공간 분포 차이**에 민감하여 오분류가 발생하는지 확인하고, CoM 정렬·위치 증강이 이를 줄이는지 검증한다.

1. 이미지 위치를 바꾸면 동일 MLP에서 예측이 변하는가?
2. 학습 데이터와 손글씨 데이터의 중심·크기 분포 차이가 분류 오류와 관련되는가?
3. CoM 정렬은 숫자·문자 정확도를 각각 어떻게 바꾸는가?
4. CoM 이후 ±1px 위치 증강은 문자 정확도 저하를 회복하는가?

[초기분석 EXP유001~002](../초기분석/README.md)에서 입력 변화 민감도의 기초를 검증하였다.

## 실험 목록

- **[EXP유003 — Spatial Response](EXP유003_Spatial_Response.md):** 공식 Training/Validation과 Hand-made의 공간 분포 차이 및 MLP 은닉층·예측 반응.
- **[EXP유004 — Automatic Centering](EXP유004_Automatic_Centering.md):** CoM·Bounding Box 정렬, 고정 모델의 입력 이동과 CoM 재학습 효과 분리.
- **[EXP유005 — CoM Translation Augmentation](EXP유005_CoM_Translation_Augmentation.md):** CoM 정렬 후 ±1px 위치 증강, 동일 샘플 5회 반복 대조.

## 결론

CoM 정렬은 Hand-made **숫자 56.31→83.19%**를 개선했으나 **문자 73.75→66.98%**로 저하시켰다. ±1px 증강에서는 숫자가 **87.00%**, 문자가 **67.08%**였고, 문자는 반복 학습 대조군 **69.48%**에 미달했다.

따라서 **입력 위치 민감도는 확인**했지만, 문자를 숫자로 잘못 구별하는 구체적 판단 근거는 확인하지 못했다. 이 문제는 [가설 A](../가설A/README.md)에서 다룬다.

Hand-made는 이미 여러 차례 확인한 4명 작성자의 개발용 데이터이며 신규 작성자 독립 검증은 하지 않았다.


## 후속 실험 — EXP유006·007 (2026-11-16, 실험 브랜치)

기존 EXP유003~005의 연구 결과·파일은 변경하지 않는다. 다음 실험은 별도 가설 B 하위 로그로 관리하며 최종 모델 채택이 아니라 개발용 비교다.

- **[EXP유006 Size/CoM 사전계획](EXP유006_Size_CoM_Comparison_PLAN.md)** · **[실측 결과](EXP유006_Size_CoM_Comparison_Results.md)**: CoM-only, BBox Size-only, Size→CoM을 동일 공식 Train으로 3 Seed씩 총 9개 MLP 학습 비교. Hand-made 전체 3-seed 평균 CoM-only 77.6953%, Size-only 58.1120%, **Size→CoM 86.9531%**. 공식 Validation에서 Size→CoM은 96.0608%로 하락.
- **[EXP유007 Stroke 결합 사전계획](EXP유007_Size_CoM_Stroke_PLAN.md)** · **[실측 결과](EXP유007_Size_CoM_Stroke_Results.md)**: Size→CoM 대조와 Size→CoM→Stroke(0.30/0.60)→CoM을 3 Seed씩 총 9개 MLP 학습. **Stroke 0.30 결합**이 Hand-made 평균 **87.9688%** 및 공식 Val **96.2656%**로 세 조건 중 최상위.
- **연구 판정:** Size 효과와 추가 Stroke 0.30 효과는 개발 데이터에서 확인. 그러나 공식 Val이 목표 97.30% 미만이고 새 작성자 일반화 검증이 없으므로 **추론 Pipeline 및 main 모델은 그대로 유지**.
- **실행 기록:** [EXP006](https://github.com/ryu980920/Machine_Learning/actions/runs/43785390739) / [EXP007](https://github.com/ryu980920/Machine_Learning/actions/runs/43786098005). 상세 예측·모델·원본 수치는 각각의 GitHub Actions Artifact에 보관(30일).

**주의:** 이 후속 로그·코드·결과 목록은 실험 브랜치 exp/normal-b-exp006-size-com에서 먼저 검토한다. 사용자 승인 전 main에 병합하지 않는다.


## 신규 비교 — EXP유008 (원본 가설 A C9 + Stroke)

- **[EXP유008 사전계획](EXP유008_C9_Stroke_PLAN.md)** · **[실측 결과](EXP유008_C9_Stroke_Results.md)**: 기존 가설 A EXP유004 Stage2 C9 원본 구현을 그대로 복제하고, **C9 단독 / C9→Stroke 0.30→CoM / C9→Stroke 0.60→CoM** 조건을 공식 Train 76,800장에 대해 각 Seed 42·43·44로 학습.
- Seed 평균 Hand-made 전체 **88.6328% / 88.9844% / 89.1146%**, 공식 Val **95.9757% / 95.8976% / 95.8628%**. Stroke를 추가하면 Hand-made는 소폭 개선되지만 공식 Validation은 하락. 일부 Seed와 작성자에서는 악화.
- 원본 C9 구현은 동일하지만 과거 C9 정답 건수를 모든 Seed에서 재현하지 못했으므로, **동일 실행 내 새 C9 대조군**으로만 추가 Stroke 효과를 판단한다.
- [실행](https://github.com/ryu980920/Machine_Learning/actions/runs/43790407107) 성공. 실험 Artifact에 CSV·모델·실제 그림 저장. 최종 모델·main 변경 없음.
