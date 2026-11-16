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

## 후속 비교 실험 — EXP유006·007 (계획·결과 통합)

실행 전 사전계획과 학습 완료 후 실측 결과를 **실험 ID별 단일 실험일지**로 통합했다. 기존 EXP유003~005의 원문 기록은 변경하지 않는다.

- **[EXP유006 — Size / CoM 비교 통합 실험일지](EXP유006_Size_CoM_Comparison.md):** CoM-only, Size-only, Size→CoM의 3 Seed × 3조건 실제 학습. Hand-made 전체 CoM-only 77.6953%, Size-only 58.1120%, Size→CoM 86.9531%; Size→CoM 공식 Val 96.0608%. [성공 실행](https://github.com/ryu980920/Machine_Learning/actions/runs/43785390739).
- **[EXP유007 — Size→CoM→Stroke 결합 통합 실험일지](EXP유007_Size_CoM_Stroke.md):** 3 Seed × 3조건 실제 학습. Stroke 0.30 결합은 Hand-made 87.9688%, 공식 Val 96.2656%. [성공 실행](https://github.com/ryu980920/Machine_Learning/actions/runs/43786098005).
- 두 실험 모두 공식 Val 목표 97.30%에 미달하여 최종 Pipeline 채택을 보류했다. Hand-made는 기존 4명의 개발용 데이터다. 코드는 [main 반영 PR #8](https://github.com/ryu980920/Machine_Learning/pull/8)에서 유지하며 모델을 교체하지 않았다.

## 가설 A로 이관된 형태 정규화 후속 실험

획 굵기·C9 크기 정규화 및 원본+C9 학습 실험은 연구 목적에 맞게 **[가설 A](../가설A/README.md)**에서 관리한다.

- [가설 A EXP유008 — 기존 C9 + Stroke 통합 실험일지](../가설A/EXP유008_C9_Stroke.md) · [실제 Actions](https://github.com/ryu980920/Machine_Learning/actions/runs/43790407107).
- [가설 A EXP유009 — 원본+C9 Data Augmentation 통합 실험일지](../가설A/EXP유009_C9_DataAugmentation.md) · [실제 Actions](https://github.com/ryu980920/Machine_Learning/actions/runs/43796627489).

> EXP유008·009는 원래 실행 당시 가설 B로 명명됐으므로 Actions·스크립트·Manifest의 당시 식별자는 재현성을 위해 보존했다. **연구 기록의 최종 분류는 가설 A**다.
