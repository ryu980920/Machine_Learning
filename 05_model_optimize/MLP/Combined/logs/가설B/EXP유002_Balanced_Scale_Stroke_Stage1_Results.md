# EXP유002 / EXP-CMB-005 — 1차(0·10%) 실측 결과와 중간 판정

> **상태: 1차 완료, 2차(20·30%) 진행 중. 최종 연구 완료·모델 채택 아님.**
>
> 실험 전 계획 [Size×Stroke 저비율 사전등록](EXP유002_Balanced_Scale_Stroke_Augmentation.md). 기존 Combined EXP-CMB-001~004는 이미 main에 등록됐으며 이번 연구는 별도의 미병합 Draft PR #16에서 수행 중이다.

## 1. 실제 실행과 불변 통제

- [1차 실측 성공 GitHub Actions #43857354379](https://github.com/ryu980920/Machine_Learning/actions/runs/43857354379). **12개 작업 모두 성공**: 4 전처리 조건×Seed 42,43,44×작성자 41~44 보류 4-Fold = **48개 신규 MLP**.
- 공식 Train 76,800장 + 각 Fold 다른 손글씨 작성자 1,920장. Hand-made OOF 총 2,560장 × 3 Seeds. OfficialVal 19,200장씩 평가. 모든 방법 동일 MLP HP와 train sample 수; CoM 공통.
- 클래스 균형 BBox Size 증강을 **공식 Train의 10%에만** 수행(목표 16·18·20·22·24px), Stroke 0/14/28%로 분기. Hand-made 학습·OOF/OfficialVal에는 별도 Size·Stroke 증강하지 않고 CoM만 적용.
- 0.8/0.9/1.0/1.1/1.2배 입력 개입: 모델 고정, 배율 변화 후 CoM 정렬. 16클래스 전체, 240개 실제→예측 오분류 방향과 writer/seed 데이터 저장. Test 및 Normal EXP유018 신규 480장 미사용.
- 실측 CSV/원본 48-fold 예측/변형 5배율·loss: [stage_low10/SUMMARY.md](../../results/Balanced_Scale_Stroke/stage_low10/SUMMARY.md) · [모든 16클래스 오류율](../../results/Balanced_Scale_Stroke/stage_low10/all_16_class_error_changes.csv) · [240개 오류 방향](../../results/Balanced_Scale_Stroke/stage_low10/all_240_direction_changes.csv).

## 2. 전체/숫자/문자 및 공식 Validation 3-Seed 평균

| 학습 조건 | OOF Accuracy | 숫자 Accuracy | 문자 Accuracy | OfficialVal | 숫자×1.2 Family Flip | 문자×0.8 Family Flip |
|---|---:|---:|---:|---:|---:|---:|
| CoM-only (0%) | 84.883% | 87.750% | 80.104% | **96.733%** | 69.771% | 35.590% |
| Size10 / Stroke0 | **89.935%** | 91.375% | **87.535%** | 96.489% | 26.771% | **11.285%** |
| Size10 / Stroke14 | 88.971% | 90.583% | 86.285% | 96.294% | 23.125% | 11.910% |
| Size10 / Stroke28 | 89.661% | **91.563%** | 86.493% | 96.565% | **18.708%** | 11.875% |

- 같은 MLP HP에서 크기 증강 0→10%는 **Hand-made 일반화와 숫자/문자 양쪽 성능을 개선**했고, 숫자×1.2 Family Flip을 크게 낮췄다.
- 10% 같은 Size 비율에서 Stroke0 vs 14/28은 **추가 변형이 정상 OOF 성능을 일률적으로 높이지 않았다**. 특히 Stroke28은 숫자×1.2 Family Flip이 가장 낮지만 정상 Accuracy는 Stroke0보다 낮다.
- 모든 10% 조건에서 OfficialVal은 CoM-only 대비 **0.168~0.439%p 감소**. 개발 데이터의 Shortcut 완화와 OfficialVal 손실이 함께 발생한다.
- 최악의 **개별 작성자×Seed B/8/D Family Flip은 Size10-Stroke0에서도 67.5%, Stroke28에서도 65.0%**다. 따라서 평균 계열 민감도가 감소했다는 이유만으로 **특정 어려운 클래스에서 Shortcut이 제거되었다고 주장하지 않는다**.

## 3. 16개 클래스·전체 오분류 방향: 선행 원본 대비 관찰

일부 핵심 방향은 별도 편집 없이 [각 조건 전체 오분류 CSV](../../results/Balanced_Scale_Stroke/stage_low10/all_240_direction_changes.csv)에서 확인한다. 아래 오류 수는 **동일한 Hand-made 2,560장에 대한 3-Seed 평균**이고, 원본 Baseline은 역사적 EXP-CMB-001 Seed42 실측이다. 두 비교는 이전 EXP-CMB-001의 전처리 구성도 다르므로 **새 증강의 순수 인과 효과는 CoM-only와 대응 Seed로 비교**해야 한다.

| 실제→예측 | 원본 EXP-CMB-001 | CoM-only 3-Seed | Size10 Stroke0 | Size10 Stroke14 | Size10 Stroke28 |
|---|---:|---:|---:|---:|---:|
| 0→D | 30 | 27.0 | 9.7 | 8.7 | **5.7** |
| D→0 | 29 | 35.0 | **27.0** | 30.0 | 31.0 |
| 8→B | 27 | 21.3 | 10.7 | 10.3 | 10.3 |
| B→8 | 26 | 35.3 | **25.7** | 31.7 | 27.3 |
| 5→F | 17 | 12.0 | 7.0 | **4.3** | **4.3** |
| 7→9 | 8 | 12.0 | 12.3 | 14.3 | **11.0** |
| 9→7 | 15 | 3.0 | 2.3 | 2.0 | 0.7 |

**추가 반례:** `7→9`는 기존 Baseline 8건에서 Size10-Stroke14 **14.3건**, `5→8`은 기존 1건에서 Size10-Stroke0 **5건**으로 증가했다. `C→0`도 기존 2건 대비 Size10-Stroke0 **4.3건**으로 증가했다. 즉 특정 혼동 감소가 모든 클래스에서 개선을 뜻하지 않는다.

## 4. 연구 설명 범위와 다음 단계

- **1차에서 관찰한 사실:** 공식 Training에 적은 비율의 클래스 균형 크기 증강만 추가해도 OOF의 숫자/문자 개선과 Family Flip 감소가 나타났다. Stroke 증가의 효과는 정확도와 Shortcut 지표에서 서로 다르게 나타났다.
- **아직 검증하지 못한 것:** 이 결과가 20%, 30%에서도 유지·포화·역전하는지, 16클래스 최악 오분류율 및 공식 Validation 감소가 어떻게 변하는지, 작용 기전을 얼마나 정량적으로 설명할 수 있는지.
- **2차 실제 학습:** [20/30% GitHub Actions #43871619789](https://github.com/ryu980920/Machine_Learning/actions/runs/43871619789). **결과가 완료되기 전에는 수치 기록·최종 통과 판정하지 않는다.**
- **미변경 사항:** 기존 Combined EXP-CMB-001~004 main 일지, 공식 Original Test, 교수님 Test, Normal 모델과 모든 HP·배포 Pipeline.
