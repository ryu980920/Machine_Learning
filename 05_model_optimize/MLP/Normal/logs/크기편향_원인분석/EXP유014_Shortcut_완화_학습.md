# EXP유014 — 크기 기반 Shortcut 완화: Training 분포 균형 증강 실측 비교

> **진행 상태:** EXP유013 성공에 이어 조건별 학습·평가를 진행.
> **실행 근거:** https://github.com/ryu980920/Machine_Learning/actions/runs/43809791116

## 연구 목적
EXP유006에서 Size-only 숫자 정확도가 36.46%로 저하되고 Size→CoM으로 84.06%까지 회복했지만, EXP유012에서 크기 개입에 따른 숫자/문자 판단 방향이 크게 달라지는 현상을 확인했다. 새로운 해결 실험은 Size→CoM의 재적용만 반복하지 않고 **학습 입력의 크기–레이블 상관관계 자체를 약화**시키는 것이다.

## 실험군
모든 새 MLP는 **동일 Official Train 76,800장에 대해 1회씩 학습**한다. 각 클래스의 50% 이미지는 원본 표현을 사용하고 나머지 50%는 클래스와 무관하게 목표 Bounding Box 긴 변 16·18·20·22·24px을 **동일 개수씩** 부여한다. 선택한 이미지와 목표 크기 할당은 시드 20261116으로 고정하여 모델 학습 Seed 간 동일한 입력을 제공한다.

| 비교 | 새 학습에 사용하는 입력 | 평가 전처리 | 목적 |
|---|---|---|---|
| Baseline CoM (원본) | EXP유006 archived MLP 재사용 | CoM | 크기 학습 Shortcut 대조 |
| **A balanced_scale** | 50% CoM-only + 50% class-balanced Scale→CoM | CoM | 클래스별 크기 상관관계 억제 |
| **B balanced_scale_stroke** | A + 증강 이미지에 한해 위상 검사를 통과한 약한 획 굵기 변형 | CoM | 크기 단서가 획 굵기/질량으로 대체되는 현상 완화 |
| Baseline Size→CoM (원본) | EXP유006 archived MLP 재사용 | Size→CoM | 기존 최고 수준의 문자 정확도 대조 |
| **C sizecom_balanced_scale** | 50% Size→CoM + 50% class-balanced Scale→Size→CoM | Size→CoM | 기존 정규화에 크기 변화 학습을 추가 |

- Seed **42·43·44**, `MLPClassifier(random_state=seed)` 및 EXP유006 기본 Hyperparameters 유지.
- C는 이미 목표 크기 20px로 재정규화하므로 증강 효과는 주로 리샘플링·획 변화의 강건성으로 나타날 수 있음.
- B의 Stroke perturbation은 클래스 레이블을 사용하지 않고, 기존 원본과 변형의 topology가 달라지면 해당 변형을 취소함. 변형 성공·취소 개수를 기록.
- Seed별 원본 EXP유006 모델 joblib SHA256 및 원본 OfficialVal/Hand-made 실제 정답 수를 검증한 경우에만 비교.
- **Validation·Hand-made 이미지는 임의 증강 학습에 미포함**: OfficialVal 19,200장 및 Hand-made 2,560장의 기본 평가에선 각 모델의 고정 전처리만 적용.

## 평가 질문
1. OfficialVal 전체 정확도, Hand-made 숫자·문자 정확도, 각 클래스 Recall 및 B↔8·D↔0·F↔5 혼동이 대조 모델보다 개선되는가?
2. 동일한 Hand-made 2,560장을 0.8·0.9·1.0·1.1·1.2배로 변형한 뒤 **각 모델에 원래 맞는 고정 전처리**를 적용했을 때 숫자/문자 확률합과 family flip rate가 얼마나 달라지는가?
3. 기존 EXP유009 결과와 비교하여 단순 데이터 노출 증가가 아니라 크기–레이블 관계를 분산한 효과를 확인했는가? 이번 실험은 원천 학습 이미지 수를 **모두 76,800행으로 동일하게 고정**하여 노출 행 수 차이를 방지함.

## 판정 기준
- 기본 성능: OfficialVal 정확도 **97.30%**, Hand-made 숫자 **85%**, 문자 **80%**의 기존 프로젝트 목표를 동시 평가.
- Shortcut 진단: 1.0배 대비 0.8·1.2배에서 절대 숫자/문자 확률 변화량, 예측 계열 flip rate의 감소.
- **성능만 높거나 민감도만 낮아서는 Shortcut을 해결했다고 판정하지 않는다.**
- Hand-made는 동일 4명의 반복 개발 데이터. Original Test 및 교수님 Test는 사용하지 않고, 신규 모델을 main/final에 적용하지 않는다.

## 결과 파일과 재현성
- [학습·평가 스크립트](../../scripts/exp014_balanced_augmentation.py)
- [3-Seed 집계](../../scripts/exp013_014_aggregate.py)
- 원본 각 Seed Artifact: `accuracy_metrics.csv`, `pair_errors.csv`, `scale_metrics.csv`, `scale_image_family_probabilities.csv`, `training_geometry.csv`, `fit_records.csv`, `manifest.json`, 후보 모델 joblib.
- 통합 Artifact: `Normal-EXP013-014-measured-shortcut-review`. 실행 완료 후 수치가 확인되기 전에는 임의 결론을 기록하지 않는다.
