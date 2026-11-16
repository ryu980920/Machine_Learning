# EXP유014 — 크기 기반 Shortcut 완화: Training 분포 균형 증강 실측 비교

> **진행 상태:** 실행 완료 · 실제 3 Seed MLP 학습/분석 성공 · 최종 Pipeline 교체 보류.
> **실행 근거:** https://github.com/ryu980920/Machine_Learning/actions/runs/43810141307

## 연구 목적
EXP유006에서 Size-only 숫자 정확도가 36.46%로 저하되고 Size→CoM으로 84.06%까지 회복했지만, EXP유012에서 크기 개입에 따른 숫자/문자 판단 방향이 크게 달라지는 현상을 확인했다. 새로운 해결 실험은 Size→CoM의 재적용만 반복하지 않고 **학습 입력의 크기–레이블 상관관계 자체를 약화**시키는 것이다.

## 실험군
모든 새 MLP는 **동일 Official Train 76,800장에 대해 1회씩 학습**한다. 각 클래스의 약 50% 이미지는 원본 표현을 사용하고 나머지 약 50%는 클래스와 무관하게 목표 Bounding Box 긴 변 16·18·20·22·24px을 **동일 개수씩** 부여한다. 실제 공식 Train의 클래스별 건수가 홀수일 수 있으므로, 각 클래스에서 변형 대상을 약 절반(5의 배수로 내림)으로 선택하고 5개 목표 크기를 동일 빈도로 배정한다. 선택한 이미지와 목표 크기 할당은 시드 20261116으로 고정하여 모델 학습 Seed 간 동일한 입력을 제공한다.

| 비교 | 새 학습에 사용하는 입력 | 평가 전처리 | 목적 |
|---|---|---|---|
| Baseline CoM (원본) | EXP유006 archived MLP 재사용 | CoM | 크기 학습 Shortcut 대조 |
| **A balanced_scale** | 약 50% CoM-only + 약 50% class-balanced Scale→CoM | CoM | 클래스별 크기 상관관계 억제 |
| **B balanced_scale_stroke** | A + 증강 이미지에 한해 위상 검사를 통과한 약한 획 굵기 변형 | CoM | 크기 단서가 획 굵기/질량으로 대체되는 현상 완화 |
| Baseline Size→CoM (원본) | EXP유006 archived MLP 재사용 | Size→CoM | 기존 최고 수준의 문자 정확도 대조 |
| **C sizecom_balanced_scale** | 약 50% Size→CoM + 약 50% class-balanced Scale→Size→CoM | Size→CoM | 기존 정규화에 크기 변화 학습을 추가 |

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

---

## 실제 실측 결과 (2026-11-16, Seed 42·43·44)

[GitHub Actions 전체 성공 Run #43810141307](https://github.com/ryu980920/Machine_Learning/actions/runs/43810141307) / 원본 Artifact `exp014-seed42`, `exp014-seed43`, `exp014-seed44` 및 `Normal-EXP013-014-measured-shortcut-review`. 모든 표는 **원본 `exp014_accuracy_summary.csv`와 `exp014_family_scale_sensitivity_summary.csv`에서 확인한 3-Seed 평균**이다.

| MLP / 평가 전처리 | OfficialVal 전체 | Hand-made 숫자 | Hand-made 문자 | Hand-made 전체 |
|---|---:|---:|---:|---:|
| 기존 EXP유006 CoM | **97.0521%** | 84.4167% | 66.4931% | 77.6953% |
| **A Class-balanced Scale + CoM** | 95.6962% | **88.3333%** | **89.7569%** | 88.8672% |
| **B Class-balanced Scale+Stroke + CoM** | 95.6007% | **89.3958%** | **88.1944%** | **88.9453%** |
| 기존 EXP유006 Size→CoM | 96.0608% | 84.0625% | **91.7708%** | 86.9531% |
| C Size→CoM + 균형 Scale | 95.8993% | 84.4583% | 90.2778% | 86.6406% |

### 크기 반사실적 개입에 대한 예측 계열 변화

손글씨 동일 이미지의 입력을 바꾸고 각 모델의 **원래 고정 전처리 Pipeline**을 적용한 다음, 숫자 10종의 예측확률 합과 문자 6종의 예측확률 합으로 family 판단을 구성했다. 1.0배와의 판단 계열 차이가 나는 비율:

| 모델 | 실제 숫자 1.2배 확대 시 flip | 실제 문자 0.8배 축소 시 flip |
|---|---:|---:|
| EXP유006 CoM | **71.60%** | **46.08%** |
| A Class-balanced Scale | **8.96%** | **6.74%** |
| B Class-balanced Scale+Stroke | **6.83%** | **8.40%** |
| 기존 Size→CoM | 6.25% | **2.50%** |
| C Size→CoM + 균형 Scale | 6.10% | 3.72% |

기존 CoM 모델에서 숫자를 키우면 문자로, 문자를 줄이면 숫자로 옮겨가던 현상이 A·B 학습으로 크게 완화됐다. 두 조건 모두 Hand-made 숫자/문자 정확도가 CoM 대조보다 높다. **그러나 OfficialVal 정확도는 CoM 대비 A −1.3559%p, B −1.4514%p 저하했다.** A/B/C 어떤 새 모델도 OfficialVal **97.30%** 목표를 만족하지 않았다.

### 해석과 결론

1. **원인 검증:** EXP유013에서 기하학적 크기 하나만으로 원본 OfficialVal 숫자·문자 분리가 99.76% 가능했다. EXP유012에서 기존 고정 MLP의 크기 개입에 따른 판단 방향도 확인한 바 있어, 원본 Training의 source-dependent 크기 차이가 주요 Shortcut 단서라는 해석이 강하게 뒷받침된다.
2. **해결 원리:** 각 클래스에서 목표 크기 16·18·20·22·24px이 동일 비율이 되도록 Training의 약 절반을 리사이즈하면 모델이 특정 크기와 숫자/문자 계열을 연결해 학습하기 어려워진다. 위상 유지 Stroke 변형은 숫자 정확도와 확대 강건성을 다소 개선하지만 문자 성능을 일부 희생한다.
3. **남은 과제:** 공식 Validation이 원본 Train과 같은 출처별 크기 차이를 공유하므로, Shortcut 제거가 공식 Val의 점수 하락과 동반되는지 검토해야 한다. 또한 두 방법이 완전히 Shortcut을 제거했다는 뜻은 아니다. 다른 필기 작성자의 독립 Test는 사용하지 않았고 성능 일반화는 추가 검증 필요.
4. **후속 개선안(미실행):** 크기 균형 증강 비율 20~30%와 50%를 동일 Train 기준으로 비교하여 공식 Val 하락 폭을 줄일 여지 점검. Width/Height 잔존 단서와 Stroke/InkMass 대체 단서를 별도 대조하여 Scale+Stroke 강도를 조절. 최종 모델 채택/배포 전에는 공식 Val·Hand-made 균형과 독립 신규 Writer 평가 조건을 추가 검토.

**최종 판정:** Shortcut 완화 및 Hand-made 숫자/문자 동시 개선은 **실측으로 확인**했다. 그러나 공식 Val 목표 미달이므로 어느 모델도 최종 Pipeline으로 교체하지 않았다. 3개 Seed 모두 같은 평가 프로토콜이고 기존 체크포인트 SHA256과 원본 정답 수를 일치 검증했다.

### 학습 데이터 편향 감소 폭 확인

원본 `exp014-seed42/training_geometry.csv` (세 Seed에서 같은 학습 이미지 구성 확인)에서, 실제 Training 입력의 BBox 긴 변 평균은 다음과 같다.

- 원본 Raw Official Train: **숫자 20.000px / 문자 23.913px** (EXP유013).
- A Class-balanced Scale+CoM Training: **숫자 19.98px / 문자 21.91px**.
- B Scale+Stroke+CoM Training: **숫자 20.41px / 문자 22.12px**.
- C Size→CoM+Balanced Scale Training: **숫자 20.00px / 문자 20.02px**.

**해석:** A·B는 크기–라벨 편향을 완전히 동일하게 만든 것이 아니라, Training 약 절반의 Scale을 계열에 무관하게 배정하여 기존 3.913px 평균 차이를 줄인 방식이다. 실제 최종 입력 분포에 잔여 크기 차이가 남고 이것이 성능에 미치는 영향은 추가 실험 대상이다. C는 긴 변 분포를 거의 같게 하지만 Hands-made의 숫자·문자 동시 개선이 나타나지는 않았다.

B의 topology 보존 Stroke 변형은 증강 대상 이미지 중 **33,359장**이 적용되었으며 **5,016장**은 topology 변화 우려로 원래 증강 표현을 유지했다. 학습 소스 이미지 수는 모든 실험군 **76,800장**으로 동일하다.
