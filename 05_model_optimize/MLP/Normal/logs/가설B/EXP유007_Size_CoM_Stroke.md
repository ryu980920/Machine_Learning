# Normal MLP · 가설 B · EXP유007 — Size→CoM→Stroke 결합 비교 (계획·실행 통합 실험일지)

> **현황:** 실제 학습·평가·집계 완료. 최종 모델 변경·채택 보류.
> **실행 근거:** https://github.com/ryu980920/Machine_Learning/actions/runs/43786098005
> **파일 통합:** 이전의 별도 사전계획과 실행 결과를 한 실험일지로 통합했습니다. 아래 '실행 전/결과 검증 전' 표현은 원래 계획을 기록한 당시의 상태이며, 현재 실험은 완료됐습니다.
> **원본 기록:** 사전계획 「Normal MLP · 가설 B · EXP유007 — Size→CoM 이후 Stroke 추가 효과 (사전등록)」 및 결과 「Normal 가설 B · EXP유007 — Size→CoM 이후 Stroke 추가 효과 실측 결과」를 섹션별로 원문 수치·근거·실행 경과를 유지해 통합했습니다.

## 1. 사전 계획 — 실행 전 작성한 기록

> EXP유006 실제 측정 결과를 확인한 후 시작하는 조건부 후속 실험이다.
> 실행 결과를 확인하기 전에는 완료나 성능 향상이라고 표시하지 않는다.

### 선행 결과 및 이유
- EXP유006에서 CoM-only 3-seed 평균 Hand-made 전체 77.695%, 문자 66.493%, 공식 Val 97.052%.
- Size→CoM은 전체 86.953%, 문자 91.771%, 공식 Val 96.061%.
- 따라서 Size의 문자 개선 효과는 확인됐으나 **공식 Validation 약 0.99%p 하락**이 남았다.
- 이번 실험은 Size→CoM을 고정한 상태에서 Stroke를 추가하면 오분류가 어떻게 변하는지 확인한다.

### 정확한 조건 (동일 학습 세팅과 seed)
- A: Size→CoM 대조군 — 다시 학습한다.
- B: Size→CoM→Stroke(강도 0.30)→CoM.
- C: Size→CoM→Stroke(강도 0.60)→CoM.
- 각 조건 Seed 42·43·44, 총 9개 신규 MLP.
- MLPClassifier(random_state=seed), 나머지 기본 설정, 28×28 /255.0.
- Official Train 76,800장만 학습, Official Validation 19,200장 및
  기존 41~44 Hand-made 2,560장은 개발 평가 전용.
- Original Test 16,000장과 교수님 Test는 절대 사용하지 않는다.

### 전처리 기준
- Size: Combined EXP-CMB-004의 BBox 긴 변 20px 등방성 bilinear affine scaling,
  threshold 32, 0-filled padding.
- CoM: threshold 32, 목표 (13.5,13.5), 반올림한 정수 이동.
- Stroke: 기존 가설 A EXP유004 브랜치의 검증된 StrokeWidthNormalizer 소스 그대로 복사.
  threshold 64, Skeleton + Distance Transform으로 두께 측정,
  공식 Training에 있는 획 굵기의 20%·80% 분위수 기준 dilation/erosion,
  원본과 변형을 alpha=0.30 또는 0.60으로 혼합.
- Stroke 정규화 기준은 **Size→CoM 변환된 Official Training만으로 fit**.
  Validation과 Hand-made 레이블 및 이미지 분포는 사용하지 않는다.
- Stroke 이후 CoM을 다시 적용해 획 변경에 따른 중심 좌표 재이동을 통제한다.
- 연결요소/Euler 수가 변하면 Stroke 변형 강도를 완화하는 기존 topology guard 유지.

### 평가
- Official Validation, Hand-made 전체·숫자·문자·4명 작성자 Accuracy/Macro F1.
- B↔8, D↔0, 5↔F 양방향 혼동, 같은 이미지별 오답 교정과 신규 오답.
- 3 seeds에서 개선 방향 일관성, 공식 Val 손실 악화/회복 여부.
- 공식 Validation 97.30% 목표 미달 시 최종 모델 후보 채택은 보류할 수 있다.
- 기존 4명은 반복 분석한 개발 데이터이므로 외부 작성자 일반화 주장은 금지한다.

### 저장·배포
- 코드: 05_model_optimize/MLP/Normal/scripts/exp007_size_com_stroke_followup.py
- 참조 구현: 05_model_optimize/MLP/Normal/scripts/stroke_normalization.py
- 워크플로: .github/workflows/normal-hypb-exp007-size-com-stroke.yml
- Artifact에만 결과 저장, GitHub main 변경/자동 결과 commit 없음.

---

## 2. 실제 실행 및 실측 결과 — 완료 기록

> **상태: 실제 9개 모델 학습·집계 성공 / 최종 채택 보류**
>
> GitHub Actions 성공 실행: https://github.com/ryu980920/Machine_Learning/actions/runs/43786098005

### 1. 실험 목적과 고정 조건

선행 EXP유006에서 Size→CoM이 기존 CoM 대비 Hand-made 문자를 크게 개선했지만 공식 Validation이 하락했다. Stroke 정규화를 같은 Training 기반으로 추가했을 때 숫자·문자 정확도와 공식 Validation 손실이 동시에 개선되는지 확인한다.

- 조건 A: Size→CoM 대조군 **재학습**.
- 조건 B: Size→CoM→Stroke(강도 0.30)→CoM.
- 조건 C: Size→CoM→Stroke(강도 0.60)→CoM.
- 동일 공식 Train **76,800장**·Dev Val **19,200장**, Hand-made 41~44 **2,560장**(개발 전용).
- 각 조건 Seed **42·43·44** = **9개 MLP 새 학습**.
- MLPClassifier(random_state=seed), 기본 하이퍼파라미터, PixelScaler255.
- Size: Combined EXP-CMB-004 구현과 동일 threshold32/BBox 긴 변 20px/Bilinear 등방성 resize.
- CoM: threshold32, 목표 (13.5,13.5), 정수 이동.
- Stroke: 앞선 Normal 가설 A EXP유004의 StrokeWidthNormalizer 소스를 그대로 사용.
  Skeleton/Distance Transform으로 굵기를 측정하고 threshold64의 Training 기반 q20/q80 기준으로 형태를 조정한다. topology guard 적용. **Stroke 기준값은 Size→CoM 처리된 공식 Training에만 fit**.
- Stroke 변경 후 CoM을 한 번 더 적용하여 중심 정렬을 복원한다.
- 모든 Training·Validation·Hand-made에 동일 방법 적용. Hand-made에 대한 학습/하이퍼파라미터 피팅 없음.
- Original Test 16,000장과 교수님 Test **미사용**.

### 2. Seed별 실측 Accuracy (%)

| 조건 | Seed | Official Val | Hand-made 전체 | 숫자 | 문자 |
|---|---:|---:|---:|---:|---:|
| Size→CoM | 42 | 96.0417 | 87.4609 | 85.8750 | 90.1042 |
| Size→CoM | 43 | 96.1823 | 85.9766 | 82.0000 | 92.6042 |
| Size→CoM | 44 | 95.9583 | 87.4219 | 84.3125 | 92.6042 |
| Size→CoM→Stroke 0.30→CoM | 42 | 96.3750 | 88.4375 | 85.4375 | 93.4375 |
| Size→CoM→Stroke 0.30→CoM | 43 | 96.1146 | 87.3828 | 84.4375 | 92.2917 |
| Size→CoM→Stroke 0.30→CoM | 44 | 96.3073 | 88.0859 | 85.5625 | 92.2917 |
| Size→CoM→Stroke 0.60→CoM | 42 | 95.6563 | 86.7188 | 84.3750 | 90.6250 |
| Size→CoM→Stroke 0.60→CoM | 43 | 95.8750 | 87.4219 | 84.2500 | 92.7083 |
| Size→CoM→Stroke 0.60→CoM | 44 | 95.4948 | 88.3984 | 85.6250 | 93.0208 |

#### 3-Seed 평균

| 방법 | Official Val | Hand-made 전체 | 숫자 | 문자 |
|---|---:|---:|---:|---:|
| Size→CoM | 96.0608% | 86.9531% | 84.0625% | 91.7708% |
| **Stroke 0.30 추가** | **96.2656%** | **87.9688%** | **85.1458%** | **92.6736%** |
| Stroke 0.60 추가 | 95.6753% | 87.5130% | 84.7500% | 92.1181% |

**0.30 추가 vs Size→CoM:** Official Val **+0.2049%p**, Hand-made 전체 **+1.0156%p**, 숫자 **+1.0833%p**, 문자 **+0.9028%p** 평균 상승. 단, 모든 Seed·작성자에서 모든 지표가 일률적으로 상승하지는 않았다.

**0.60 추가 vs Size→CoM:** 공식 Val **−0.3854%p**, Hand-made 전체 **+0.5599%p**, 숫자 **+0.6875%p**, 문자 **+0.3472%p**. 전체 성능 개선은 더 작고 공식 Val 저하가 더 컸다.

### 3. 연구 결론과 제한

- **세 가지 전처리 결합은 가능하고 실제 학습에서 검증됐다.** 이번 범위에서 Size→CoM→Stroke 0.30→CoM이 공식 Val과 Hand-made 전체·숫자·문자 평균을 가장 균형 있게 달성했다.
- **공식 Validation 손실은 해결되지 않았다.** 0.30의 공식 Val 96.2656%는 기존 Normal 가설 A의 97.30% 기준보다 낮다. 따라서 최종 추론 Pipeline이나 모델을 교체하지 않는다.
- 결과는 Hand-made 작성자 4명을 반복 검토한 개발용 성능이다. 신규 작성자 데이터에서 일반화를 검증해야 한다.
- 후속 실험을 한다면 공식 데이터 성능을 회복할 수 있는 원본·정규화본 혼합 학습, 동일 노출량/학습 예산 대조, 획 변형 강도별 반례 확인이 합리적이다. 이번 EXP유007에서 혼합 학습은 진행하지 않았다.

### 4. 근거와 보존

- 실험 코드: 05_model_optimize/MLP/Normal/scripts/exp007_size_com_stroke_followup.py
- Stroke 원본: 05_model_optimize/MLP/Normal/scripts/stroke_normalization.py (기존 가설 A 브랜치 소스)
- Actions: https://github.com/ryu980920/Machine_Learning/actions/runs/43786098005
- 9개 모델 Artifact: 각 조건·Seed별 실제 model checkpoint, run_manifest.json, confusion_matrix.csv, pair_errors.csv, loss_curve.csv, 2,560개 행 예측 CSV.gz.
- 종합 Artifact Normal-HypB-EXP007-complete-review: metrics_all.csv, paired_vs_size_com.csv, SUMMARY.md (보존 30일).
- 기존 main의 문서·모델과 추론 Pipeline은 변경하지 않았다.
