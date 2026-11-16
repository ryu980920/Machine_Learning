# Normal 가설 B · EXP유007 — Size→CoM 이후 Stroke 추가 효과 실측 결과

> **상태: 실제 9개 모델 학습·집계 성공 / 최종 채택 보류**
>
> GitHub Actions 성공 실행: https://github.com/ryu980920/Machine_Learning/actions/runs/43786098005

## 1. 실험 목적과 고정 조건

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

## 2. Seed별 실측 Accuracy (%)

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

### 3-Seed 평균

| 방법 | Official Val | Hand-made 전체 | 숫자 | 문자 |
|---|---:|---:|---:|---:|
| Size→CoM | 96.0608% | 86.9531% | 84.0625% | 91.7708% |
| **Stroke 0.30 추가** | **96.2656%** | **87.9688%** | **85.1458%** | **92.6736%** |
| Stroke 0.60 추가 | 95.6753% | 87.5130% | 84.7500% | 92.1181% |

**0.30 추가 vs Size→CoM:** Official Val **+0.2049%p**, Hand-made 전체 **+1.0156%p**, 숫자 **+1.0833%p**, 문자 **+0.9028%p** 평균 상승. 단, 모든 Seed·작성자에서 모든 지표가 일률적으로 상승하지는 않았다.

**0.60 추가 vs Size→CoM:** 공식 Val **−0.3854%p**, Hand-made 전체 **+0.5599%p**, 숫자 **+0.6875%p**, 문자 **+0.3472%p**. 전체 성능 개선은 더 작고 공식 Val 저하가 더 컸다.

## 3. 연구 결론과 제한

- **세 가지 전처리 결합은 가능하고 실제 학습에서 검증됐다.** 이번 범위에서 Size→CoM→Stroke 0.30→CoM이 공식 Val과 Hand-made 전체·숫자·문자 평균을 가장 균형 있게 달성했다.
- **공식 Validation 손실은 해결되지 않았다.** 0.30의 공식 Val 96.2656%는 기존 Normal 가설 A의 97.30% 기준보다 낮다. 따라서 최종 추론 Pipeline이나 모델을 교체하지 않는다.
- 결과는 Hand-made 작성자 4명을 반복 검토한 개발용 성능이다. 신규 작성자 데이터에서 일반화를 검증해야 한다.
- 후속 실험을 한다면 공식 데이터 성능을 회복할 수 있는 원본·정규화본 혼합 학습, 동일 노출량/학습 예산 대조, 획 변형 강도별 반례 확인이 합리적이다. 이번 EXP유007에서 혼합 학습은 진행하지 않았다.

## 4. 근거와 보존

- 실험 코드: 05_model_optimize/MLP/Normal/scripts/exp007_size_com_stroke_followup.py
- Stroke 원본: 05_model_optimize/MLP/Normal/scripts/stroke_normalization.py (기존 가설 A 브랜치 소스)
- Actions: https://github.com/ryu980920/Machine_Learning/actions/runs/43786098005
- 9개 모델 Artifact: 각 조건·Seed별 실제 model checkpoint, run_manifest.json, confusion_matrix.csv, pair_errors.csv, loss_curve.csv, 2,560개 행 예측 CSV.gz.
- 종합 Artifact Normal-HypB-EXP007-complete-review: metrics_all.csv, paired_vs_size_com.csv, SUMMARY.md (보존 30일).
- 기존 main의 문서·모델과 추론 Pipeline은 변경하지 않았다.
