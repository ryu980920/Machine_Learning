# Normal 가설 B · EXP유006 — Size-only / CoM-only / Size→CoM 실측 결과

> **상태: 실제 학습·집계 완료, 실험 브랜치 보관 / 최종 Pipeline 채택 보류**
>
> GitHub Actions 최종 성공 실행: https://github.com/ryu980920/Machine_Learning/actions/runs/43785390739

## 1. 연구 질문과 방법

Combined EXP-CMB-004에서 사용했던 BBox 긴 변 Size 정규화를 Normal MLP의 같은 공식 Training에 적용했을 때, 단일 Size, 단일 CoM, 결합 Size→CoM 중 어느 방법이 숫자/문자 오류에 유리한가?

- 고정한 공식 Train **76,800장**만 MLP 학습; 공식 Dev Validation **19,200장**, 기존 작성자 41~44 Hand-made **2,560장** 개발 평가.
- 3조건 × Seed 42·43·44 = **9개 MLP 새로 학습**.
- MLPClassifier(random_state=seed), 나머지 기본 설정, PixelScaler255.
- Size: Combined의 exp_cmb004_geometry_normalization.py에서 image_geometry/apply_size 직접 재사용, threshold 32, 공식 Train BBox 긴 변 중앙값 **20px**, bilinear 보간, BBox 중심 고정 등방성 확대·축소.
- CoM: 검증된 center_batch(method=mass), threshold 32, 목표 (13.5,13.5), 정수 이동.
- 방법 A: CoM-only; B: Size-only; C: Size→CoM.
- 모든 조건에서 Training·Validation·Hand-made에 **같은 변환** 적용. Hand-made 학습·Size 목표 결정에 사용하지 않음.
- **Original Test 16,000장과 교수님 Test 모두 미사용.**

## 2. Seed별 정확도 (단위: %)

| 조건 | Seed | 공식 Val | Hand-made 전체 | 숫자 | 문자 |
|---|---:|---:|---:|---:|---:|
| CoM | 42 | 96.9167 | 77.1094 | 83.1875 | 66.9792 |
| CoM | 43 | 97.0469 | 78.2422 | 85.8750 | 65.5208 |
| CoM | 44 | 97.1927 | 77.7344 | 84.1875 | 66.9792 |
| Size | 42 | 96.8906 | 60.5469 | 40.0625 | 94.6875 |
| Size | 43 | 96.7969 | 56.6406 | 34.4375 | 93.6458 |
| Size | 44 | 96.3698 | 57.1484 | 34.8750 | 94.2708 |
| Size→CoM | 42 | 96.0417 | 87.4609 | 85.8750 | 90.1042 |
| Size→CoM | 43 | 96.1823 | 85.9766 | 82.0000 | 92.6042 |
| Size→CoM | 44 | 95.9583 | 87.4219 | 84.3125 | 92.6042 |

### Seed 3개 평균

| 조건 | 공식 Val | Hand-made 전체 | 숫자 | 문자 |
|---|---:|---:|---:|---:|
| CoM-only | **97.0521%** | 77.6953% | **84.4167%** | 66.4931% |
| Size-only | 96.6858% | 58.1120% | 36.4583% | **94.2014%** |
| **Size→CoM** | 96.0608% | **86.9531%** | 84.0625% | 91.7708% |

**Size→CoM vs CoM-only:** Hand-made 전체 **+9.2578%p**, 문자 **+25.2778%p**, 숫자 **−0.3542%p**, 공식 Validation **−0.9913%p**. Hand-made 전체와 문자 개선 방향은 3개 Seed 모두 동일하다.

## 3. 연구 해석

- **Size-only는 최종 후보로 부적절:** 문자는 94.20%까지 개선되지만 숫자가 36.46%로 급락해 전체 성능이 CoM 대비 크게 악화.
- **Size의 추가 효과 확인:** CoM에 Size를 먼저 적용하면 문자 오류가 크게 줄면서 숫자는 평균 0.35%p만 감소해 전체 Hand-made가 86.95%로 향상.
- **공식 Validation 반례:** Size→CoM은 공식 Val 96.06%로 CoM-only 97.05%보다 약 0.99%p 하락. 기존 가설 A 연구 목표였던 공식 Val 97.30%에도 미달.
- 기존 네 작성자를 반복 검토했으므로 외부 작성자 일반화 성능을 확인했다고 주장하지 않는다. 최종 모델은 동결·교체하지 않는다.

## 4. 실행·검증 이력 및 다음 연구

- 최초 실행 https://github.com/ryu980920/Machine_Learning/actions/runs/43785129420 은 seed42 CoM에서 기존 정답 수와 공식 Val **+10장**, Hand-made **+12장** 차이가 발견되어 엄격한 동일성 검사로 중단했다.
- 재실행 https://github.com/ryu980920/Machine_Learning/actions/runs/43785390739 은 **9개 학습·결과 집계 모두 성공**, CoM seed42에서 기존 EXP유004 **18,608/19,200 및 1,974/2,560** 정답 수를 정확히 재현했다. 처음의 차이는 재현성 이력으로 남긴다.
- **수치 근거:** 실행 로그의 실제 COMPLETE 기록, 각 모델의 run_manifest.json, per_class.csv, confusion_matrix.csv, pair_errors.csv, compressed per-image predictions; 집계 all_metrics.csv·paired_vs_com.csv·SUMMARY.md.
- **보관:** 해당 Run의 개별 9개 Artifact 및 Normal-HypB-EXP006-complete-review Artifact (보존 30일).
- **다음:** 합의한 조건부 후속 **가설 B EXP유007**에서 Size→CoM 뒤에 Stroke를 추가해 재정렬/재학습하고 독립적으로 평가한다.
- GitHub main 및 기존 Normal 학습 모델/추론 Pipeline은 변경하지 않았다.
