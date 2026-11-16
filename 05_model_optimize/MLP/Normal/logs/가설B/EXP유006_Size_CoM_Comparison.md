# Normal MLP · 가설 B · EXP유006 — Size·CoM 정규화 비교 (계획·실행 통합 실험일지)

> **현황:** 실제 학습·평가·집계 완료. 최종 모델 변경·채택 보류.
> **실행 근거:** https://github.com/ryu980920/Machine_Learning/actions/runs/43785390739
> **파일 통합:** 이전의 별도 사전계획과 실행 결과를 한 실험일지로 통합했습니다. 아래 '실행 전/결과 검증 전' 표현은 원래 계획을 기록한 당시의 상태이며, 현재 실험은 완료됐습니다.
> **원본 기록:** 사전계획 「Normal MLP · 가설 B · EXP유006 — BBox Size / CoM Center 비교 (사전등록)」 및 결과 「Normal 가설 B · EXP유006 — Size-only / CoM-only / Size→CoM 실측 결과」를 섹션별로 원문 수치·근거·실행 경과를 유지해 통합했습니다.

## 1. 사전 계획 — 실행 전 작성한 기록

> 상태: 실행 코드 등록 / 결과 검증 전. **실측 결과로 해석하지 않는다.**

### 연구 질문
Combined EXP-CMB-004에서 Size와 CoM이 서로 다른 숫자/문자 오류 방향에 작용했다.
Normal MLP에서도 Size-only와 Size→CoM을 독립적으로 학습하면,
CoM-only에서 관찰한 문자 인식 하락과 B↔8, D↔0 혼동을 완화하는가?

### 고정 비교
- A: CoM-only (신규 동일 조건 학습 대조)
- B: BBox Size-only (신규)
- C: BBox Size → CoM (신규)
- 기존 Normal 원본 62.8516%와 기존 CoM 77.1094%는 역사적 참조치이며
  새로운 3-seed 비교에서는 A/B/C를 같은 학습 예산으로 재학습해 비교한다.
- **Stroke 정규화는 이 실험에 포함하지 않는다.**
  Size-only 또는 Size→CoM의 추가 효과가 실제 확인된 뒤 별도 후속 실험에서 검토한다.

### 데이터
- 공식 v3 NPZ의 원래 Training 96,000장 내부에서 기존 StratifiedGroupKFold
  첫 Fold: Official Train 76,800 / Official Dev Val 19,200.
- Hand-made 현재 표준 작성자 41·42·43·44 각 640장, 합계 2,560장.
- 손글씨는 MLP 학습·Size 기준 선정에 사용하지 않으며, 반복 탐색된 개발용 평가 데이터다.
- **공식 Original Test 16,000장 및 교수님 Test 미사용**.

### 정확한 전처리 구현
- Combined의 exp_cmb004_geometry_normalization.py에서 image_geometry,
  apply_size 함수를 직접 사용해 구현 차이를 제거한다.
- 크기 기준: Training 이미지의 BBox 긴 변 중앙값 **20px** (실행 시 재검증).
- 전경 threshold=32/255, BBox 중심을 고정한 등방성 affine resize,
  SciPy bilinear order=1, zero padding, 원본 강도 0..255.
- CoM: Normal의 검증된 center_batch(X, "mass") 재사용.
  threshold=32, 무게중심 목표 (13.5,13.5), 정수 픽셀 이동, zero fill.
- 변환 순서 C: **Size 먼저, CoM 나중**.
- 각 조건/seed에서 Training과 Validation/Hand-made 입력에 완전히 동일한 변환 적용.

### 모델·실행
- Pipeline(PixelScaler255(), MLPClassifier(random_state=seed)),
  기타 MLP 파라미터는 sklearn 기본 설정. Seeds **42,43,44**.
- 총 **3개 조건 × 3 seeds = 9개 신규 MLP** 학습.
- seed 42 CoM은 기존 EXP유004의 공식 Val 정답 18,608장 및
  Hand-made 정답 1,974장을 재현하는지 검증.
- GitHub Actions 결과는 오직 Artifact로 수집하며 자동 commit 또는 main 변경하지 않는다.

### 평가·판정
1. 공식 Validation과 Hand-made 전체 Accuracy/Macro F1, 숫자·문자,
   작성자 41~44, 16개 클래스별 Recall을 분리한다.
2. B↔8, D↔0, 5↔F 오분류를 양방향으로 보고하고,
   A 대비 B/C의 같은 이미지 오답→정답 및 정답→오답을 전량 대조한다.
3. 크기 정규화가 긴 변 분산을 낮췄는지, CoM이 중심 오차를 줄였는지,
   보간으로 인해 잉크량과 가장자리 손실이 발생했는지 함께 본다.
4. Size의 추가 효과는 C 대 A의 **문자 오분류 개선·전체 성능·공식 Val 손실**,
   seed 반복성과 작성자별 반례를 함께 고려해 판정한다.
5. Size가 문자 개선에 유의미한 단서를 주더라도 공식 Val 손실이 크면
   최종 Pipeline 채택은 보류하고 후속 Stroke 결합의 연구 후보로만 사용한다.
6. 모든 수치는 기존 4명 개발 데이터 기반. 신규 작성자 독립 검증 전에는
   일반화 개선으로 주장하지 않는다.

### 실행 위치
- 코드: 05_model_optimize/MLP/Normal/scripts/exp006_size_com_comparison.py
- Workflow: .github/workflows/normal-hypb-exp006-size-com.yml
- 리뷰 Artifact: Normal-HypB-EXP006-complete-review 및 9개 개별 모델 Artifact.
- 본 실험 브랜치: exp/normal-b-exp006-size-com.

---

## 2. 실제 실행 및 실측 결과 — 완료 기록

> **상태: 실제 학습·집계 완료, 실험 브랜치 보관 / 최종 Pipeline 채택 보류**
>
> GitHub Actions 최종 성공 실행: https://github.com/ryu980920/Machine_Learning/actions/runs/43785390739

### 1. 연구 질문과 방법

Combined EXP-CMB-004에서 사용했던 BBox 긴 변 Size 정규화를 Normal MLP의 같은 공식 Training에 적용했을 때, 단일 Size, 단일 CoM, 결합 Size→CoM 중 어느 방법이 숫자/문자 오류에 유리한가?

- 고정한 공식 Train **76,800장**만 MLP 학습; 공식 Dev Validation **19,200장**, 기존 작성자 41~44 Hand-made **2,560장** 개발 평가.
- 3조건 × Seed 42·43·44 = **9개 MLP 새로 학습**.
- MLPClassifier(random_state=seed), 나머지 기본 설정, PixelScaler255.
- Size: Combined의 exp_cmb004_geometry_normalization.py에서 image_geometry/apply_size 직접 재사용, threshold 32, 공식 Train BBox 긴 변 중앙값 **20px**, bilinear 보간, BBox 중심 고정 등방성 확대·축소.
- CoM: 검증된 center_batch(method=mass), threshold 32, 목표 (13.5,13.5), 정수 이동.
- 방법 A: CoM-only; B: Size-only; C: Size→CoM.
- 모든 조건에서 Training·Validation·Hand-made에 **같은 변환** 적용. Hand-made 학습·Size 목표 결정에 사용하지 않음.
- **Original Test 16,000장과 교수님 Test 모두 미사용.**

### 2. Seed별 정확도 (단위: %)

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

#### Seed 3개 평균

| 조건 | 공식 Val | Hand-made 전체 | 숫자 | 문자 |
|---|---:|---:|---:|---:|
| CoM-only | **97.0521%** | 77.6953% | **84.4167%** | 66.4931% |
| Size-only | 96.6858% | 58.1120% | 36.4583% | **94.2014%** |
| **Size→CoM** | 96.0608% | **86.9531%** | 84.0625% | 91.7708% |

**Size→CoM vs CoM-only:** Hand-made 전체 **+9.2578%p**, 문자 **+25.2778%p**, 숫자 **−0.3542%p**, 공식 Validation **−0.9913%p**. Hand-made 전체와 문자 개선 방향은 3개 Seed 모두 동일하다.

### 3. 연구 해석

- **Size-only는 최종 후보로 부적절:** 문자는 94.20%까지 개선되지만 숫자가 36.46%로 급락해 전체 성능이 CoM 대비 크게 악화.
- **Size의 추가 효과 확인:** CoM에 Size를 먼저 적용하면 문자 오류가 크게 줄면서 숫자는 평균 0.35%p만 감소해 전체 Hand-made가 86.95%로 향상.
- **공식 Validation 반례:** Size→CoM은 공식 Val 96.06%로 CoM-only 97.05%보다 약 0.99%p 하락. 기존 가설 A 연구 목표였던 공식 Val 97.30%에도 미달.
- 기존 네 작성자를 반복 검토했으므로 외부 작성자 일반화 성능을 확인했다고 주장하지 않는다. 최종 모델은 동결·교체하지 않는다.

### 4. 실행·검증 이력 및 다음 연구

- 최초 실행 https://github.com/ryu980920/Machine_Learning/actions/runs/43785129420 은 seed42 CoM에서 기존 정답 수와 공식 Val **+10장**, Hand-made **+12장** 차이가 발견되어 엄격한 동일성 검사로 중단했다.
- 재실행 https://github.com/ryu980920/Machine_Learning/actions/runs/43785390739 은 **9개 학습·결과 집계 모두 성공**, CoM seed42에서 기존 EXP유004 **18,608/19,200 및 1,974/2,560** 정답 수를 정확히 재현했다. 처음의 차이는 재현성 이력으로 남긴다.
- **수치 근거:** 실행 로그의 실제 COMPLETE 기록, 각 모델의 run_manifest.json, per_class.csv, confusion_matrix.csv, pair_errors.csv, compressed per-image predictions; 집계 all_metrics.csv·paired_vs_com.csv·SUMMARY.md.
- **보관:** 해당 Run의 개별 9개 Artifact 및 Normal-HypB-EXP006-complete-review Artifact (보존 30일).
- **다음:** 합의한 조건부 후속 **가설 B EXP유007**에서 Size→CoM 뒤에 Stroke를 추가해 재정렬/재학습하고 독립적으로 평가한다.
- GitHub main 및 기존 Normal 학습 모델/추론 Pipeline은 변경하지 않았다.
