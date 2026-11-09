# W02 — 2주차 실험 진행 현황 및 연구계획

- **기간:** 2026-11-09 ~ 2026-11-15
- **기준일:** 2026-11-10
- **현재 상태:** **실험 1~3 완료**(데이터 방향·Grayscale 관련 후속 비교 포함). **Combined 추가 학습, 5개 모델 본격 최적화, 최종 모델 선정 및 독립 Test는 미실시**
- **연구 목표:** MNIST·EMNIST에서 높았던 분류 성능을 새로운 사람의 실제 손글씨에서도 확보할 수 있도록 데이터·모델을 개선한다.
- **관련 문서:** [전체 연구계획](../../01_project/RESEARCH_PLAN.md) · [결정 이력](../../01_project/DECISION_LOG.md) · [데이터 수집·분할안](../../02_data/HANDMADE_SPLIT_PLAN.md) · [W01 기록](../W01_2026-11-02_11-08/README.md)

## 1. 완료한 실험 및 근거

**실험 로그 ID는 W02에서 다시 `EXP유001`부터 시작한다.** 옛 결과 CSV·이미지의 `EXP-유-002` ~ `EXP-유-005`는 과거 식별자를 보존한 파일명이며 결과의 내용·수치는 변경하지 않는다.


| 구분 | 실제 수행 내용 | 상태 | 주요 근거 |
|---|---|---|---|
| **실험 1 — 후보 선정** | 13개 분류 후보를 공식 Training 일부로 빠르게 평가해 **5개 모델 선정** | **완료** | [EXP유001](logs/EXP유001_13개후보_초기평가_5개선정.md), [13개 후보 비교표](results/EXP-유-002_13models_ranked.csv) |
| **실험 2 — Normal Baseline** | 동일한 공식 Train **76,800장**으로 기본 설정의 5개 모델 학습, Validation **19,200장** 평가 | **완료** | [EXP유002](logs/EXP유002_5개모델_Normal_Baseline.md), [5개 모델 비교표](results/EXP-유-003_5models_comparison.csv) |
| **실험 3 — Hand-made 평가** | 실험 2에서 학습한 5개 모델을 **추가 학습 없이** 팀원 손글씨 **2,560장**에 적용. 방향 오류 및 성능 저하 분석 | **완료** | [EXP유003](logs/EXP유003_Handmade_방향보정_성능평가.md), [방향 보정 비교표](results/EXP-유-004/comparison_before_after.csv) |
| **추가 분석 — 최신 데이터 반영** | 새 42번 Grayscale 원본과 구 원본 비교, 새 42번을 반영한 방향 보정 전후 예측 비교 | **완료 (전체 재추론과 구분)** | [EXP유004](logs/EXP유004_42번_Grayscale_원본재제출_비교.md), [EXP유002 8절](logs/EXP유002_5개모델_Normal_Baseline.md) |

### 실험 1 — 13개 후보 평가 및 5개 모델 선정

- 예비 비교 대상 13개 항목: Logistic Regression OvR, Softmax Regression, SGDClassifier, LinearSVC, SVC, KNN, Decision Tree, Random Forest, Extra Trees, AdaBoost, HistGradientBoosting, Perceptron, MLP. SVC 커널 계열은 후보 한 항목으로 관리하며, 13개 항목이 각각 완전히 독립적인 알고리즘 13종이라는 의미는 아니다.
- **공통 예비 실험:** Train **12,000장(클래스당 750장)** / Validation **3,200장(클래스당 200장)**. 원본·증강본 그룹 분리 및 13개 후보의 평가 인덱스 통일.
- Accuracy 90% 이상인 모델과 모델 계열 중복을 고려해 Random Forest를 제외하고 **SVC(RBF), HistGradientBoosting, MLP, Extra Trees, KNN**을 후속 비교 대상으로 확정.
- W01의 LinearSVC·Random Forest·MLP 이론적 우선 검토 결과는 [EXP유001](../W01_2026-11-02_11-08/logs/EXP유001_16class_분류모델_선정.md)에 당시 기록으로 보존한다. 이것을 실험 1의 실측 선정과 혼동하지 않는다.

### 실험 2 — Normal Baseline (기본 설정)

- 사용 데이터: [최종 v3 공식 NPZ](../../02_data/datasets/MNIST_EMNIST_official_split_112000.npz)에서 **공식 Training 96,000장만** 사용. 원본·증강본을 같은 그룹으로 묶는 **StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)의 첫 Fold**로 Train **76,800장** / Validation **19,200장** 고정.
- 입력은 **28×28 Grayscale → 784차원**. SVC·MLP·KNN은 `PixelScaler255`로 픽셀을 `/255` 스케일링. HGB·Extra Trees는 0–255 픽셀 그대로 사용.
- 모델은 기본 하이퍼파라미터(해당 모델의 `random_state=42` 지정)를 사용. **Hand-made 학습은 하지 않음**.

| 모델 | Validation Accuracy | Macro F1 | 입력 스케일 |
|---|---:|---:|---|
| SVC (RBF) | 97.67% | 0.9766 | /255 |
| HistGradientBoosting | 97.46% | 0.9745 | 원본 0–255 |
| MLP | 97.32% | 0.9730 | /255 |
| Extra Trees | 96.97% | 0.9695 | 원본 0–255 |
| KNN | 96.29% | 0.9628 | /255 |

**수치는 기존 실험 2에서 실제 측정한 결과**이며 새 모델 코드로 재학습해 재검증한 값이 아니다. 클래스별 지표·Confusion Matrix·학습/예측 시간·실행 경고는 [EXP유002](logs/EXP유002_5개모델_Normal_Baseline.md)과 [원본 결과 CSV](results/EXP-유-003_5models_comparison.csv) 참조.

### 실험 3 — 손글씨 성능 하락과 입력 데이터 문제

- 팀원 **41·42·43·44번 × 각 640장 = 총 2,560장**을 사용하여, **학습된 동일한 Normal 모델 5개를 재학습하지 않고** 평가했다.
- 과거 구 42번 원본을 사용한 당시 실험에서는 방향 미보정 Accuracy가 **19.88~23.12%**, 방향 보정 후 Accuracy가 **53.24~62.93%**였다. 이는 [EXP유003](logs/EXP유003_Handmade_방향보정_성능평가.md)에 그대로 보존한다.
- 이후 **42번은 재제출된 256종 밝기 Grayscale 원본**으로 교체했다. 현재 표준 전처리: **41번 90° 반시계 회전 / 새 42번 원본 그대로 / 43번 원본 그대로 / 44번 흑백 반전 후 90° 반시계 회전**. 42번의 과거 회전·반전 설명을 현행 입력 처리로 혼용하지 않는다.
- 42번의 구/신 원본은 픽셀 배열까지 달라, 정확도 차이를 밝기 단계 하나의 순수한 인과 효과로 주장하지 않는다.

**현재 표준(신 42번) 기준 방향 보정 이후 비교 결과:**

| 모델 | Hand-made 2,560장 Accuracy |
|---|---:|
| SVC (RBF) | 61.56% |
| HistGradientBoosting | 54.88% |
| MLP | 62.85% |
| Extra Trees | 61.60% |
| KNN | 58.83% |

위 최신 표는 **기존 41·43·44번의 검증된 예측 재사용 + 새 42번의 신규 추론 결과를 결합**한 자료다. 5개 모델×4명 전체를 새로 독립 재실행했다는 뜻은 아니다. 근거: [EXP유002 8절](logs/EXP유002_5개모델_Normal_Baseline.md), [최신 비교 CSV](results/EXP-유-003_rotation_v4/rotation_before_after_5models.csv), [예측 출처 설명](results/EXP-유-003_rotation_v4/README.md), [구/신 42번 비교](logs/EXP유004_42번_Grayscale_원본재제출_비교.md).

**실험을 통해 확인한 문제:** 공식 MNIST·EMNIST Validation에서는 높은 정확도를 보였으나 팀원들이 직접 작성한 Hand-made에서는 큰 하락을 관찰했다. 이미지 방향·극성 등 입력 형식 문제는 일부 확인·보정했지만, **모델 자체의 손글씨 인식 성능 개선은 아직 시작하지 않았다**.

## 2. 교수님 최신 강조사항과 다음 연구계획

**이번 계획부터 4명 전원이 오류 분석·데이터 특성 비교·자동 전처리·증강 실험에 참여**한다. 기존 W02 실험 1~3 수치와 이미지 변환 이력(위 1절)은 그대로 보존하며, 이후 자동 Pipeline 개발이 완료된 것으로 해석하지 않는다.

| 순서 | 실행할 작업 | 공통 조건·완료 근거 | 상태 |
|---|---|---|---|
| **① 4인 공통 오류·데이터 분석** | 팀원 전원이 기존 공식·손글씨 개발 데이터의 오분류 및 중심 위치·기울기·크기·획 굵기·밝기·극성·클래스/작성자 편차 분석 | 실제 이미지/정량 특성·가설·누출 검토, 팀원별 실험일지 | **계획** |
| **② Normal vs Combined 기본 비교** | SVC·HGB·MLP·Extra Trees·KNN 5개에서 **손글씨 추가만 변경**, 같은 모델 설정과 Validation 사용 | 공식 Train 76,800 vs 동일 Train + 기존 Hand-made 학습용 샘플. 개발용 동일 Validation Accuracy·Macro F1·시간 | **계획** |
| **③ 전처리·증강 중심 최적화** | 각 팀원 최소 하나 이상의 자동 Pipeline 또는 Training Augmentation 가설 시험, 이후 5개 모델의 하이퍼파라미터 최적화와 결합 | Raw vs Pipeline, 증강 전후, 개선 및 악화 사례. 전처리 함수·순서·시간·데이터 버전·Seed 기록 | **계획** |
| **④ 최종 모델 선정 및 동결** | 개선된 5개 모델을 공통 Validation으로 비교하고 1개 모델·입력 Pipeline을 선정 | Normal/Combined/Optimized 효과 구분, Accuracy·F1·시간·Learning Curve, 실행 Notebook | **계획** |
| **⑤ 자체 Original Test** | 공식 MNIST·EMNIST Test **16,000장**으로 최종 비교. 신규 독립 손글씨 Test는 **선택 사항** | 최종 설정 이후 평가. Test 기반 튜닝 금지 | **계획** |
| **⑥ 교수님 1차 Hand-made Test** | 완성한 모델·고정 Pipeline으로 전체 성능 확인 | 개별 이미지·클래스별 오분류·분포 분석 금지, 충분한 성능이면 종료 | **계획** |
| **⑦ 교수님 2차 Hand-made Test (조건부)** | 1차가 미흡할 때 1차 자료를 분석하지 않고 **기존 개발 자료만**으로 개선 후 2차 평가 | 2차 점수 보고 후 종료. 1차 충분하면 2차 생략 | **조건부 계획** |

①의 **오류·원인 분석은 ②·③에서도 반복**하지만, ②의 추가 데이터 자체 효과 실험에서는 전처리와 하이퍼파라미터를 고정한다. ③에서 Pipeline과 증강의 기여를 따로 평가한다.

### 팀원별 병렬 수행 — 각자 모두 데이터·오류·Pipeline 연구

팀원별 상세 작업 배분·병렬 일정은 팀장이 직접 작성할 예정이며, 현재는 아직 배정하지 않았다. 오류·데이터 분석, 전처리·증강 실험은 교수님이 강조한 공통 연구 방향이다.

| 공통 수행 | 모든 팀원이 반드시 직접 기록할 사항 |
|---|---|
| **오류 분석** | 자신이 찾은 손글씨 오분류 사례·원인 가설·클래스/작성자별 차이. 모델 오류와 데이터 방향/레이블 오류를 구분 |
| **데이터 수치 비교** | Original vs Hand-made의 관련 픽셀 특징·분포 차이 정량화, 비교 그림·계산 방법 |
| **입력 개선** | 자신이 제안·시험한 자동 Pipeline/Training augmentation, 변경 전·후 입력 예시 |
| **성능 검증** | 같은 Validation 기준 Accuracy·Macro F1·학습/추론·Pipeline 시간, 실패·악화 사례, CSV/Notebook 및 `EXP유001` 형태 주차별 로그 |

**개인별 집중 분야나 담당 모델은 아직 확정하지 않았다.** 팀장이 직접 정할 예정이며, 특정 팀원의 선행·후행 작업을 미리 배정하지 않는다.

## 3. 데이터 구분·평가 안전 규칙

| 데이터 | 실제 수량/상태 | 목적 |
|---|---|---|
| 공식 Training 내부 Train / Validation | **76,800 / 19,200** 확보 | 5개 Normal/Combined 개발 성능 비교 |
| 공식 Original Test | **16,000** 확보 | 완성 모델의 공식 데이터 평가. 교수님 MNIST/EMNIST 구두 분할 설명과 일치 |
| 기존 손글씨 41~44번 | **2,560** 확보 | 개발·오류 분석 및 Combined Training **후보**. 학습용과 검증용 샘플 중복 금지 |
| 신규 손글씨 Validation | **640장 권장·수집 전** | 신규 8명×80장 제안, 작성자 분리 |
| 신규 손글씨 독립 Test | **640장 선택·미확정** | 교수 Test가 별도로 있으므로 의무 아님 |
| 교수님 1차·2차 배포 | 별도 | **둘 다 외부 Test 전용**, 2차는 성능 조건부 |

기존 2,560장 **전체를 Combined에 쓰는 경우에는 다른 개발용 Hand-made Validation을 확보**해야 한다. 신규 Validation이 없다면 기존 손글씨 작성자 홀드아웃을 하되, 홀드아웃한 작성자 이미지는 Combined Training에 넣지 않는다.

**교수님 Test 규칙:** 데이터 자체를 회전·중심화·리사이즈하여 따로 저장하고 개별 사례에 맞추는 일 금지. Test별 이미지·오분류 문자·Confusion Matrix·클래스별 특징 분석 금지. **사전 고정된 자동 Pipeline은 추론 시 적용 가능**하다. 교수님이 명시적으로 허용한 최소 Cleaning 이외 원본 수정 금지. 1차 정확도가 부족해도 1차 자료는 모델 개선 근거로 열람하지 않는다.

자세한 경계와 제안 수량: [HANDMADE_SPLIT_PLAN.md](../../02_data/HANDMADE_SPLIT_PLAN.md).

### 작성자 기준 CV 및 추가 Hand-made (제안·미실시)

- 교수님은 기존 제출분 2,560장과 별도로 새 손글씨를 추가 수집해 Training을 보강하는 방법을 허용했다. 분포 개선 효과를 검증하며 수집량과 사용처는 미확정이다.
- 신규 작성자 Validation이 없다면 기존 2,560장 내 **실제 작성자 기준 Group CV**를 권장한다. 제출 파일 41~44번이 각각 독립된 한 작성자라면 4-Fold(Hand-made 1,920장 Training + 640장 Validation)가 가능하다. 다수 작성자가 한 파일에 있다면 실제 작성자 ID로 그룹을 정한다. CV는 최종 독립 Test가 아닌 개발용 평가다.
- 신규 작성자 Validation을 확보하면 기존 2,560장 전부를 Combined Training에 사용하고 새 Validation으로 평가할 수 있으므로 CV는 필수가 아니다.

## 4. 모델 코드·재현성

[학습 전 모델 5개 및 실행 가이드](../model_optimization/README.md)는 **현재 공식 Training만 대상으로 한 초기 Baseline/Candidate** 코드다. 5개 모델의 기본 설정과 고정 분할을 먼저 재현하고, 향후 Combined 로더·자동 이미지 Pipeline·Training Augmentation을 별도 구현·검증한다. **새 코드에 이 기능이 이미 구현되어 있다는 뜻은 아니다.**

- 입력 이미지의 학습/추론 변환은 **하나의 재사용 가능한 고정 Pipeline**으로 설계한다. Test별 임의 수동 보정이나 정답 기반 조건 분기 금지.
- 결과는 공통 [실험일지 양식](../EXPERIMENT_TEMPLATE.md)에 **문제 → 원인/가설 → 실험 변수·근거 → 전후 성능·시간 → 결론·다음 작업**으로 기록한다.
- 교수님 가이드의 Notebook 및 Learning Curve 요구를 반영한다. MLP는 반복 학습 기록을 활용하고, Epoch 개념이 없는 모델은 학습 크기별 곡선 등 적합한 비교를 실시한다. 가상의 Epoch를 기록하지 않는다.
- 향후 계획의 `계획` 표기는 실행을 확인한 후에만 `완료`로 바꾼다. 기존 실험 로그·CSV·NPZ와 이미 측정된 수치는 변경하지 않는다.
