# W02 — 2주차 연구계획 (교수님 피드백 반영)

- **기간:** 2026-11-09 – 2026-11-15
- **상태:** **A단계(13개 모델 평가), B단계(Normal Baseline), C단계(Hand-made 평가·방향 보정) 완료 / D–F는 미실시**
- **근거:** 2026-11-09 팀장이 전달한 교수님 피드백. W01의 이론 검토만으로 최종 모델을 확정하지 않고 **실측 데이터로 후보군을 좁힌 뒤 최적화 결과로 최종 선택**한다.
- **이전 기록:** [W01 작업 및 결과](../W01_2026-11-02_11-08/README.md) · [W01 모델 검토 로그](../W01_2026-11-02_11-08/logs/EXP-유-001_16class_분류모델_선정.md)

## 1. 연구 목표와 변경 사항

**13개 후보 기본 평가(완료) → 5개 모델 Normal 학습·Validation → Hand-made 인식률 저하 확인 → 데이터·모델 원인 분석 → 공통 Preprocessing 및 5개 모델별 Hyperparameter 개선 → Combined Dataset 학습 효과 비교 → 필요 시 상위 2개 동시 Fine-tuning → 최종 1개 선정.**

W01에서 검토한 **13개 후보 항목 모두를 초기 예비 평가 대상으로 확정**한다. 이전에 정한 **LinearSVC, RandomForestClassifier, MLPClassifier**에 **KNeighborsClassifier(KNN)**를 추가한 조합은 당시의 **우선 검토 4종**으로 두었다. 이는 **실측 전 잠정 후보**였다. 2026-11-09에 13개 후보를 실제 평가한 뒤 **Accuracy 90% 이상 + Random Forest 계열 중복 제외** 기준으로 최종 비교 대상 **5개(SVC RBF, HistGradientBoosting, MLP, Extra Trees, KNN)**를 선정했다. W01의 이론적 선정 결과를 실제 성능 평가 완료로 간주하지 않는다.

| 이전 W02 계획 | 수정된 방향 |
|---|---|
| 사전 지정 3개 모델의 기본 성능 확인 | **W01의 13개 후보를 모두 간단히 평가**한 수치로 **5개 모델 선정** |
| 초기 성능에 따라 1개 모델 우선 선정 | **선정된 5개 모두** 전처리·Hyperparameter 개선 실험 |
| Accuracy 중심의 러프한 비교 | 성능 외에 **추론 시간, 수렴 여부, 반복 실행 안정성** 확인 |
| 최적화는 선정 후에 1개에 집중 | **최적화 후 최종 1개 선정**; 상위 2개가 근접하면 **동시 Fine-tuning** |
| Hand-made를 전부 테스트 전용으로 제한 | 교수님 요구에 맞춰 **Normal Dataset vs Combined Dataset** 학습 비교. 단, 추가 학습과 평가는 겹치지 않게 분할 |

## 2. 단계별 실험 계획과 완료 기준

### 단계 A — 기존 13개 후보 전수 빠른 예비 평가 (범위 확정)

**예비 평가 대상은 W01에서 이론적으로 검토한 아래 13개 항목 모두로 확정**한다. 이 중 Logistic Regression OvR·Softmax는 같은 선형 계열의 서로 다른 방식이고, SVC의 여러 커널은 하나의 후보 항목에 묶여 있다. 따라서 **13개 항목 ≠ 서로 독립적인 알고리즘 13종**이다.

| 번호 | W01의 예비 평가 후보 | 예비 실험 메모 |
|---:|---|---|
| 01 | Logistic Regression + OvR | `OneVsRestClassifier` 등 구현 방식 기록 |
| 02 | Softmax Regression | 다중 클래스 Logistic Regression의 공동 확률 모델 |
| 03 | SGDClassifier | 사용 `loss`와 반복 제한을 명시 |
| 04 | LinearSVC | 수렴 경고, `max_iter` 기록 |
| 05 | SVC (linear / polynomial / RBF) | **한 후보 항목**; 초기에는 대표 커널 하나를 정해 기록하고, 다른 커널 비교 시 별도 실행 행으로 기록. 계산 시간에 유의 |
| 06 | KNeighborsClassifier (KNN) | `fit`보다 **예측 시간** 및 메모리 사용을 중요하게 기록 |
| 07 | DecisionTreeClassifier | 단일 트리 기본 성능·과적합 여부 |
| 08 | RandomForestClassifier | 트리 수와 학습/예측 시간 |
| 09 | ExtraTreesClassifier | Random Forest와 비교 |
| 10 | AdaBoostClassifier | 약한 학습기 수·시간 |
| 11 | HistGradientBoostingClassifier | 반복 수·학습 시간·실행 예산 |
| 12 | Perceptron | 반복 수·수렴 기록 |
| 13 | MLPClassifier | 반복 수·수렴 경고·입력 스케일 |

**1차 평가의 공통 조건 (2026-11-09 실제 실행, 증빙: EXP-유-002)**

1. **데이터와 누출 방지:** 최종 v3 공식 Training 96,000장을 **원본+증강본 그룹 단위로 먼저** Train/Validation으로 나누고 그 후 계층화·균형화된 소규모 표본을 추출한다. 예비 실험은 **Train 12,000장(클래스당 750장) / Validation 3,200장(클래스당 200장)**을 사용했으며 원본·증강본 그룹 누출 0을 확인했다. 13개 후보는 **완전히 동일한 샘플 인덱스**를 사용한다.
2. **최소 공통 전처리:** 모든 모델은 28×28 → 784개 특성의 `float32` 입력을 사용했다. 선형·거리 기반·MLP 등 필요한 모델에만 픽셀 `/255`를 적용하고, 트리·앙상블은 원본 픽셀 스케일을 사용했다. `StandardScaler`처럼 모델별로 필요한 추가 전처리를 사용하는 경우 **학습 데이터에서만 `fit`하는 Pipeline**에 넣고 적용 여부를 표시한다. 초기 탐색에서 모델별 전처리가 다르면 단일 변수만의 효과로 해석하지 않는다.
3. **실행 예산:** 기본값 또는 단순한 초기 설정으로 **후보당 짧게 1회 학습**한다. 전체 데이터 최적화, GridSearch 및 다중 Seed 반복은 이 단계에서 수행하지 않는다. 동일 컴퓨터·CPU 스레드 등 실행 환경과 후보별 **사전 시간 상한**을 실행 전에 기록한다. 제한 시간 초과·메모리 부족은 **실패 사유**로 기록하며 정확도 0점으로 간주하지 않는다.
4. **독립 평가 데이터:** 이 단계는 **공식 Training에서 분리한 Validation만** 사용한다. 공식 Test 16,000장 및 Hand-made 2,560장은 13개 초기 후보의 학습·순위 산정에 섞지 않는다. Hand-made를 통한 Normal/Combined 비교는 후속 C단계에서 수행한다.

**초기 평가 지표 및 선정 우선순위**

| 순서 | 항목 | 정의·기록 기준 | 활용 |
|---|---|---|---|
| **1 (핵심)** | **Validation Accuracy** | 전체 정답 수 / 같은 Validation 이미지 수 | 1차 성능 순위 |
| **2 (핵심)** | **Macro F1** | 16개 클래스 F1의 단순 평균 | 클래스별 성능 편차 확인, 정확도 근접 시 판단 |
| **3 (필수)** | **추론 시간** | 동일 Validation 3,200장에 대한 전체 `predict` 시간 및 이미지당 평균 시간. 전처리 포함 여부 통일 | 근접 모델의 실제 사용 효율 비교 |
| **4 (필수)** | **학습 시간** | `fit` 전체 실행 시간. Pipeline의 전처리 학습 포함 | 반복 최적화 비용과 실행 가능성 |
| **보조** | 수렴/실행 안정성 | 경고, 오류, 수렴 반복 수, 메모리·시간 제한, `random_state` 적용 여부 | 결과의 신뢰성 및 실행 불가 사유 |

- **선정 원칙:** Accuracy를 우선하고 Macro F1로 검증한다. **성능 차이가 근소한 후보군**에서는 추론 시간을 더 중요하게 비교하고, 유사한 성능·비용이라면 **분류 원리의 다양성**도 고려한다. 근소함을 구분할 수치 범위는 초기 결과의 분포·반복 변동을 본 후 확정한다. 근거 없는 임의 가중치로 종합 점수 하나를 만들지 않는다.
- **중요:** 기존 우선 검토 모델 **LinearSVC / Random Forest / MLP / KNN** 4종은 *사전 잠정안*일 뿐이다. 13개 결과가 다른 조합을 더 설득력 있게 보여주면 **최종 비교 대상 5개는 이미 실측·선정 기준으로 확정했다.**
- **원본 기록 보존:** W01의 EXP-유-001은 당시의 **이론 기반 3개 선정 이력**으로 남겨두고, W02의 **실측 기반 13개 → 5개 재선정**을 새 실험으로 기록한다.
- **완료:** 13개 후보 전부 단일 실행 완료, 비교표·코드·실험로그 기록, **5개 선정/8개 제외** 완료. 실측값은 [W02 실험 1 기록](logs/EXP-유-002_13개후보_초기평가_5개선정.md) 및 [CSV](results/EXP-유-002_13models_ranked.csv)에 보존한다.

### 단계 B — Normal Dataset 기반 5개 모델 Baseline (**실험 2 완료**)

- **목적:** Hand-made를 학습에 넣기 전에, 선택된 5개 모델이 MNIST·EMNIST만으로 어느 정도의 분류 성능을 내는지 확인한다.
- **분할:** 공식 v3 Training 96,000장 안에서 **Train:Validation≈8:2(목표 76,800:19,200)**로 분리한다. 원본과 증강본은 같은 그룹에 배치하며, 실제 개수와 그룹 중복 검사를 로그에 남긴다.
- **공통 조건:** 16개 레이블, 28×28 → **784 features**, 동일한 Training/Validation 인덱스. 초기 Baseline에서는 모델의 기본 Hyperparameter를 우선 사용하며, 필요한 모델에만 적절한 스케일링을 적용한다. 전처리·정규화의 적용 범위를 기록한다.
- **지표:** Accuracy, Macro F1, Confusion Matrix 및 클래스별 성능, 학습 시간, 추론 시간, 수렴/반복 기록. 학습 정확도는 보조로 기록하되 모델 선정의 주지표는 고정 Validation 결과다.
- **실제 완료 결과:** 5개 모델 모두 고정 Normal Validation **19,200장** 평가 완료. SVC(RBF) 97.67%, HistGradientBoosting 97.46%, MLP 97.32%, Extra Trees 96.97%, KNN 96.29%. 원본 결과·학습/추론 시간·클래스별 지표·Confusion Matrix는 [EXP-유-003 실험 2](logs/EXP-유-003_5개모델_Normal_Baseline.md) 및 [CSV](results/EXP-유-003_5models_comparison.csv)에 기록.
- **상태:** **완료 — Normal 5개 모델 기본 성능 평가에 한함**. [실험 1](logs/EXP-유-002_13개후보_초기평가_5개선정.md)의 12,000/3,200장 비교와 **별도 실험**이다. Hand-made 평가·Combined 학습·최적화는 아직 미실시.

**실험 2 Confusion Matrix 시각화:** [SVC (RBF)](results/confusion_matrices/SVC_RBF_confusion_matrix.png) · [HistGradientBoosting](results/confusion_matrices/HistGradientBoosting_confusion_matrix.png) · [MLPClassifier](results/confusion_matrices/MLPClassifier_confusion_matrix.png) · [ExtraTreesClassifier](results/confusion_matrices/ExtraTrees_confusion_matrix.png) · [KNeighborsClassifier](results/confusion_matrices/KNeighbors_confusion_matrix.png) · [시각화 설명 및 원본 연결](results/confusion_matrices/README.md). 실제 Validation 19,200장에 대한 **5개 모델의 16×16 컬러 Heatmap PNG**이며, [EXP-유-003](logs/EXP-유-003_5개모델_Normal_Baseline.md)에서 각각 확인할 수 있다.

### 단계 C — Hand-made 인식 성능 평가 및 방향 보정 (실험 3 완료)

- **목적:** Hand-made 데이터를 아직 학습하지 않은 Normal 모델 5개를 실제 손글씨에 적용해 공식 Validation 대비 성능 하락과 작성자별 특성을 분석한다.
- **데이터:** `02_data/Hand_made/ml26_41.npz` – `ml26_44.npz` (작성자 4명×640장=2,560장). 이미지 방향·글씨 위치·극성·픽셀 범위를 검사한다. **2026-11-09에 Hand-made 4개 NPZ를 원본 JPG 기반 0–255 Grayscale로 교체**했으며, 42·44번은 흑백 반전했다. 입력에 추가 `*255`를 하지 않고 실험 2 모델의 기존 스케일링 Pipeline만 유지한다.
- **평가 구분:** Hand-made **전체 2,560장에 대한 초기 탐색적 평가**는 작성자별 성능·오분류를 기술하는 데 쓸 수 있다. 다만 뒤에서 Combined 학습에 추가할 이미지는 최종 미사용 평가셋이 아니므로, **학습 전/후 정량 비교용으로는 미리 지정한 보류 작성자 이미지**를 고정해서 사용한다.
- **지표:** `Hand-made Validation Accuracy - Normal Validation Accuracy`(**%p**), 클래스별 Macro F1/Confusion Matrix, 작성자별 Accuracy, 추론 시간. 이 차이는 입력 분포가 다른 두 데이터에 대한 **관찰된 성능 차이**이지 그 자체로 원인 증명은 아니다.
- **완료 기준:** 5개 Normal 모델 각각의 공식 Validation 및 Hand-made 정확도 비교, 작성자별 결과, 대표 오분류 이미지, 입력 데이터 검사 기록.
- **실제 완료:** 원본 JPG에서 42·44 흑백 극성을 반전한 Grayscale 데이터로 5개 모델 평가(**19.88–23.12%**). 41·42·44번 방향 불일치를 발견해 반시계 90° 보정 후 같은 모델로 재평가(**53.24–62.93%**). **보정 전·후 결과 모두 보존**, 실험 2의 학습 모델 재학습 없음.
- **상태:** **완료 — [EXP-유-004 실험일지](logs/EXP-유-004_Handmade_방향보정_성능평가.md)** · [전체 비교](results/EXP-유-004/comparison_before_after.csv) · [작성자별 비교](results/EXP-유-004/per_writer_before_after.csv) · [변환 검증](../../02_data/Hand_made/GRAYSCALE_VERIFICATION.md).

### 단계 D — 성능 하락 원인 분석 및 5개 모델별 개선 (후속 실험 4)

1. **데이터·입력 분포 요인:** 방향 회전, 중심 정렬, 글자 크기, 획 두께, 역상/스케일, 원본과 증강본 품질 등을 확인한다. 확인된 입력 문제는 **공통 Preprocessing**을 우선 설계하되, 반드시 동일한 고정 Validation으로 변경 전/후 효과를 비교한다. 실제 개선되지 않은 원인을 확정 사실로 기록하지 않는다.
2. **모델·학습 설정 요인:** Training과 Validation 차이, 과적합, 수렴 경고, 모델 복잡도, 거리·커널 특성을 검토한다. 공통 전처리와 모델별 Hyperparameter 효과를 가능한 한 분리하여 기록한다.
3. **변경 이력:** 5개 모두 기본 설정(Baseline)을 보존하고, 변경한 변수·값·이유·Seed·지표·소요 시간을 기록한다. `random_state`가 학습 변동에 의미가 있는 모델에서 반복 측정하고, `max_iter`에 도달한 모델은 수렴 경향을 보고 횟수를 늘린다. KNN·Extra Trees처럼 `max_iter`가 없는 모델에 반복 최적화 설정을 억지로 적용하지 않는다.

| 모델 | 조정 가능한 Hyperparameter 예시 | 특히 확인할 점 |
|---|---|---|
| **SVC (RBF)** | `C`, `gamma` | 분류 정확도와 긴 추론 시간, 입력 스케일 |
| **HistGradientBoosting** | `max_iter`, `learning_rate`, `max_leaf_nodes` | 긴 학습 시간·조기 종료 여부·성능 변화 |
| **MLP** | `hidden_layer_sizes`, `alpha`, `learning_rate_init`, `max_iter` | 입력 스케일, `loss_curve_`, 수렴 여부 |
| **Extra Trees** | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | 과적합 및 학습·추론 비용 |
| **KNN** | `n_neighbors`, `weights`, `metric`, `p` | 거리·스케일링의 영향과 예측 시간 |

- **완료 기준:** 5개 모델의 각 Baseline과 최소 하나 이상의 **실제 검증된 개선 시도**(성공/실패 포함), 공통 Preprocessing 전후 비교 및 실험 코드·로그.
- **상태:** **계획**. 변경할 구체적인 탐색 횟수와 시간 예산은 실행 전 확정한다.

### 단계 E — Combined Dataset 재학습 및 Normal 대비 분석 (후속 실험 5)

- **목적:** MNIST·EMNIST만 학습한 Normal 모델과 **Hand-made를 추가 학습한 Combined 모델**을 같은 데이터로 평가하여 실제 손글씨 인식 개선 효과를 확인한다.
- **Hand-made 분할:** 작성자 단위로 분리한다. **예시:** 41·42·43번 1,920장을 추가 학습에 사용하고, 44번 640장을 검증에 유지한다. 특정 작성자에 대한 우연한 유리함을 줄이기 위한 **작성자 교체 검증(4개 Fold)**은 후속 세부 계획에서 실행 예산과 함께 확정한다.
- **Normal Training:** 공식 Training에서 고정된 학습용 부분(약 76,800장). **Combined Training:** 정확히 같은 공식 Training + 학습용 Hand-made 일부(예시 1,920장).
- **검증:** 공식 Validation(약 19,200장)과 Hand-made 보류 작성자(예시 640장)를 **Normal/Combined 두 모델 모두 동일하게** 사용한다. 공식 성능의 유지·변화, Hand-made Accuracy·Macro F1 개선량(%p), 클래스별 오분류, 학습/추론 시간을 각각 비교한다.
- **주의:** Hand-made 학습 이미지가 검증 이미지에 포함되지 않아야 한다. 모델별 보류 작성자가 달라지면 직접적인 성능 순위를 매기기 어렵기 때문에 **5개 모두 같은 작성자 분할을 적용**한다.
- **완료 기준:** 5개 모델의 Normal/Combined 비교표(같은 평가셋), Train/Validation 인덱스와 파일·Seed, 학습에 추가한 이미지 수, 전처리·설정 변경 내용 및 결과 분석.
- **상태:** **계획**.

### 단계 F — 모델 간 비교·최종 1개 선정·추가 Fine-tuning (후속 실험 6)

- **순위 기준:** 최적화된 5개 모델의 공식 Validation Accuracy·Macro F1, 보류 작성자 Hand-made Accuracy, Confusion Matrix와 **추론 시간**을 함께 비교한다. 단순히 초기 Accuracy 1위였다는 이유로 최종 모델을 확정하지 않는다.
- **교수님 피드백:** 1·2위 성능 차이가 근소하다면 **두 모델을 동시에 추가 Fine-tuning**하여 추론 시간·변동성까지 포함해 다시 평가한 뒤 최종 1개를 선정한다.
- **독립 평가:** 선정·설정 확정 이후 공식 Test 16,000장으로 별도 평가한다. 공식 Test를 모델 선택 또는 Hyperparameter 조정에 반복 사용하지 않는다. 교수님이 배포하는 단계별 외부 평가 데이터는 해당 지침에 따른다.
- **완료 기준:** 최종 1개 모델 선정 사유(실측 수치), 5개 모델 최적화 후 성능표, 근접 후보 추가 Fine-tuning 결과(필요 시), 독립 Test 결과.
- **상태:** **계획**.

> **주차 범위:** W02에서는 **A단계(실험 1: 13개 후보 평가·5개 선정) 및 B단계(실험 2: Normal Dataset 5개 모델 Baseline)가 완료**되었다. **C는 완료했고 D–F는 아직 진행하지 않았으며**, 후속 주차에 걸쳐 진행할 수 있다. D–F의 수치나 완료 상태를 미리 채우지 않는다.

## 3. 실험 간 공통 비교 원칙

1. **데이터:** [최종 v3 112,000장 NPZ](../../02_data/datasets/MNIST_EMNIST_official_split_112000.npz). 공식 Training **96,000장** 안에서 고정된 Training/Validation을 만든다. 공식 Test **16,000장**은 최종 독립 평가용으로 분리한다.
2. **누출 방지:** 기존 증강본과 생성 원본이 같은 학습/검증 그룹에 있도록 분할하고, Validation의 정보로 학습 전처리 통계량을 산출하지 않는다. 정확한 그룹핑 기준과 인덱스/Seed를 저장한다.
3. **기본 입력:** 16개 클래스(0–9, A–F), 28×28 → 784. Normal·Hand-made 모두 **0–255 Grayscale** 기준이며, 42번은 원본 JPG부터 실질적인 중간 밝기 단계가 적다. SVC·MLP·KNN은 저장된 Pipeline 내부에서 `/255`를 수행하므로 Hand-made 입력에 `*255`를 하지 않는다.
4. **평가:** Accuracy, Macro F1, 클래스별 점수, Confusion Matrix, 학습 시간 및 **실제 예측 시간(추론 시간)**. 필요하면 수렴 횟수·경고·시간 제한을 보고한다.
5. **공정 비교:** 같은 Validation 샘플과 공통 성능 계산 로직을 사용하되, **모델별로 적절한 전처리·Hyperparameter 탐색은 허용**한다. 예비 평가 결과와 최적화 결과는 별도 표로 기록한다.
6. **재현성:** 입력 파일/버전, 추출 표본/분할 방식/Seed, 코드·scikit-learn 버전, 장치/스레드, 모델별 파라미터, 시작/종료 시각과 실제 실행 결과를 기록한다.

## 4. 산출물 및 역할 배정 (실험 1 완료 / 후속 세부안 확정 전)

| 산출물 | 내용 | 상태 |
|---|---|---|
| 후보 전체 **빠른 예비 실험 비교표** | 대상 모델·데이터 수·기본 설정·성능·실행 시간·제외 사유 | **완료 — EXP-유-002** |
| **5개 모델 선정 근거** | Accuracy≥90%·Random Forest 계열 중복 제외 기준 | **완료 — EXP-유-002** |
| 모델별 실험 코드·Notebook | **5개 모델 Normal Baseline 코드·지표 완료** / 개선 변경 이력은 후속 단계 | **Baseline 완료 / 최적화 예정** |
| 모델별 실험 로그 | **EXP-유-003 Normal Baseline 평가표·Confusion Matrix 기록 완료** / 후속 변경 기록은 예정 | **Baseline 완료 / 후속 예정** |
| Normal/Combined 비교 결과 | Hand-made 학습 편입 여부·홀드아웃 출처 명시 | **후속 단계** |
| 상위 2개 추가 비교 및 최종 선정 | 성능이 근접할 경우 동시 개선, 추론 시간 반영 | **후속 단계** |

**실험 1의 실제 담당·산출:** 팀장 유용성 — [EXP-유-002 로그](logs/EXP-유-002_13개후보_초기평가_5개선정.md), [13개 후보 CSV](results/EXP-유-002_13models_ranked.csv). 이후 5개 모델의 개별 최적화 담당은 미확정.

**역할 분담 원칙:** 4명의 팀원(유용성·김다연·이선형·곽지호)이 병렬로 참여하도록 배분한다. **누가 어느 알고리즘을 맡을지**, 5개 모델의 개별 최적화·담당 배분, 실험 시간 예산, 주요 전처리 및 Hyperparameter 탐색 횟수는 다음 회의에서 확정한다. W01의 완료 기록과 실제로 실행하지 않은 W02 결과를 혼동하지 않는다.

## 5. W01에서 이어지는 사항 및 기록 규칙

- W01: 이론적 모델 검토 및 데이터셋 v3 구성·검증. W02: **기존 13개 후보 전체의 소규모 예비 학습 → 실제 수치에 근거한 5개 모델 선정 → 최적화 실험 착수**.
- 기존 [실험일지 양식](../EXPERIMENT_TEMPLATE.md)에 따라 개별 실험의 입력, 조건, 변경 이유, 실행 코드, 결과 및 검증을 기록한다. 실패한 실험도 보존한다.
- W02 종료 시 `WEEK_SUMMARY.md`를 작성하되 **실제로 확인한 결과만** 기록한다. 현재 W02의 **실험 1(13개 후보 실측·5개 선정)**, **실험 2(Normal 5개 Baseline)** 및 **실험 3(Hand-made 미보정·방향 보정 평가)**까지 완료했다. 5개 최적화 및 최종 1개 선정은 **미실시/미확정**이다.
