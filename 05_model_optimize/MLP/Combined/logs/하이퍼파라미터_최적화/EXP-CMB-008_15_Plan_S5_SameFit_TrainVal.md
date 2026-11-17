# EXP-CMB-008 · 15 사전등록 — 같은 학습 실행의 Training / Validation Learning Curve

> 상태: 신규 실험 실행 전. 기존 S5 및 12~14번 진단 파일은 보존한다. 사용자 요청은 Accuracy(세로축)-Epoch(가로축)에 실제 Training 및 Validation 두 곡선을 그려 과적합 발생 여부를 점검하는 것이다.

## 문제 정의

기존 Learning Curve 데이터에는 Training Loss와 내부 Validation Accuracy/Log-loss는 있었지만, **동일 Epoch의 실제 Training Accuracy는 기록되어 있지 않았다**. 또한 서로 다른 실행의 Early Stopping 결과가 3/12 Fold만 정확히 재현되는 문제가 있었으므로, 다른 실행에서 얻은 Train·Val 곡선을 단순히 결합하는 방식은 사용하지 않는다.

## 학습 조건 및 비교 대상

- **학습 데이터:** Size30% / Stroke 강도 28% / CoM. Official Train 76,800장과 학습 대상 Writer 3명 1,920장 = **78,720장/Fold**. 그 안에서 sklearn의 실제 Train(90%, **70,848장**) / Internal Validation(10%, **7,872장**) 분할을 직접 계측한다. Writer held-out 640장은 학습·내부 종료에 사용하지 않는다.
- **모델:** MLP 단층 (256,), alpha=0.0001, learning_rate_init=0.001, ReLU/Adam, sklearn 1.8.0, Seed 42/43/44, Writer 41/42/43/44.
- **하나의 학습 실행에서 120 Epoch:** early_stopping=True, validation_fraction=0.10, n_iter_no_change=200으로 진단 구간을 연장. **동일 가중치·동일 Adam 상태·동일 Train/Val Split**을 유지하며 120 Epoch에서 종료한다.
- **ES 조기 종료 지점은 Counterfactual로 계산:** 원래 S5의 patience=10이었으면 실제 Validation Accuracy 이력에서 언제 멈추었는지와 그 이전 최고 Validation Epoch를 기록한다. 원본 S5와 다른 실행의 Stop Epoch가 100% 재현됐다고 주장하지 않는다.
- **학습 단위:** 3 Seed × 4 Writer-held-out = **12개 모델**, 추가 독립 Test나 교수님 Test, 신규 Hand-made 480장 사용하지 않는다.

## 매 Epoch 지표 (Epoch 1, 5, 10, …, 120)

1. **Training Accuracy:** sklearn이 실제 학습한 70,848장 **전체**에 대한 정확도. 샘플 평균이나 오차 기반 추정치가 아님.
2. **Internal Validation Accuracy:** 동일 Fit에서 분리한 7,872장에 대한 정확도. sklearn의 실제 Early Stopping 판단 지표.
3. **Training Objective Loss:** sklearn의 실제 Loss Curve 값 (규제 포함).
4. **Internal Validation Log-loss:** 같은 Epoch와 동일 가중치에서 내부 검증 데이터로 계산한 확률적 Cross-Entropy (규제항 포함 안 함).
5. **보조:** Writer OOF Accuracy, D→0/B→8/8→B/0→D는 **진단만**. 이 값을 근거로 Epoch를 고르거나 모델을 배포하지 않음.

## 그래프 6종과 판정

- Figure 1 **Training Accuracy vs Internal Validation Accuracy** (x=Epoch, y=Accuracy %) — 사용자가 요청한 핵심 그래프.
- Figure 2 **Training Objective Loss vs Validation Log-loss** (x=Epoch, y=Loss).
- Figure 3 **Loss 확대** — 낮아지던 Validation Loss가 다시 상승하는 시점 확인.
- Figure 4 **Train−Val Accuracy 일반화 격차**.
- Figure 5 **Internal Validation Accuracy vs Writer OOF** (진단용).
- Figure 6 **4방향 오분류 곡선** (진단용).
- Figure 1과 2에는 같은 실행 내에서 추정한 patience=10의 **평균 최고 Validation Epoch와 평균 Would-stop Epoch**를 표시하되, 평균 수직선을 하나의 실제 Fold 중지 시점으로 오해하지 않도록 Fold별 CSV를 제공한다.
- **과적합 징후:** Training Accuracy 상승/높은 유지와 Validation Accuracy의 정체 또는 하락, 혹은 Training Loss 하락과 Validation Loss 상승이 공존하는지 평가한다. Accuracy가 뚜렷하게 하락하지 않더라도 Loss가 악화될 수 있다.
- **변경 금지:** OOF 최고 Epoch를 최적 학습 길이로 채택 금지. S1~S5 기존 개발용 HP 결과를 별도 검증 전 변경하지 않는다.

## 산출물

결과 경로 `05_model_optimize/MLP/Combined/results/Hyperparameter_Optimization/S5_SameFit_TrainVal/` 에 6 PNG, 원본 집계 CSV, Fold별 Best/Stop Epoch 및 검증 JSON을 저장하고, `EXP-CMB-008_16_S5_SameFit_TrainVal_Results.md`에 **실측 그림과 분석**을 기록한다. 학습 코드·분석 코드·데이터 SHA 및 실제 학습 분할은 모두 추적 가능해야 한다.
