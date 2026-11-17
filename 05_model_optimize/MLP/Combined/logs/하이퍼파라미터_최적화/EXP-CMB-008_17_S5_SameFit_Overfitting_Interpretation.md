# EXP-CMB-008 · 17 Same-fit Learning Curve — Overfitting 및 Early Stopping 종료위치 검증

> 근거: [16번 같은 실행의 실제 Training/Validation Accuracy·Loss 6개 그래프](EXP-CMB-008_16_S5_SameFit_TrainVal_Results.md) · [원본 집계 CSV](../../results/Hyperparameter_Optimization/S5_SameFit_TrainVal/same_run_train_val_curve_aggregate.csv) · [12 Fold Best/Stop CSV](../../results/Hyperparameter_Optimization/S5_SameFit_TrainVal/per_fold_best_stop.csv). 독립 Test 미사용.

## 1. 이번에는 실제 Training Accuracy를 측정했다

- 같은 모델의 같은 Epoch에서 실제 Training Partition 70,848장 전체 Accuracy, 내부 Validation 7,872장 Accuracy, Training Objective Loss, Validation Cross-Entropy Loss를 동시에 계측했다.
- 세 Seed 42/43/44 × Writer Held-out 41/42/43/44 = 12 Fold, 120 Epoch. 학습 구조 (256,), Size30/Stroke28/CoM, alpha=1e-4, LR=.001 고정. 다른 실행의 Training과 Validation을 결합하지 않았다.
- [Accuracy Learning Curve PNG](../../results/Hyperparameter_Optimization/S5_SameFit_TrainVal/figures/01_training_vs_validation_accuracy.png) · [Loss curve PNG](../../results/Hyperparameter_Optimization/S5_SameFit_TrainVal/figures/02_training_vs_validation_loss.png) · [과적합 확대](../../results/Hyperparameter_Optimization/S5_SameFit_TrainVal/figures/03_overfit_loss_zoom.png).

## 2. 정량 실측과 과적합 신호

| Epoch | Training Accuracy | Internal Validation Accuracy | Training Objective Loss | Internal Validation Log-loss | Accuracy gap |
|---:|---:|---:|---:|---:|---:|
| 1 | 90.011% | 89.239% | 0.619413 | 0.382708 | 0.772%p |
| 10 | 98.627% | 95.889% | 0.071280 | **0.140814** (평균 곡선 최저) | 2.738%p |
| 40 | 99.793% | 96.116% | 0.009255 | 0.189815 | 3.677%p |
| 80 | **100.000%** | **96.635%** (평균 곡선 최고) | 0.001840 | 0.203820 | 3.365%p |
| 120 | 99.969% | 96.527% | 0.004459 | **0.219020** | **3.442%p** |

Training Accuracy는 20~120 Epoch 구간에서 거의 100%이고, 내부 Validation Accuracy는 96% 전후에서 완만하게 변동했다. 두 곡선의 차이는 약 3.4%p로 확대·유지됐다. 내부 Validation Accuracy가 뚜렷하게 하락하는 형태는 아니므로 그 사실을 과장하지 않는다.

반면 Validation Log-loss는 평균 곡선의 10 Epoch에서 최저를 기록한 뒤 대체로 상승했고, Training Objective Loss는 낮은 수준으로 감소했다. 즉 **분류 정확도가 plateau를 보이는 동안 오답의 확신도 또는 확률 품질이 악화되는 과적합 신호**를 확인했다. Training Objective는 규제 포함, Validation Log-loss는 규제항을 제외한 Cross-Entropy이므로 절대값을 동일 목적 함수처럼 단순 비교하지 않는다.

## 3. Early Stopping patience=10은 적절한 Epoch에 멈추었는가?

- 이번 120 Epoch 단일 학습 경로에서 sklearn의 실제 내부 Validation Accuracy 및 no-improvement count를 기록하여 patience=10을 적용했을 때의 중단 Epoch를 **같은 실행 내에서** 재구성했다.
- 가상의 평균 Stop Epoch **42.17**, 그 전에 최고 Validation Accuracy를 기록한 평균 Best Checkpoint Epoch **31.17**. 실제 학습은 인과 진단을 위해 patience=200으로 120 Epoch까지 이어갔고, 120 Epoch 중 모델을 OOF 최고점으로 선정하지 않았다.
- **12/12 Fold 모두** 120 Epoch 내 내부 Validation Accuracy 최고 Epoch가 patience=10 기준 가상 Stop Epoch **이후**에 존재했다. 즉 이번 동일 학습 경로에서는 patience=10이 후속 Validation Accuracy 개선 구간을 전부 놓쳤다.
- 그러나 이는 **검증 Accuracy를 기준으로 했을 때** 지나치게 빨리 멈췄다는 뜻이지, Validation Log-loss를 최소화하거나 새로운 Writer D→0 오분류를 최소화하는 Epoch가 뒤에 있었다는 뜻이 아니다.
- 평균 Validation Log-loss 최저는 Epoch 10이므로, **Accuracy 기반 Early Stopping의 Best Epoch와 Loss 기반 과적합 신호가 다르다**는 결론을 분명히 남긴다.

## 4. 의사결정

현재 ES=False는 기존 S1~S5에서 정한 개발용 잠정값으로 유지한다. 이번 결과는 ES=True를 단순 기각할 근거로 쓰기보다, 조기 종료의 모니터링 지표/Patience를 재검증할 필요성을 뒷받침한다. 다음 단계에서 Patience를 바꿀 경우 **먼저 학습 내부 Validation만으로 사전 규칙을 정의**하고, Writer OOF와 네 방향(D→0/B→8/8→B/0→D) 조건 및 독립 Test 보호를 유지한 채 별도 평가해야 한다.

원본 S5와 이번 실행 간 완전 재현 문제가 있었음을 숨기지 않는다. 다만 이번 두 Learning Curve는 **동일 학습 실행**에서 측정했기 때문에 서로 다른 실행 간 재현 불일치에 직접 의존하지 않는다.
