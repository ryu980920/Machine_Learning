# EXP-CMB-008 · 12 S5 Early Stopping Learning Curve 진단 사전등록

> **상태: 신규 진단 학습 실행 전 사전등록.** 기존 S5의 ES=False 선택은 이번 진단을 마치기 전까지 잠정 결정이다. 이 실험은 원본 S5와 분리된 원인 분석이며, 어떠한 최종 독립 Test 데이터도 확인하지 않는다.

## 1. 왜 S5 결과만으로는 Early Stopping이 적절했다고 결론 내릴 수 없는가?

원본 S5의 ES=False는 4-Writer×3-Seed OOF **92.214%**, ES=True는 **91.771%**였다. 주요 오분류 D→0은 **67→77건**, 8→B는 **38→49건**으로 증가했다. 평균 학습 반복은 63.4→47.2로 감소했지만, 이는 최종 성능과 학습 횟수 비교일 뿐 **ES가 정확히 언제 멈췄는지**, **내부 Validation 최고 Epoch**, **후속 Epoch의 OOF 추세**, **90% 학습 데이터 분리의 영향**을 설명하지 못한다.

scikit-learn 1.8.0의 `MLPClassifier(early_stopping=True)`는 Training Fold 내부 10%를 별도 Validation으로 떼어내고, 최고 Validation **Accuracy**의 모델 가중치를 복원한다. Stop Epoch와 복원되는 Best Epoch는 같지 않을 수 있으며, 종료 후 최종 테스트처럼 OOF를 보는 것은 **진단 전용**이다.

## 2. 사전 가설·변수 통제

| 가설 | 실험 질문 | 비교 |
|---|---|---|
| **H1 조기 종료** | 내부 Validation에 따른 종료 후 학습을 지속하면 Writer OOF와 D→0이 개선되는가? | es10 vs internal120의 동일 Epoch 연속 궤적 및 사전 고정 Epoch 40/60/80/100/120 |
| **H2 검증 기준 불일치** | 내부 Validation Accuracy가 최고인 Epoch와 새로운 Writer OOF가 개선되는 구간이 다른가? | 내부 Val 곡선 vs Writer OOF 정확도·방향별 오분류 곡선 |
| **H3 학습 장수 차이** | 내부 검증용으로 10%를 분리한 효과가 ES 종료 효과와 구별되는가? | 같은 Epoch의 full120(100% Training) vs internal120(90% Training) |

**고정된 설정:** Size30%-Stroke28% 증강, CoM, 단층 (256,), alpha=1e-4, Adam/ReLU, learning_rate_init=0.001, tol=1e-4, Seed 42/43/44, held-out Writer 41/42/43/44, 기존 data SHA/증강 Seed 및 scikit-learn 1.8.0.

## 3. 3개 실험 Arm × 3Seed × 4Writer = 36개 학습

| Arm | 학습 데이터 | 설정 | 목적 |
|---|---|---|---|
| `es10` | 90% (10% 내부 Validation) | `early_stopping=True`, `n_iter_no_change=10`, `max_iter=200` | **원본 S5 재현:** Stop Epoch, Best Validation Epoch, 실제 복원 가중치 검증 |
| `internal120` | `es10`과 동일한 90% | `early_stopping=True`, `n_iter_no_change=200`, `max_iter=120` | **학습 지속 실험:** optimizer state·split·초기화 동일, 120 Epoch까지 관측 |
| `full120` | Training Fold 100% | `early_stopping=False`, `n_iter_no_change=200`, `max_iter=120` | **동일 Epoch 100% 학습 대조** |

- `internal120`은 `es10`의 10% Validation 분할을 **동일 sklearn random_state**로 재현한다. Stop 이전 모든 Epoch에 대해 Training Loss, 내부 Validation Score, OOF 정확도 및 핵심 오분류 네 방향이 `es10`과 **완전히 일치해야** 한다. 다르면 실험 집계를 실패 처리한다.
- `es10`의 실제 Stop Epoch/OOF 최종 예측을 **원래 S5 GitHub Actions run #44020696572의 원본 아티팩트와 대조**한다. 불일치 시 그래프를 연구 결과로 채택하지 않는다.
- 120 Epoch는 **Learning Curve 원인 진단 구간**이며 새로 선정된 최종 `max_iter`가 아니다. 비교 Arm의 설정 변경 사항은 사전 고정한다.
- 90% vs 100% 조건은 내부 Validation을 사용하는지 여부·학습 미니배치 흐름도 함께 달라지는 비교다. 그러므로 두 모델 간 차이를 무조건 '10% 표본 감소만의 인과 효과'로 표현하지 않는다.

## 4. Learning Curve와 정량 지표

1. **Training Loss vs Epoch:** 각 Arm 실제 관측값, 12 Fold별 Epoch 및 평균. 90/100% Training Loss 절대 수준 단순 비교 금지.
2. **Internal Validation Accuracy vs Epoch:** Val 최고 Epoch, 실제 Stop Epoch 분리. Val Loss는 5 Epoch마다 부가 기록(조기 종료는 Val Accuracy를 사용).
3. **Writer-held-out OOF Accuracy vs Epoch:** 3 Seed×4Writer OOF. **OOF 최고 Epoch는 후속 학습 선택에 사용하지 않고 설명용으로만 표시**.
4. **오분류 방향별 Epoch 곡선:** **D→0**, B→8, 8→B, 0→D 각각 실제 클래스당 총 480회에 대한 오분류율, 목표 5% 기준선.
5. **Fold별 Best/Stop Epoch 점도:** 전체 12개 Fold의 산포.
6. **사전 고정 Epoch 40/60/80/100/120**에서 internal120와 full120을 비교; 사후 OOF 최적 Epoch를 선정하면 데이터 누수/선택 편향이라는 점을 명시한다.

## 5. 과적합 판단과 다음 결정을 위한 조건

- Early Stopping을 활성화한다는 사실만으로 과적합이 줄었다고 결론 내리지 않는다.
- **H1 지지:** 동일한 90% 학습 집합에서 기존 Stop 이후의 미리 정한 Epoch(80/100/120 등)에 OOF 정확도가 지속 개선되고 주요 오류가 감소.
- **H2 지지:** 최고 내부 Validation Accuracy/복원 가중치가 보류 Writer의 정확도·D→0 개선 구간과 체계적으로 다를 때. 오직 하나의 우연한 Fold로 결론 내리지 않고 Writer·Seed별 패턴을 공개.
- **H3 지지:** 같은 Epoch의 100% Training 대조에서 90%보다 일관되게 좋아지는 경우. 다만 데이터 장수 이외의 차이를 함께 밝힌다.
- 모든 경우 **실제 Test는 S5 보완 실험에 사용하지 않는다**. OOF 자체를 매 Epoch 관측했다고 해서 OOF 최고 Epoch의 가중치를 골라 배포할 수 없다.
- 진단 결과를 근거로 patience·validation_fraction의 **후속 별도 사전등록 실험**을 설계할 수는 있지만, 5% 방향별 오류율 및 전체 정확도·Writer 안전 Gate는 이전 사용자 기준대로 적용한다.

## 6. 산출물·성공 조건

- `results/Hyperparameter_Optimization/S5_LearningCurves/learning_curve_epoch_aggregate.csv`
- `.../fold_checkpoint_summary.csv`, `diagnostic_summary.json`
- `.../figures/01_training_loss.png`, `02_internal_validation_accuracy.png`, `03_heldout_writer_oof.png`, `04_directional_errors.png`, `05_best_vs_stop_epoch.png`
- [13_실측 보고서](EXP-CMB-008_13_S5_LearningCurves_Results.md)에는 위 실제 그림과 수치를 삽입한다.
- 원본 S5 재현(Stop Epoch·예측), ES 연장 학습의 종료 이전 Epoch 일치, 모든 36개 학습 완료, SHA/Writer 분리 검증 **모두 성공해야** 결과를 `main`에 기록한다.

[진단 학습 코드](../../scripts/exp_cmb008_s5_learning_curve_trace.py) · [실제 차트/통계 생성 코드](../../scripts/exp_cmb008_s5_learning_curve_aggregate.py).
