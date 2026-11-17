# EXP-CMB-008 · 14 S5 Learning Curve 정량 해석과 후속 의사결정

> **실측 결과 검증:** [13 실측 보고서와 실제 그림 5개](EXP-CMB-008_13_S5_LearningCurves_Results.md) · [최초 재현성 불일치 기록](EXP-CMB-008_12A_S5_Reproducibility_Deviation.md). 분석 대상은 S5 원본 결과를 덮어쓴 모델이 아닌, 새로 측정한 **3 Seed × 4 Writer × 3 조건 = 36회 학습**이다. Original Test·교수님 Test·신규 손글씨 480장 미사용.

## 1. 실제 ES가 언제 멈췄나?

- **원본 S5 ES=True** 평균 종료 Epoch **47.17**, OOF **91.771%** (D→0 **77건**, B→8 **49건**, 8→B **49건**, 0→D **14건**).
- **새 `es10` 추적 실험** 평균 실제 종료 Epoch **42.75**, 평균 최고 내부 Validation Epoch **31.75**. 내부 점수 최고 모델의 가중치가 복원된다. **'42.75 Epoch의 마지막 가중치'가 최종 모델은 아니다.** 대부분 종료는 최고점 이후 patience를 더 기다려 발생한다.
- 새 `es10`의 **복원 가중치 OOF 92.044%**, 방향별 D→0 71, B→8 59, 8→B 46, 0→D 15. 수치는 원본 S5와 다르므로 명확히 구분한다.

## 2. Learining Curve에서 관측된 변화

| 조건 | 실제 측정 Epoch | OOF Accuracy | D→0 | B→8 | 8→B | 0→D |
|---|---:|---:|---:|---:|---:|---:|
| 새 ES10 (best Val 복원) | 평균 Best 31.75, Stop 42.75 | 92.044% | 71 | 59 | 46 | 15 |
| 90% Train + 내부 Val, 지속 | 40 | 92.018% | 64 | 52 | 47 | 14 |
| 90% Train + 내부 Val, 지속 | 60 | **92.591%** | 67 | 48 | 39 | 16 |
| 90% Train + 내부 Val, 지속 | 80 | 92.500% | 69 | 51 | **32** | 18 |
| 90% Train + 내부 Val, 지속 | 100 | 92.214% | 61 | 49 | 57 | 19 |
| 90% Train + 내부 Val, 지속 | 120 | **92.604%** | **59** | **43** | 40 | 20 |
| 100% Train, 지속 | 60 | 92.344% | 64 | **36** | 38 | 9 |
| 100% Train, 지속 | 100 | 92.591% | 68 | 44 | 37 | 10 |
| 100% Train, 지속 | 120 | 92.578% | 70 | 44 | 47 | 10 |

- 90% Training을 **Epoch120까지 이어간 조건은 새로운 ES10 복원 모델보다 OOF 정확도 +0.560%p** 높고 D→0 **−12건**, B→8 **−16건**, 8→B **−6건**이다. 그러나 **0→D는 15→20건으로 악화**되었다.
- 기존 S5 선정 조건인 **ES=False OOF 92.214%, D→0/B→8/8→B/0→D=67/43/38/14**와 비교해도 90% 연장 120 Epoch의 OOF는 높지만 8→B **38→40**, 0→D **14→20**으로 증가해 **기존 4방향 각각 비악화 기준을 통과하지 못한다**.
- OOF는 Epoch 60→100 사이 **92.591→92.214%**로 하락했다가 120에서 **92.604%**로 다시 증가했다. *더 오래 학습하면 반드시 좋다*는 단조 증가도 관찰되지 않았다. 특정 Epoch의 우승을 OOF에서 고르면 탐색 과적합이다.

## 3. 내부 Validation이 실제 목표에 부합하는가?

| 90% Train 지속 조건 | Epoch40 | Epoch60 | Epoch80 | Epoch100 | Epoch120 |
|---|---:|---:|---:|---:|---:|
| Train Loss | 0.008217 | 0.003766 | **0.001839** | 0.002761 | 0.004459 |
| 내부 Validation Accuracy | 96.135% | 96.502% | **96.636%** | 96.346% | 96.528% |
| 내부 Validation Log-loss | 0.18875 | 0.19392 | 0.20330 | 0.22879 | 0.21879 |
| Writer OOF Accuracy | 92.018% | 92.591% | 92.500% | 92.214% | **92.604%** |

- 내부 Validation **Accuracy**는 계속 개선·변동하는 반면 **Log-loss**는 Epoch40 이후 높아졌다. 이는 점수 개선과 예측 확신도/확률 보정 악화가 공존할 가능성을 시사한다. 내부 Validation Loss는 보조 측정치이며 실제 Early Stopping 기준은 Accuracy다.
- Epoch80의 높은 내부 Validation Accuracy가 반드시 Writer OOF 최고 또는 D→0 최저와 일치하지 않는다. 단, 4 Writer 반복 탐색을 통한 OOF 관측이므로 독립적 최적 종료 지점이라는 의미는 아니다.

## 4. 사전등록 가설의 검토

- **H1 — 너무 이른 종료 가능성:** 새 ES10 모델에 비해 미리 정한 Epoch60·120의 학습 지속 OOF가 더 높고 D→0도 Epoch120에서 감소. **조기 종료가 지나치게 보수적일 가능성을 지지하는 관측**. 하지만 정확히 같은 경로를 따라간 Fold가 9/12개이므로 **인과적으로 확정 불가**.
- **H2 — Validation/Writer 일반화 시점 차이:** 내부 Validation Accuracy와 OOF·D→0의 개선 Epoch가 일정하게 일치하지 않음. 다른 Writer로의 일반화와 내부분할 정확도가 완전한 대리 목표가 아니라는 가능성.
- **H3 — 10% 분리 영향:** Epoch120에서 90% Train OOF **92.604%**, 100% Train **92.578%**. **90%가 더 나빴다고 할 수 없고**, 특정 방향은 100%가 더 좋거나 나빴다. 데이터 비율 외에 내부 Val 유무·학습 순서가 함께 바뀌었으므로 데이터 수 감소만으로 원인 확정 불가.

## 5. 재현성 제한과 의사결정

**엄격한 재현성 검사 미통과:**
- 원본 S5 vs 재측정 ES10 Stop Epoch/예측이 정확히 일치한 Fold **3/12**.
- `es10` vs `internal120` Stop 전 매 Epoch Loss·Validation·OOF가 일치한 Fold **9/12**.
- 동일 데이터 SHA/Seed를 확인했으나, 서로 다른 GitHub Actions 러너의 MLP 학습 경로가 완전히 결정론적으로 재현된 것은 아니다. **부동소수점/스레드 연산 차이는 가설일 뿐 원인을 특정하지 않았다**.

따라서 이 실험의 결론은 **현재 ES=True를 최종 채택할 근거는 부족하지만, ES=False가 과적합 면에서 확실히 우수하다는 주장도 성립하지 않는다**는 것이다. 기존 (256,), alpha=1e-4, LR=.001, **ES=False는 개발용 잠정 설정으로 유지**한다.

**후속 필요 검증:** 같은 실행 환경·동일 Fold 안에서 ES10과 장기 학습을 순차 실행해 Stop 이전의 전 Epoch가 완전히 일치하도록 통제하고, 종료 방식을 사전 규정된 내부 Validation 규칙으로만 바꿔 재평가해야 한다. **OOF 정확도 최고 Epoch를 직접 선택하는 것은 금지**한다. Early Stopping 파라미터 변경은 이 재현성 검증 뒤 별도 사전등록 실험으로 수행한다.

[실제 Train Loss 그림](../../results/Hyperparameter_Optimization/S5_LearningCurves/figures/01_training_loss.png) · [Validation 그림](../../results/Hyperparameter_Optimization/S5_LearningCurves/figures/02_internal_validation_accuracy.png) · [OOF 그림](../../results/Hyperparameter_Optimization/S5_LearningCurves/figures/03_heldout_writer_oof.png) · [주요 혼동 방향 그림](../../results/Hyperparameter_Optimization/S5_LearningCurves/figures/04_directional_errors.png) · [Best vs Stop](../../results/Hyperparameter_Optimization/S5_LearningCurves/figures/05_best_vs_stop_epoch.png)
