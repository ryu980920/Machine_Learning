# EXP021 — Early Stopping ON/OFF 3-Seed Learning Curve 실측 보고서

**연구일:** 2026-11-17  | **상태:** 3-Seed의 동일 alpha 재학습·epoch별 모니터링 완료 / 배포 모델 변경 없음

## 1. 연구 질문 및 기준

기존 EXP017 모델의 `early_stopping=False` 설정이 타당한지 실제 Loss/Accuracy Learning Curve에서 확인한다. 목표는 훈련 손실이 계속 줄더라도 검증 손실이 더 좋아지지 않는 시점을 식별하고, scikit-learn의 Early Stopping이 그 지점을 얼마나 정확히 선택하는지 알아보는 것이다.

- **공통 하이퍼파라미터:** sklearn 1.8.0 `MLPClassifier(hidden_layer_sizes=(256,128), alpha=0.03, learning_rate_init=0.001, activation=relu, solver=adam, max_iter=200)`; Seed42/43/44, CoM, Size 80% + Stroke 강도 14%.
- **데이터:** SHA256 검증된 Official v3 NPZ(아래 manifest); Official Train 76,800장, Official Validation 19,200장. Size/Stroke 증강 Train 대상 61,440장, 획 형태 변형 57,642장, 거부 3,798장.
- **OFF:** 학습 Train 76,800장, `early_stopping=False`, `n_iter_no_change=10`, 훈련 손실 개선 여부로 종료.
- **ON:** Train 69,120장 + 내부 검증 7,680장(`validation_fraction=0.1`, 내부 단일 stratified split), `early_stopping=True`, `n_iter_no_change=15`, 내부 검증 **Accuracy**가 최고인 epoch의 가중치를 복원.
- **공통 측정:** 매 epoch Training Loss, 공식 외부 Validation의 Log Loss(NLL) 및 Accuracy를 기록. 외부 OfficialVal은 gradient/내부 Early Stopping 판단에 사용되지 않았지만, 여러 epoch 성능을 살펴본 **진단용 자료**이므로 최저점 선택 후의 수치는 독립 Test로 취급하면 안 된다.

## 2. 학습 곡선 그림

- [전체 Learning Curve: 3 Seed×(Training Loss, OfficialVal Log Loss, Accuracy)](EXP021_Learning_Curves_3Seeds.png)
- [Validation 구간 확대: epoch≥10](EXP021_Validation_Plateau_Zoom_3Seeds.png)
- [ON 내부 Accuracy 기준과 OfficialVal Accuracy 비교](EXP021_EarlyStopping_Internal_vs_Official_Accuracy.png)

**그래프 해석 시 주의:** Train Loss에는 L2 규제항이 포함되고, OfficialVal NLL은 규제항이 없는 원본 입력의 log loss다. 그러므로 두 손실값의 절대적인 차이보다 각 추세를 비교한다. ON/OFF는 학습행 수와 patience도 다르기 때문에 단순히 한 가지 옵션만 바꾼 인과 실험이 아니다.

## 3. Seed별 직접 측정값

| Seed | OFF 종료 | OFF 외부 NLL 최저 epoch | OFF 정확도 | ON 종료→복원 | ON 외부 NLL 최저 epoch | ON 복원 정확도 |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 81 | 51 | 96.6562% | 75→**59** | 59 | 96.7031% |
| 43 | 61 | 55 | 96.4896% | 66→**50** | 46 | 96.5312% |
| 44 | 57 | 26 | 96.5208% | 52→**36** | 52 | 96.1771% |

- 신규 OFF 재학습 3-Seed 평균 OfficialVal Accuracy: **96.5556%**, NLL **0.12989**.
- 신규 ON 재학습 3-Seed 평균 OfficialVal Accuracy: **96.4705%**, NLL **0.13219**.
- 이전 동결 EXP017 체크포인트 3-Seed 평균 OfficialVal Accuracy: **96.5365%**, NLL **0.13062**. 원본 체크포인트는 다시 학습하지 않았으며 seed별 n_iter=[72, 101, 57].

## 4. 학습 종료 시점에 관한 결론

1. **OFF 종료의 지연 가능성은 확인됐다.** 세 Seed 모두 외부 OfficialVal Loss의 최저점(42: 51, 43: 55, 44: 26)보다 뒤(81, 61, 57)에서 종료했다. Training Loss 기반의 종료가 일반화 손실 최저점과 일치한다는 근거는 없다.
2. **ON이 모든 Seed에서 검증 최적점을 맞춘 것은 아니다.** ON 모델의 내부 검증 최고 Accuracy epoch는 42: 59, 43: 50, 44: 36이며 외부 OfficialVal NLL 최저점은 각각 59, 46, 52다. 특히 Seed44에서는 내부 검증이 36 epoch를 선택해, 외부 NLL이 개선된 52 epoch를 놓쳤다.
3. **ON을 즉시 채택할 근거도 부족하다.** 같은 alpha에서 3 Seed 평균 정확도는 신규 OFF 96.5556% vs ON 96.4705%로, ON이 평균적으로 더 높지 않았다. 이 비교에는 10% 내부 검증용 분리 및 patience 차이가 섞였다.
4. **외부 OfficialVal 최고 epoch를 그대로 선택하면 안 된다.** 다중 조회한 외부 Validation의 최저 NLL은 과거를 사후 진단하는 자료이지 독립적인 배포 모델 선택 기준이 아니다.

## 5. 권고 — 다음 통제 실험

**권장되는 엄밀한 후속 실험은 동일한 90% 학습 자료와 동일한 경로에서 종료 조건만 바꾸는 것**이다. 공식 Train에서 이미지 출처 그룹을 고려해 69,120 Train / 7,680 내부 Holdout을 먼저 분리하고, Size/Stroke 증강은 69,120 Train에만 적용한다(내부 Holdout은 CoM만). 세 Seed별로 동일한 학습 실행의 epoch 체크포인트를 기록하여 (A) 마지막 훈련 Loss 기반 종료, (B) 내부 Validation Accuracy 최고점, (C) 내부 Validation Log Loss 최저점을 비교한다. 조기 종료 patience와 min_delta는 사전 정의하여 고정한다. 이후 선택된 각 체크포인트를 Official Validation에 한 번 평가하고 7→9/9→7·B/8/D 크기 Shortcut Gate를 검사한다.

**현재 판정:** 기존 동결 모델 유지. 과적합 신호와 조기 종료 정책의 Seed 의존성은 확인했으나, ON 정책이 가장 좋다고 결론 내리지는 않는다.

## 6. 재현 파일

- `EXP021_epochwise_3seed_learning_curves.csv`: 모든 epoch의 세 Loss/Accuracy 지표
- `EXP021_comparison_by_seed.csv`: 종료 epoch, 최저 외부 NLL epoch, 정확도 집계
- `EXP021_original_frozen_model_check.csv`: 원본 동결 모델의 별도 SHA 기반 검증 성능
- `monitor_training_v3.py`: 실제 매 epoch 후 평가 코드 (sklearn 내부 hook 사용)
- `prepare.py`, `train_local_warp.py`, `pixel_transform.py`: 공식 v3의 Train 구성/기존 80-14 증강 재구성
- `fit_*.json`, `curve_*.csv`: 실행 단위별 원본 측정 자료

원본 v3 파일, 3개 기존 동결 모델은 이 ZIP에 재포함하지 않음. 사용자 보유 원본 ZIP 경로 및 SHA를 `SHA256_Inputs.txt`에 기록. 독립 Test 및 교수님 Test는 실행하지 않음.
