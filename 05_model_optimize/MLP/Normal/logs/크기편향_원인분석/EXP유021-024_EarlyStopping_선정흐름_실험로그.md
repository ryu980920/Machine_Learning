# EXP유021–024 — Early Stopping 검증과 최종 모델 유지 결정 로그

- **기록일:** 2026-11-17
- **연구 위치:** `05_model_optimize/MLP/Normal/logs/크기편향_원인분석/EXP유021-024_EarlyStopping_선정흐름_실험로그.md`
- **기록 성격:** 실제 완료된 EXP유021 → EXP유022 → EXP유023 → EXP유024의 선택 사유·대조 실험·실측 결과·불채택 근거를 연결한 후속 로그
- **현재 최종 상태:** **EXP유017 `slot1_alpha_up` 동결 모델 유지.** Early Stopping 정책과 Full-Train 재학습 모델로 최종 교체하지 않음
- **실험 관리:** Official Train/Validation과 기존 Hand-made 개발 검증은 구분; Original Test·교수님 Test 미사용. 모델·하이퍼파라미터·원본 데이터는 덮어쓰지 않음

## 1. 출발점 — EXP유017에서 어떤 기준으로 모델을 선정했는가

EXP유015–017의 목표는 공식 검증 정확도를 높이되, 문자/숫자 계열을 크기로 구별하는 Shortcut을 악화시키지 않는 것이었다. 따라서 **정확도가 가장 높은 모델을 무조건 선택하지 않는 순서**를 EXP유017에서 사전 고정했다.

**선정 순서:** ① OfficialVal 16클래스·B/8/D·숫자/문자 크기변형 안전 Gate 통과 → ② 통과한 모델 중 OfficialVal 원본(1.0배) 정확도 최고 → ③ 동률이면 B·8·D 최악 Family Flip 최소 → ④ 다시 동률이면 전체 16클래스 최악 Family Flip 최소.

주요 보호 Gate의 일부는 B·8·D의 0.8/1.2배 **평균 Family Flip ≤20%, 개별 Seed 최악 ≤30%**, 전체 16클래스 최악 ≤25%, 숫자×1.2·문자×0.8 Family Flip 각각 ≤10%와 확률 이동 상한, 최소 OfficialVal 정확도 및 숫자/문자별 정확도 등이었다. **Family Flip은 크기변형 전후 숫자/문자 계열 결정의 변경 비율로, B/8/D 정답 Recall과 다른 지표**다. 기준 전체는 EXP유017 원본 로그가 우선한다.

- 학습 설정은 Size 80% (긴 변 16/18/20/22/24px) + Stroke 14% (형태 보존 검사) + CoM, MLP `(256,128)`, `alpha=0.03`, `learning_rate_init=0.001`, ReLU/Adam, Seed 42/43/44, `early_stopping=False`.
- EXP유017은 총 46개 하이퍼파라미터 조합×3 Seed=138개 모델을 비교했고, **Gate 통과 10/46조건** 중 `slot1_alpha_up`를 선택했다.
- 채택 당시 3-Seed OfficialVal **96.5365%**, B·8·D 최악 Family Flip **16.67%**. 이후 같은 네 작성자의 기존 손글씨 2,560장 평가에서 **92.0703%**.
- 그보다 원본 OfficialVal 정확도가 근소하게 높은 LR 0.0014 후보 **96.5521%**는 B·8·D 최악 Family Flip **21.69%**로 사전 Gate에 탈락했다. 따라서 `slot1_alpha_up` 선정은 정확도만 보고 한 사후 선택이 아니다.

**보존 원칙:** 이후 Early Stopping 실험은 이 선정의 타당성을 추가 검증하는 후속 연구이며, 기존 동결 모델·Gate·선정 이력을 소급 수정하지 않는다.

## 2. 후속 연구 질문의 발생과 실험 순서

| 단계 | 이 단계에서 생긴 질문 | 고정하거나 바꾼 조건 | 해당 단계에서 도달한 판단 |
|---|---|---|---|
| **EXP유021** | 기존 OFF 모델은 더 학습하면서 과적합 신호가 있었는가? 단순 ON/OFF를 비교할 수 있는가? | 기존 MLP/증강 유지, OFF·내부 Accuracy 기반 ON 학습곡선 측정 | Train Loss 하락과 검증 NLL 악화가 함께 나타남. 하지만 ON/OFF의 실제 학습 이미지 수·patience가 달라 인과 결론 보류 |
| **EXP유022** | 학습 데이터와 가중치 갱신 경로가 같다면 Validation Loss 선택이 Accuracy 선택보다 유리한가? | OfficialTrain 내부 고정 그룹분할 69,119/7,681. 동일 Seed의 동일 학습 경로에서 Loss Best/Accuracy Best/Last 체크포인트 비교 | 두 Best가 Last보다 OfficialVal에서 우수. Loss Best가 Accuracy Best보다 낫다는 결과는 아님 |
| **EXP유023** | EXP유022에서 선택한 체크포인트가 실제 팀 손글씨에도 유리한가? | EXP유022 **원본 체크포인트 9개 고정**, GitHub 손글씨 2,560장 추론만 수행 | 두 Best가 Last보다 손글씨 평균에서 우수하나, 기존 EXP유017에는 미달. 크기 취약 클래스 잔존 |
| **EXP유024** | 내부 검증용으로 뗀 7,681장을 다시 학습에 넣어 전체 76,800장을 쓰면 기존 최적 Epoch가 유효한가? | EXP유022가 정한 Epoch 고정, 76,800장으로 Seed별 처음부터 재학습 후 체크포인트 추출 | **Last가 OfficialVal·손글씨 모두 평균 1위.** 기존 EXP유017보다 OfficialVal 낮음 → 최종 모델 유지 |

실험은 **원인 관찰 → 동일 경로 통제 → 손글씨 재검증 → 전체 학습 데이터 재학습의 안정성 확인** 순서로 진행했다. EXP유024 결과를 보고 EXP유022의 선정 Epoch를 다시 고르지 않았다.

## 3. EXP유021 — Learning Curve에서 조기 종료 필요성 검토

- OFF: OfficialTrain **76,800장** 학습, Training Loss 기반 종료. ON: 내부 10% 분리로 **69,120장** 가중치 학습, 내부 Validation **Accuracy** 기준 조기 종료. patience도 OFF 10, ON 15로 달랐다.
- 실제 3-Seed OfficialVal Accuracy: 새로 학습한 **OFF 96.5556%, ON 96.4705%**. 이 차이를 Early Stopping만의 인과 효과라고 해석하지 않는다.
- OFF에서 외부 OfficialVal NLL 최저 시점이 종료보다 앞서 있고, 그 이후 Train Loss는 감소했지만 Val NLL은 다시 증가·진동했다. 이는 **과적합 위험 신호**이나 외부 OfficialVal 최고 시점으로 가중치를 사후 선정하면 안 된다.
- **다음 실험으로 넘어간 이유:** 내부 검증 Loss와 Accuracy를 동일한 학습 경로에서 직접 비교해야 했기 때문이다.

## 4. EXP유022 — 동일 분할·동일 학습 경로에서 체크포인트 선택

- OfficialTrain 76,800장에서 원본 출처 그룹 중복을 막는 `StratifiedGroupKFold` 고정 분할로 **학습 69,119장 / 내부 Validation 7,681장** 구성. 내부 Validation은 학습용 증강 없이 CoM만 적용.
- MLP `(256,128)`, `alpha=.03`, LR `.001`, Seed 42/43/44. **같은 Seed에서는 동일한 하나의 학습 경로**를 사용하며 내부 검증 NLL 최소, 내부 검증 Accuracy 최고, 실제 마지막 Epoch를 각각 저장.
- 두 기준 모두 개선되지 않는 기간을 **18 Epoch**로 두고 최대 **105 Epoch**. 외부 OfficialVal 19,200장은 체크포인트 결정 이후 평가용으로 사용.

| Seed | 내부 Loss Best epoch | 내부 Accuracy Best epoch | Last epoch |
|---:|---:|---:|---:|
| 42 | 48 | 63 | 81 |
| 43 | 65 | 65 | 83 |
| 44 | 48 | 48 | 66 |

| EXP유022 선택 방식 | OfficialVal Accuracy (3-Seed 평균) | OfficialVal NLL |
|---|---:|---:|
| 내부 **Accuracy Best** | **96.5399%** | **0.13510** |
| 내부 **Loss Best** | 96.4601% | 0.13559 |
| **Last** | 96.3090% | 0.14235 |

**중간 판정:** 검증 기반 체크포인트 선택은 동일한 69,119장 학습 조건에서 Last보다 일반화 성능이 높았다. 하지만 **Loss 선택의 우월성은 입증되지 않았고**, Accuracy 선택이 OfficialVal에서 근소하게 앞섰다. 이는 전체 76,800장을 사용한 최종 모델 채택 결론과는 별개다.

## 5. EXP유023 — 원본 9개 체크포인트의 손글씨 2,560장 검증

- 사용자 제공 EXP유022 ZIP의 Seed42/43/44 × 3종 체크포인트 **9개 원본 SHA-256 검증**, **추가 학습 없이** 추론.
- 손글씨는 GitHub Actions `43906447255`의 `ml26_41`~`ml26_44` NPZ, **4명×640장=2,560장**, 16클래스×160장, 원본 SHA-256 검증.
- 손글씨 CoM, `/255` 스케일, 숫자/문자별 정확도, 작성자별 정확도, 16클래스 confusion, B/8/D Recall·0.8~1.2배 크기 민감도 평가.

| EXP유023 선택 방식 | 손글씨 Accuracy (3-Seed 평균) | 손글씨 NLL | OfficialVal Accuracy |
|---|---:|---:|---:|
| **Accuracy Best** | **91.6927%** | 0.34028 | **96.5399%** |
| **Loss Best** | 91.6146% | **0.33794** | 96.4601% |
| **Last** | 91.2109% | 0.34997 | 96.3090% |
| **기존 EXP유017 Frozen** | **92.0703%** | — | 96.5365% |

Seed별 손글씨 정확도: Seed 42는 Accuracy Best **91.7188%**/Loss Best **91.4844%**/Last **90.7031%**, Seed 43은 Best 두 종류 **91.9141%**/Last **91.0938%**, Seed 44는 Best 두 종류 **91.4453%**/Last **91.8359%**. **43·44는 Loss Best와 Accuracy Best가 동일 체크포인트**였다.

B·8·D Recall(Seed 평균): Accuracy Best **89.38/82.71/77.71%**, Loss Best **90.21/81.46/78.75%**, Last **87.71/83.75/73.33%**, EXP유017 **92.71/82.29/83.75%**. 전체 정확도가 일부 높아도 취약 클래스에서 기존 최종 모델보다 좋아졌다고 단정할 수 없다.

**중간 판정:** 검증 기반 선택은 손글씨에서도 Last 대비 평균 개선을 보였지만, 두 Best의 손글씨 정확도 격차 **0.0781%p**는 매우 작았고, 기존 EXP유017 동결 모델을 넘어섰다는 근거가 없었다. 특히 69,119장 학습과 76,800장 학습의 데이터 수 차이가 남아 있으므로 최종 모델 교체 보류.

## 6. EXP유024 — 이전 최적 Epoch를 고정하고 Full Train 76,800장 재학습

### 6-1. 왜 이 실험을 했는가

EXP유022에서 내부 Validation에 사용한 **7,681장**을 실제 가중치 학습에 다시 포함하면 모델 성능이 달라질 가능성이 있었다. 따라서 **손글씨나 외부 OfficialVal에서 새 Epoch를 찾지 않고**, EXP유022 내부 검증으로 이미 고정한 `48/63/81`, `65/65/83`, `48/48/66`을 그대로 옮겨 확인했다.

### 6-2. 고정 조건과 평가 경계

- Official v3 SHA-256 `0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879`; OfficialTrain **76,800장** 전부 가중치 학습, OfficialVal **19,200장** 결과 평가용.
- Size80%+Stroke14%+CoM 재현 검증. 학습 입력 픽셀 합계 **2,133,060,736** 일치; Stroke 적용 **57,642장**, topology 거부 **3,798장**.
- MLP `(256,128)`, ReLU/Adam, alpha `.03`, 초기 LR `.001`. Seed마다 **Last까지 한 번 학습하면서** EXP유022가 정한 Epoch에서 가중치 스냅샷 저장. **9개 각각 따로 초기화해 학습한 실험이 아니다.**
- 비교 대상은 FullTrain의 Loss Best/Accuracy Best/Last **총 9개 체크포인트**. 같은 OfficialVal과 같은 기존 Hand-made 2,560장에 평가.
- **구분:** 이 실험은 내부 Validation을 사용한 실시간 `early_stopping=True` 학습이 아니라 **과거에 결정된 종료 Epoch를 전량 학습에 이식하는 실험**이다. 따라서 결과가 ‘Early Stopping의 보편적 무효’를 입증하는 것은 아니다.

### 6-3. 전체 결과 — 세 Seed 평균

| EXP유024 선택 방식 | OfficialVal Accuracy | OfficialVal NLL | 손글씨 Accuracy | 손글씨 NLL | 손글씨 숫자 Accuracy | 손글씨 문자 Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Loss Best | 96.0799% | 0.15708 | 91.7578% | 0.31548 | 91.5417% | 92.1181% |
| Accuracy Best | 96.1215% | 0.15689 | 91.4583% | 0.32787 | 90.7083% | **92.7083%** |
| **Last** | **96.1979%** | **0.15148** | **92.0703%** | **0.31384** | **91.8333%** | 92.4653% |
| **EXP유017 Frozen** | **96.5365%** | — | **92.0703%** | — | 91.5833% | **92.8819%** |

### 6-4. 기존 EXP유022(69,119장)에서 FullTrain으로 바뀐 차이

| 선택 | OfficialVal: 69,119 → 76,800 | 변화 | 손글씨: 69,119 → 76,800 | 변화 |
|---|---:|---:|---:|---:|
| Loss Best | 96.4601% → 96.0799% | −0.3802%p | 91.6146% → 91.7578% | +0.1432%p |
| Accuracy Best | 96.5399% → 96.1215% | −0.4184%p | 91.6927% → 91.4583% | −0.2344%p |
| Last | 96.3090% → 96.1979% | −0.1111%p | 91.2109% → 92.0703% | +0.8594%p |

학습에 쓰는 이미지를 늘린 효과가 **선택 Epoch마다 다르게 나타났다.** FullTrain에서 Last의 손글씨 정확도는 크게 올라갔지만, 모든 설정의 OfficialVal 정확도가 이전보다 내려갔다. 단순한 ‘Training 데이터 증가 = 일반화 개선’은 관찰되지 않았다.

### 6-5. Seed별 손글씨·B/8/D 검증

| Seed | Loss Best | Accuracy Best | Last | Seed 내 1위 |
|---:|---:|---:|---:|---|
| 42 | **92.2266%** | 91.3281% | 91.7188% | Loss Best |
| 43 | 91.2500% | 91.2500% | **92.1094%** | Last |
| 44 | 91.7969% | 91.7969% | **92.3828%** | Last |

| 손글씨 B/8/D Recall (3-Seed 평균) | B | 8 | D |
|---|---:|---:|---:|
| Loss Best | 85.00% | **85.83%** | 83.54% |
| Accuracy Best | 87.50% | 82.08% | **83.75%** |
| Last | 90.00% | 84.79% | 80.63% |
| **EXP유017 Frozen** | **92.71%** | 82.29% | **83.75%** |

FullTrain Last는 전체 손글씨 평균에서 1위이지만 **D Recall은 80.63%**로 기존 EXP유017 **83.75%**보다 낮았다. 0.8~1.2배 B·8·D 크기 변형에서도 인식률 변동이 남았다.

**선정 Gate 관련 주의:** EXP유024 결과에 손글씨 B/8/D Family Flip 일부가 기록되어 있더라도, EXP유017의 **OfficialVal 전체 16클래스×배율, 숫자/문자 확률 이동, 개별 Seed 조건까지 포함한 전체 사전 Gate를 동일한 방식으로 완전 재평가한 기록은 없다.** 손글씨 Recall 몇 개로 ‘Gate 통과’라고 표기하지 않는다.

## 7. 후보 선정 흐름과 최종 결정

```text
[EXP유017] 증강·MLP 후보 탐색
  → 사전등록 Shortcut Gate 전체 통과 여부 검사
  → Gate 합격 모델 중 OfficialVal 원본 정확도 최고 채택
  → slot1_alpha_up: (256,128) / alpha 0.03 / LR 0.001 / ES OFF
  → 모델·파라미터 동결 (OfficialVal 96.5365%, 손글씨 92.0703%)
          |
          v
[EXP유021] OFF에서 Train Loss↓, Val NLL 악화 신호 확인
  → ON/OFF 학습 데이터 수·patience 달라 판단 유보
          |
          v
[EXP유022] 내부 그룹 분할 69,119 Train / 7,681 Val
  → 같은 경로에서 Loss Best vs Accuracy Best vs Last 비교
  → 두 Best가 Last보다 OfficialVal 우세, Loss 우월성 미확인
          |
          v
[EXP유023] EXP022 원본 9개 체크포인트로 손글씨 2,560장 추론
  → 두 Best > Last (평균), EXP017에는 미달
  → 기존 최종 모델 유지·FullTrain 교차 확인 결정
          |
          v
[EXP유024] EXP022 선정 Epoch를 고정하고 76,800장 재학습
  → Last 평균 우세; Seed42는 Loss Best 우세 (Seed 의존)
  → FullTrain Last 손글씨 92.0703%, OfficialVal 96.1979%
  → EXP017 손글씨 92.0703%, OfficialVal 96.5365%
  → EXP017 전체 사전 Gate에 대한 동등 재검증도 없음
          |
          v
[최종] 새 후보 채택 없음 / EXP유017 Frozen 유지
```

### 최종 채택·보류 기준별 판정

| 판단 축 | 이번 실측 근거 | 판정 |
|---|---|---|
| 동일한 69,119장 학습에서 검증 기반 Best가 Last보다 좋은가? | EXP유022 OfficialVal, EXP유023 손글씨 평균에서 둘 다 우수 | **해당 조건에 한해 Yes** |
| 내부 Validation Loss 기준이 Accuracy 기준보다 우수한가? | OfficialVal과 손글씨 정확도는 Accuracy Best가 근소하게 우세, 손글씨 NLL은 Loss Best가 소폭 우세 | **우위 확정 불가** |
| EXP유022 최적 Epoch를 전체 76,800장에 옮겨도 최적 모델인가? | EXP유024에서 평균 Last가 최고, Seed별 순위도 다름 | **재현되지 않음** |
| FullTrain Last가 기존 EXP유017을 대체하는가? | 손글씨 평균 동률(92.0703%), OfficialVal 96.1979%로 EXP017 96.5365%보다 낮음; B/D 일부 취약 | **교체 보류** |
| EXP유017 사전 Shortcut Gate 통과가 새 모델에 입증됐는가? | EXP유024는 원본 Gate 전체에 대한 완전 동등 검증이 없음 | **미확인 (통과 주장 금지)** |

**최종 결론:** 검증 점수를 이용한 체크포인트 선택은 학습 데이터와 선정 정책이 고정된 EXP유022에서 의미가 있었지만, 그 최적 Epoch를 전체 Train으로 이식했을 때는 동일한 우위가 유지되지 않았다. 따라서 Early Stopping을 무조건 ON으로 바꾸거나, Loss Best·Accuracy Best를 최종 배포 모델로 교체하지 않는다. **EXP유017에서 사전 정의한 ‘안전 Gate → 정상 OfficialVal 정확도 → 취약 클래스 안정성’ 선정 순서와 `slot1_alpha_up` 동결을 그대로 유지한다.**

## 8. 해석 제한 및 재현 파일

1. **EXP유023/024 손글씨 2,560장은 같은 작성자 4명의 기존 개발 검증용 자료**이며, 새로운 독립 필체 테스트가 아니다. Seed별 평가 결과가 3개 있어도 독립 이미지가 7,680장으로 늘어나는 것은 아니다.
2. EXP유024는 FullTrain 76,800장의 **고정 Epoch 재학습**, EXP유022는 69,119장 학습의 **내부 검증 체크포인트 선택**이다. 각 모델의 매개변수와 학습 경로가 달라 수치 차이를 Early Stopping의 순수한 인과효과로 단정할 수 없다.
3. 한 번의 실험 결과를 보고 OfficialVal이나 손글씨의 최고점으로 새 Epoch를 재선정하지 않는다. **새로운 정책 채택은 별도 데이터·사전 기준·독립 검증을 요구**한다.
4. 기존 동결 모델의 재학습·덮어쓰기와 Original Test/교수님 Test 접근은 수행하지 않았다. GitHub에는 이 로그 문서만 추가하며, 기존 모델·실험 데이터는 변경하지 않는다.

**원본 근거와 재현 경로**

- GitHub 원본: `EXP유017_80-14_MLP_하이퍼파라미터_최적화.md`, `EXP유021_EarlyStopping_LearningCurve_검증보고서.md` (같은 로그 디렉터리)
- EXP유022 원본: `EXP022_SameSplit_ValidationStop_3Seed_Results(1).zip` 내 소스 모델 9개, `EXP022_manifest.json` 등. 내부 검증에서 선택한 Epoch와 가중치 SHA 기준.
- EXP유023 검증: `EXP023_EXP022_Handmade2560_Evaluation_Results.zip` 및 `EXP023_EXP022_Handmade2560_Validation/FINAL_DECISION.md`; 모델·손글씨 NPZ SHA 검증 및 예측 CSV.
- EXP유024 재학습·검증: `EXP024_FullTrain_ValidationStop_Results.zip` 및 `EXP024_FullTrain_ValidationStop/FINAL_REPORT.md`·`audit_manifest.json`·`metrics_9models.csv`·`aggregate_metrics.csv`·`handmade_groups_by_seed.csv`·`scale_B8D_by_seed.csv`·`seed_42~44/manifest.json`·`seed_42~44/training_curve.csv`.
- GitHub 손글씨 근거: [기존 2,560장 검증 원본 Actions #43906447255](https://github.com/ryu980920/Machine_Learning/actions/runs/43906447255), SHA-256 확인된 4개 NPZ.
- GitHub EXP유017 사전 선택·검증: [138모델 집계 재현 및 5배율 감사 #43846851354](https://github.com/ryu980920/Machine_Learning/actions/runs/43846851354).

**기록 상태:** 사용자 승인에 따라 이 Markdown 연구로그를 GitHub `main`에 추가. 기존 EXP유017 동결 모델·체크포인트·원본 데이터·기존 로그는 변경하지 않음.