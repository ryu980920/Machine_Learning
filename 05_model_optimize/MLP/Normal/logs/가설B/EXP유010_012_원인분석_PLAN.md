# Normal MLP · EXP유010·011·012 원인 규명 실험 사전계획

> **상태:** 사전 등록 계획. 결과는 성공한 GitHub Actions 실제 계측 후 기록한다.
> **연구 질문:** 왜 CoM-only에서는 숫자가 문자보다 정확하지만, Size-only에서는 문자 우세로 역전되며 Size→CoM에서는 숫자가 회복되고 문자 우세가 유지되는가?

## 이전 기준선 (EXP유006 실측, 동일 Seed 42·43·44 평균)

| 조건 | Hand-made 숫자 | Hand-made 문자 | Official Val |
|---|---:|---:|---:|
| CoM-only | 84.4167% | 66.4931% | 97.0521% |
| Size-only | 36.4583% | 94.2014% | 96.6858% |
| Size→CoM | 84.0625% | 91.7708% | 96.0608% |

**우선 통제:** 기존 EXP유006의 9개 모델(C-o-M, Size, Size→CoM × Seed42,43,44)을 원본 Actions #43785390739에서 받아 **checkpoint SHA256와 최초 성공 실행의 정확한 정답 수를 대조**한다. 새 모델 훈련 없이, 가중치를 고정한 대조실험으로 시작한다.

## EXP유010 — Training과 Hand-made의 글자 크기 분포 차이

**가설:** EMNIST 공식 문자 크기 분포가 작은 Hand-made 문자와 달라, Size 변환이 이 불일치를 줄여 문자 인식률을 높였다.

**측정:** Official Train76,800/Val19,200/Hand-made2,560, 16개 클래스와 전체·숫자·문자별, 작성자41·42·43·44별 원본/Size/Size→CoM 형상을 측정.
- BBox width/height/long-side, CoM x/y/offset, bbox center, ink ratio, ink mass, edge mass, 실제 scaling factor, CoM shift.
- 클래스별 Raw → Size → SizeCoM Training-vs-Handmade 분포 차이: Wasserstein-1 on long-side/width/height/position/mass, OfficialVal을 분포 안정성 참조로 사용.
- 같은 문자의 수정/신규 오류는 EXP유011의 **동일 이미지 대응 예측**과 결합해서 해석.
- **지지 기준:** Size 후 Training-HM 분포 격차가 감소하고, 격차가 큰 문자에서 정확도 개선이 크다면 지지. 분포 거리만으로 인과 결론은 금지.
- 산출물: geometry_summary.csv, geometry_distance.csv, handmade_geometry_per_image.csv.

## EXP유011 — Size-only 숫자 붕괴 원인 (CoM 위치 vs 학습 가중치)

**가설:** Size-only의 숫자 정확도 급락은 Size 후 남는 중심 오프셋/픽셀 좌표 위치에 의해 발생했으며, CoM 재정렬이 이를 보정했다.

**조작:** TRAINED model 3조건 × EVALUATION input 4조건(raw, CoM, Size, Size→CoM), 모든 Seed. 같은 모델과 같은 이미지에서 Size 입력 ↔ Size→CoM 입력을 비교. 동일 평가 입력에서는 Size-trained vs Size→CoM-trained 차이를 비교.
- 3×4 교차표를 OfficialVal과 Handmade에 대해 숫자·문자·전체 및 16개 클래스 Recall로 산출.
- 주요 오분류 B↔8, D↔0, F↔5 감소/증가, 2,560장 정확한 정오답 전환 pairs와 CoM 실제 이동량·질량 변화를 결합.
- 진단 해석: 고정 모델 내 입력 이동 변화 = 위치 민감도 근거. 모델이 다른 대각선 비교는 **학습+입력 양쪽 변경**이므로 이를 오직 위치의 인과 효과라고 부르지 않는다.
- 학습/평가 전처리 불일치(off-diagonal)에서는 분포 이동 OOD 가능성이 있어 성능으로 모델 추천 금지.
- 성공 실행 원본 9 checkpoint의 SHA256, seeds, 예측 정확히 재현 못하면 **실험 즉시 실패**.
- 산출물: cross_metrics.csv, cross_predictions.csv, paired_effects.csv, pair_errors.csv, size_com_geometry_and_outcomes.csv.

## EXP유012 — 크기가 B↔8, D↔0, F↔5 분류 단서인가?

**가설:** 같은 획 형태의 글자라도 MLP가 BBox 크기를 분류 단서로 사용하여 작은 B/D/F를 8/0/5로 판단한다.

**조작:** 모든 Hand-made 16개 클래스에 똑같은 Size scaling factor **0.8,0.9,1.0,1.1,1.2**를 적용하고, EXP유006과 같은 BBox 중심 기반 scipy affine_transform(order1, cval0), 이후 항상 CoM 재정렬. scale=1은 크기 변경 없이 CoM만 재정렬한 입력. 위상·픽셀 밝기 질량·정렬 오차를 별도로 계측.
- 기존 MLP(CoM-trained, Size-trained, Size→CoM-trained)를 완전히 고정.
- 각 클래스와 목표 혼동쌍(B/8, D/0, F/5)에서 상대 경쟁 클래스 확률 및 정확도, 동일 이미지의 크기 변화에 따른 예측 전환·평균 pair margin 측정.
- 진단 해석: 규모에 따라 B↔8 등 예측이 특정 방향으로 일관되게 이동하면 크기 관련 입력 민감도 근거. **스케일 변환은 획 두께·보간·형태까지 변화**시키므로 크기만의 순수 인과 효과라는 주장은 하지 않는다. topology 변화율, ink-mass 비율, CoM 잔여 오차를 반드시 함께 보고한다.
- 산출물: scale_class_response.csv, scale_geometry.csv, 클래스쌍별 margin 곡선/전환율.

## 고정 조건·연구윤리

- **필수:** 기존 EXP유006 Combined Size 구현 그대로(apply_size, BBox foreground threshold32, 공식 Train 긴 변 target20, bilinear affine warp); CoM (center_batch mass threshold32, integer shifts to 13.5).
- Seed42/43/44, 3가지 사전 학습 모델; sklearn1.8.0, numpy2.3.5, scipy1.17.0.
- **기존 9개 모델 가중치 고정**. 새로운 Training/모델 교체 금지. 공식 Original Test, 교수님 Test 미접근. Hand-made는 개발용 4명 반복 사용; 독립 Test 성능으로 해석하지 않는다.
- 기록: 사전 계획·원본 계측·대응 정확도·그림과 주의점. **GitHub 실험 브랜치만 사용**, main/pipeline 변경은 별도 명시적 승인 후 수행.
- 코드 작성 전/후 main 기존 실험·README를 변경하지 않는다.

## Actions 근거

- 원본 모델 실행: https://github.com/ryu980920/Machine_Learning/actions/runs/43785390739
- 새 EXP유010·011·012는 별도 GitHub Actions run 및 30일 보관 Artifact로 검증.
