# MLPClassifier — Normal 모델 최적화

**현재 상태: EXP유001 Pixcel_Attribution, EXP유002 Input_Sensitivity, EXP유003 Spatial_Response 완료.** 세 실험 모두 원인 진단이며, 자동 중심 정렬을 포함한 실제 Pipeline 최적화는 아직 실시하지 않았다.

## 등록한 실험

| ID | 내용 | 상태 | 근거 |
|---|---|---|---|
| **EXP유001 — Pixcel_Attribution** | 공식 Validation과 Hand-made 정답·오분류 이미지 98장 추출, Integrated Gradients 기여도 분석 | **완료 (진단)** | [실험 로그](logs/EXP유001_Pixcel_Attribution.md) · [실행 코드](scripts/Pixcel_Attribution.py) · [실제 결과](results/Pixcel_Attribution/README.md) |
| **EXP유002 — Input_Sensitivity** | 기존 MLP를 고정하고 위치·밝기·굵기·크기 20조건을 21,760장에 적용해 **456,960건**의 민감도를 비교 | **완료 (진단)** | [실험 로그](logs/EXP유002_Input_Sensitivity.md) · [실행 코드](scripts/Input_Sensitivity.py) · [실제 결과](results/Input_Sensitivity/README.md) |
| **EXP유003 — Spatial_Response** | 공식 Train·Validation/손글씨의 **98,560장 공간 분포 비교**, **130,560건**의 고정 모델 반응, **72개 IG** | **완료 (진단)** | [실험 로그](logs/EXP유003_Spatial_Response.md) · [실행 코드](scripts/Spatial_Response_Analysis.py) · [결과 원본](results/Spatial_Response/README.md) |

## 연구 대상

- **Training:** 공식 MNIST·EMNIST 데이터만 사용. 기존 Baseline Train 76,800장·Validation 19,200장 분할을 기준으로 비교.
- **오류 분석:** 공식 Validation Confusion Matrix와 개발용 Hand-made Confusion Matrix 비교. Hand-made는 Normal Training에 편입하지 않는다.
- **기존 Baseline 실행:** [초기 모델 코드](../../../03_experiments/model_optimization/README.md)에서 `--model mlp --variant baseline` 선택.

## 매 실험의 순서

1. **Confusion Matrix와 오분류 사례 분석** — 사용 데이터/레이블, 주요 혼동과 비율 기록.
2. **오차 원인 가설 및 강의 개념 연결** — 이미지·모델 특성과 연결한 검증 가능한 가설.
3. **개선 조건 설정·실행** — 전처리, Training 증강, 데이터 구성, Hyperparameter의 변경 변수 분리.
4. **동일 개발 Validation으로 재평가** — Accuracy·Precision·Recall·F1, Confusion Matrix 및 Iteration에 따른 Training Loss 및 가능하면 Validation 점수, 학습·추론 시간.
5. **결론 및 후속 실험** — 성공과 실패 모두 실험일지·CSV·그림에 연결.

## 저장 경로

- [logs/](logs/README.md): [공통 실험일지 양식](../../EXPERIMENT_TEMPLATE.md)을 복사해 개별 Markdown 저장
- [scripts/](scripts/README.md): 실행 Python·Notebook 및 사용 방법
- [results/](results/README.md): 지표 CSV, Confusion Matrix, Learning Curve, 실제 실행 증빙

**원칙:** 공식 Original Test 16,000장과 교수님 최종 2차 Test는 모델 튜닝에 사용하지 않는다. 교수님 1차 데이터는 최초 Test 후 성능이 부족할 때에만 Validation으로 전환한다. 기록에는 모델 버전과 데이터 누출 여부를 명시한다.

[상위 모델 안내](../README.md) · [05 전체 안내](../../README.md) · [연구 흐름도](../../../01_project/RESEARCH_FLOW.md).
