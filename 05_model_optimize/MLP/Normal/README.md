# MLPClassifier — Normal 모델 최적화

**현재 상태:** 초기분석·가설 A/B 실험을 구분해 기록하며, EXP유010·011·012의 원인 분석은 Normal 바로 아래 **세 독립 폴더**에서 관리한다. 모델·결과 원본의 위치는 유지하며 최종 Pipeline 반영은 별도로 결정한다.

## 등록한 실험

| ID | 내용 | 상태 | 근거 |
|---|---|---|---|
| **EXP유001 — Pixcel_Attribution** | 공식 Validation과 Hand-made 정답·오분류 이미지 98장 추출, Integrated Gradients 기여도 분석 | **완료 (진단)** | [실험 로그](logs/초기분석/EXP유001_Pixcel_Attribution.md) · [실행 코드](scripts/Pixcel_Attribution.py) · [실제 결과](results/Pixcel_Attribution/README.md) |
| **EXP유002 — Input_Sensitivity** | 기존 MLP를 고정하고 위치·밝기·굵기·크기 20조건을 21,760장에 적용해 **456,960건**의 민감도를 비교 | **완료 (진단)** | [실험 로그](logs/초기분석/EXP유002_Input_Sensitivity.md) · [실행 코드](scripts/Input_Sensitivity.py) · [실제 결과](results/Input_Sensitivity/README.md) |
| **EXP유003 — Spatial_Response** | 공식 Train·Validation/손글씨의 **98,560장 공간 분포 비교**, **130,560건**의 고정 모델 반응, **72개 IG** | **완료 (진단)** | [실험 로그](logs/가설B/EXP유003_Spatial_Response.md) · [실행 코드](scripts/Spatial_Response_Analysis.py) · [결과 원본](results/Spatial_Response/README.md) |
| **EXP유004 — Automatic_Centering & CoM Decision Analysis** | 공식 Train **76,800장**으로 CoM/Bounding Box MLP 학습·평가 후, **고정 MLP에 CoM 이동만 적용**하여 이미지 이동과 재학습 영향을 분리. **Hand-made 62.85→77.11%, 숫자 +26.88%p·문자 −6.77%p**; 이동만으로 숫자 +25.69%p·문자 −21.35%p 확인 | **완료 (단일 실험 로그·단일 결과 폴더, 최종 Pipeline 채택 보류)** | [통합 실험 로그](logs/가설B/EXP유004_Automatic_Centering.md) · [정렬 실험 코드](scripts/Automatic_Centering_Experiment.py) · [CoM 판단 코드](scripts/CoM_Decision_Diagnostic.py) · [통합 결과](results/Automatic_Centering/README.md) |

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

## 독립 원인 분석 실험 — EXP유010·011·012 (2026-11-16)

Size·CoM 정규화 이후 **숫자·문자 정확도가 역전되는 원인**을 각각 독립적인 실험 폴더에서 검증한다. 아래 세 폴더는 일반 가설 로그의 하위가 아니라 **Normal 최상위 폴더와 나란히** 배치되어 있다.

| 독립 실험 폴더 | 연구 질문 | 검증 결과 | 폴더 구성 |
|---|---|---|---|
| [EXP010_Geometry_Distribution](EXP010_Geometry_Distribution/README.md) | 공식 Training과 Hand-made의 글자 크기 분포 차이가 있는가? | B/D/F 긴 변 24px vs 20px 확인 | README·RESULTS·RUN·summary.csv |
| [EXP011_CoM_Effect](EXP011_CoM_Effect/README.md) | Size-only 숫자 정확도 급락은 CoM 보정·재학습 중 무엇 때문인가? | 기존 Size 모델의 숫자 36.46→73.81% (평가 CoM만 변경) | README·RESULTS·RUN·summary.csv |
| [EXP012_Size_Shortcut](EXP012_Size_Shortcut/README.md) | MLP가 글자 크기 관련 특징을 숫자/문자 구분 단서로 쓰는가? | 같은 CoM 모델에서 확대·축소만으로 숫자/문자 판단 반전 | README·RESULTS·RUN·summary.csv |

- **실측 및 증빙:** [원본 Actions #43801194796](https://github.com/ryu980920/Machine_Learning/actions/runs/43801194796) · 재사용한 [EXP유006 체크포인트 #43785390739](https://github.com/ryu980920/Machine_Learning/actions/runs/43785390739).
- **공통 코드:** [exp010_geometry_distribution.py](scripts/exp010_geometry_distribution.py), [exp011_012_causal_probes.py](scripts/exp011_012_causal_probes.py), [exp010_012_aggregate.py](scripts/exp010_012_aggregate.py). EXP유011·012는 원래 한 실행에서 산출했지만 독립 실험 폴더로 문서·요약 결과를 나눠 관리한다.
- **데이터 보존:** 각 `summary.csv`는 선택한 핵심 지표의 축약본. 모든 이미지별 전수 CSV 및 학습 모델 원본은 Actions Artifact에 보존되며 보존 기간 제한이 있다.
- **경계:** 기존 학습 모델 9개를 고정해 분석함. 새 학습, 공식 Original Test·교수님 Test 사용, 최종 모델 교체 없음.

## 저장 경로

- [logs/](logs/README.md): [공통 실험일지 양식](../../EXPERIMENT_TEMPLATE.md)을 복사해 개별 Markdown 저장
- [scripts/](scripts/README.md): 실행 Python·Notebook 및 사용 방법
- [results/](results/README.md): 지표 CSV, Confusion Matrix, Learning Curve, 실제 실행 증빙

**원칙:** 공식 Original Test 16,000장과 교수님 최종 2차 Test는 모델 튜닝에 사용하지 않는다. 교수님 1차 데이터는 최초 Test 후 성능이 부족할 때에만 Validation으로 전환한다. 기록에는 모델 버전과 데이터 누출 여부를 명시한다.

[상위 모델 안내](../README.md) · [05 전체 안내](../../README.md) · [연구 흐름도](../../../01_project/RESEARCH_FLOW.md).

## 가설별 로그

- [초기분석](logs/초기분석/README.md)
- [가설 B](logs/가설B/README.md)
- [가설 A](logs/가설A/README.md)
