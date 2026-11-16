# MLPClassifier — Normal 모델 최적화

**현재 상태:** 초기분석·가설 A/B 실험을 구분해 기록하며, EXP유010·011·012의 원인 분석은 `logs/크기편향_원인분석/`의 **단일 연구 폴더**에서 관리한다. 모델·결과 원본의 위치는 유지하며 최종 Pipeline 반영은 별도로 결정한다.

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

## 크기 기반 정확도 역전 원인과 완화 실험 — EXP유010~014

기존 원인 분석 3실험과 후속 직접 검증/완화 학습 2실험을 **[통합 연구 폴더](logs/크기편향_원인분석/README.md)**에서 함께 관리한다. 이 폴더는 `logs/가설A`, `logs/가설B`와 같은 깊이에 있다.

| ID | 내용 | 실험일지 / 요약 CSV |
|---|---|---|
| **EXP유010** | 공식 Training과 Hand-made의 글자 크기 분포 비교 | [실험일지](logs/크기편향_원인분석/EXP유010_크기분포_분석.md) · [CSV](results/크기편향_원인분석/EXP유010_summary.csv) |
| **EXP유011** | CoM 재정렬과 재학습 효과 분리 | [실험일지](logs/크기편향_원인분석/EXP유011_Size_CoM_교차분석.md) · [CSV](results/크기편향_원인분석/EXP유011_summary.csv) |
| **EXP유012** | 글자 크기 기반 Shortcut 민감도 | [실험일지](logs/크기편향_원인분석/EXP유012_크기분류단서.md) · [CSV](results/크기편향_원인분석/EXP유012_summary.csv) |
| **EXP유013** | 기하 특징 4개만으로 숫자·문자 이진분류, 데이터 Shortcut 확인 | [실측 실험일지](logs/크기편향_원인분석/EXP유013_기하특징_이진분류.md) · [CSV](results/크기편향_원인분석/EXP유013_binary_4feature_summary.csv) |
| **EXP유014** | Class-balanced Scale / Stroke Training으로 MLP Shortcut 완화 실험 | [실측 실험일지](logs/크기편향_원인분석/EXP유014_Shortcut_완화_학습.md) · [정확도 CSV](results/크기편향_원인분석/EXP유014_accuracy_3seed_summary.csv) · [크기 민감도 CSV](results/크기편향_원인분석/EXP유014_scale_flip_3seed_summary.csv) |

공통 실행 코드는 기존 `scripts/`에, EXP유010~012 원본은 [Actions #43801194796](https://github.com/ryu980920/Machine_Learning/actions/runs/43801194796), EXP유013~014 원본은 [3-Seed 성공 Actions #43810141307](https://github.com/ryu980920/Machine_Learning/actions/runs/43810141307)에 보존한다. 이번 후속 실험의 공식 Validation 목표 97.30%는 미달했다. 최종 모델·데이터는 변경하지 않는다.

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
- [크기편향 원인분석 — EXP유010~014](logs/크기편향_원인분석/README.md)


## EXP유015 — 크기 Shortcut을 먼저 억제하는 전처리 최적화

[실측 실험일지](logs/크기편향_원인분석/EXP유015_증강비율_획강도_최적화.md) · [78모델 중 Gate 합격 9개 결과 CSV](results/크기편향_원인분석/EXP유015_shortcut_gate_통과9조건_요약.csv) · [OfficialVal 변형 진단 CSV](results/크기편향_원인분석/EXP유015_OfficialVal_크기변형_요약.csv).

Training에만 Class-balanced Scale(10~50%, 중간 25/35% 포함) 및 Stroke(0~42% 강도)를 적용하여 **78개 신규 MLP**를 3 Seed로 학습했다. Shortcut 재발 방지 기준을 먼저 적용해 9개 후보를 선별한 뒤 공식 Validation으로 비교했다. 최고 공식 Validation은 **30% Scale / 28% Stroke / CoM의 96.1806%**, 공식 목표 97.30%는 아직 미달. Hand-made B·8 등 잔여 크기 민감도로 최종 모델 채택 보류. 전체 원본 결과는 [Github Actions](https://github.com/ryu980920/Machine_Learning/actions/runs/43815571141) 및 [공식 Val 배율 진단](https://github.com/ryu980920/Machine_Learning/actions/runs/43817096445) 참조.


## EXP유016 — 60~100% Training 전용 증강과 16클래스 Shortcut 검증

[EXP유016 계획·전체 실측·교차 검증](logs/크기편향_원인분석/EXP유016_고비율_크기증강_한계확인.md) · [23개 후보별 Gate 판정 CSV](results/크기편향_원인분석/원본_실측_아카이브/EXP유016_60모델/all_23_conditions_ranked_gates.csv) · [16클래스 5배율 전환율 CSV](results/크기편향_원인분석/원본_실측_아카이브/EXP유016_60모델/all_16class_size_flip_3seed.csv) · [공식 Validation 3-Seed 5배율 교차 검증 CSV](results/크기편향_원인분석/원본_실측_아카이브/EXP유016_OfficialVal/officialval_scale_3seed_summary.csv).

60/70/80/90/100% 크기 증강 × 0/14/28/42% 획 혼합 × Seed42/43/44 = **60개 신규 MLP 실측 성공**. EXP유015의 기존 30-28·40-28·50-28 고정 체크포인트와 비교. 사전등록한 Hand-made 개별 B·8·D 20% Family flip Gate를 통과한 모델은 6개. 70-14와 80-14가 Hand-made 최악 17.50%로 상위이나, OfficialVal 개별 D/B에서 25~28% 전환 잔존. **70-0**은 Hand-made 최악 18.54% 및 OfficialVal 17.03%로 두 개발 집단에서 균형적이며 공식 Val 95.28%. 추가 독립 Writer 검증 전에는 final/Main 변경·모델 교체 없음. 97.30% 공식 Val 목표 미달.


### EXP유010~016 코드·로그·결과 위치

[logs — 실험일지](logs/크기편향_원인분석/README.md) · [scripts — 학습/평가 Python](scripts/README.md) · [results — CSV·검증 원본](results/크기편향_원인분석/README.md). 재현용 코드만 추가하며 기존 최종 모델은 변경하지 않는다.


## EXP유017 — 80-14 MLP 하이퍼파라미터 최적화·모델 동결

[전체 선정 기준·하이퍼파라미터 변경 근거·실측 로그](logs/크기편향_원인분석/EXP유017_80-14_MLP_하이퍼파라미터_최적화.md) · [OfficialVal 5배율 결과](results/크기편향_원인분석/EXP유017_OfficialVal_5배율_3Seed_요약.csv) · [SHA-256 기록](results/크기편향_원인분석/EXP유017_동결모델_SHA256_Manifest.json).

80-14+CoM 전처리 고정 후 **26+20조건, Seed 42·43·44, 총 138개 MLP**를 평가. 사전 Gate 통과 10/46조건 중 정확도가 최고인 **`(256,128), alpha=0.03, LR=0.001, early_stopping=False`**를 선택했다. OfficialVal **96.5365%**, B·8·D 최악 Family Flip **16.67%**, 5배율 안정성 검사 통과. 목표 97.30% 미달. 동결 후 Hand-made 2,560장 및 별도 수집 480장의 평가 결과는 별도 기록하였고, 모델 변경에는 사용하지 않았다. 현재 `main`에는 **실험 문서와 결과 지표만** 반영하며 학습 코드·동결 체크포인트·배포 모델을 변경하지 않는다.

## EXP유018 — 신규 수집 손글씨 480장 판독 필터 평가

[원인분석 실험일지](logs/크기편향_원인분석/EXP유018_신규손글씨480_판독필터_혼동쌍_진단.md) · [3-Seed 성능 CSV](results/크기편향_원인분석/EXP유018_3Seed_그룹별성능.csv) · [이전/신규 CM 주요 혼동쌍](results/크기편향_원인분석/EXP유018_이전CM_신규CM_주요쌍비교.csv).

EXP유017 동결 MLP로 별도 수집 데이터 **480장 전체 83.06%**, 판독 품질 제외 25장 후 **455장 85.57%**. 신규 CM의 7→9·7→A·4→F 증가와 원본 이미지 그룹별 집중을 기록했으며, 원인 가설은 아직 검증 중. 원본 이미지·NPZ 및 모델 가중치 미업로드, 배포 모델 수정 없음.
