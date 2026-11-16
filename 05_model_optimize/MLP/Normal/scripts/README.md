# MLPClassifier / Normal — 실행 코드

실제로 수행한 `.py`, `.ipynb` 등을 저장한다. 코드 파일별 입력 데이터·실행 명령·패키지 버전·Seed를 해당 실험 로그에 연결한다.

**시작점:** [기존 초기 MLPClassifier 코드](../../../../03_experiments/model_optimization/mlp.py). 초깃값을 수정하지 말고 이번 최적화 실험의 실행 코드는 여기에서 독립 버전으로 관리한다.

- **[Pixcel_Attribution.py](Pixcel_Attribution.py)**: 공식 Training 기반으로 기존 Normal MLP를 재학습·검증하고, 공식 Validation/Hand-made의 정상·오분류 이미지 98개에서 Integrated Gradients를 계산한다. 실행 결과는 [EXP유001](../logs/초기분석/EXP유001_Pixcel_Attribution.md)에 기록. **추론 입력 변경이나 하이퍼파라미터 최적화는 하지 않음.**
- [재현 GitHub Actions](../../../../.github/workflows/mlp_pixcel_attribution.yml) — 실제 데이터·코드가 있는 저장소에서 실행하여 [결과](../results/Pixcel_Attribution/README.md)를 생성한다.

- **[Input_Sensitivity.py](Input_Sensitivity.py)**: 고정된 Normal MLP로 입력 변형 20조건+원본을 비교하고, 456,960개 예측 결과를 저장하는 실험. 코드 변경·재학습 모델을 다른 변형마다 변경하지 않음. [EXP유002 로그](../logs/초기분석/EXP유002_Input_Sensitivity.md).
- **[plot_input_sensitivity.py](plot_input_sensitivity.py)**: EXP유002의 CSV에서 공식/손글씨 비교, 숫자/문자 상충, 작성자별 민감도 그래프 3개 재생성.

- **[Spatial_Response_Analysis.py](Spatial_Response_Analysis.py)**: EXP유002의 보관된 Normal MLP 가중치를 수정·재학습하지 않고, 공식/Hand-made 공간 분포·오류 교정 관련성·은닉층/IG 반응을 재현한다. [EXP유003 로그](../logs/가설B/EXP유003_Spatial_Response.md) · [결과](../results/Spatial_Response/README.md) · [수동 GitHub Actions](../../../../.github/workflows/mlp_spatial_response_stage4.yml).

- **[Automatic_Centering_Transformer.py](Automatic_Centering_Transformer.py)**: EXP유004의 레이블·작성자 정보 비사용 정렬 구현. 목표 좌표 (13.5,13.5), 전경 threshold=32, 정수 픽셀 이동·0 채움, Center of Mass / Bounding Box.
- **[CoM_Decision_Diagnostic.py](CoM_Decision_Diagnostic.py)**: **동일한 EXP유004의 두 번째 검증 단계**. 저장된 Normal/CoM MLP를 재학습하지 않고 Hand-made 원본/CoM 이동을 비교하며, 고정 경쟁 클래스(5/F, B/8)의 Signed IG를 산출한다. [통합 로그](../logs/가설B/EXP유004_Automatic_Centering.md) · [전체 산출물](../results/Automatic_Centering/README.md) · [성공 실행](https://github.com/ryu980920/Machine_Learning/actions/runs/43686466736). 원본 코드의 `review-only`는 원래 실행 당시의 상태를 뜻한다.
- **[Automatic_Centering_Experiment.py](Automatic_Centering_Experiment.py)**: 기준 MLP 체크포인트와 두 후보 MLP의 학습·공식 Val/Hand-made 평가를 수행하며, 원본 결과를 GitHub Actions 아티팩트로 보관한다. [EXP유004 정식 로그](../logs/가설B/EXP유004_Automatic_Centering.md) · [배포된 실제 결과](../results/Automatic_Centering/README.md) · [실행 워크플로](../../../../.github/workflows/mlp_exp004_automatic_centering_review.yml). 코드의 `review-only` 설명은 **원래 실행의 배포 전 상태**를 뜻하며, 이후 사용자 승인으로 실험 결과가 배포되었다.

- **[CoM_Translation_Augmentation_Experiment.py](CoM_Translation_Augmentation_Experiment.py)**: [가설 B 실험 5](../logs/가설B/EXP유005_CoM_Translation_Augmentation.md)에 사용한 CoM 이후 ±1px 학습 증강 및 반복 대조 실험 코드.


## EXP유013~016 재현 코드 전체 목록

| 실험 | 학습·평가·집계 코드 |
|---|---|
| **EXP유013** | [기하 특징 이진분류](exp013_geometry_shortcut_binary.py), [3-Seed 통합 집계](exp013_014_aggregate.py) |
| **EXP유014** | [크기·획 증강 학습](exp014_balanced_augmentation.py), [통합 집계](exp013_014_aggregate.py) |
| **EXP유015** | [1차 격자 학습](exp015_scale_stroke_grid.py), [1차 집계](exp015_aggregate_and_rank.py), [미세 탐색](exp015_fine_search.py), [78개 모델 집계·순위](exp015_joint_rank.py), [OfficialVal 크기 개입](exp015_officialval_scale_audit.py), [OfficialVal 집계](exp015_officialval_audit_aggregate.py) |
| **EXP유016** | [60~100% 비율 학습](exp016_high_fraction_grid.py), [16클래스·60모델 평가](exp016_aggregate_shortcut_risks.py), [OfficialVal 5배율 검증](exp016_officialval_scale_audit.py), [OfficialVal 집계](exp016_officialval_audit_aggregate.py) |

**GitHub Actions 실행·보존 코드:** [013·014](../../../../.github/workflows/normal-exp013-014-shortcut-remedy.yml) · [015 1차](../../../../.github/workflows/normal-exp015-shortcut-safe-grid.yml) · [015 미세 탐색](../../../../.github/workflows/normal-exp015-fine-tuning.yml) · [015 검증](../../../../.github/workflows/normal-exp015-officialval-scale-audit.yml) · [016 학습](../../../../.github/workflows/normal-exp016-high-fraction-shortcut.yml) · [016 검증](../../../../.github/workflows/normal-exp016-officialval-scale-audit.yml) · [실측 아카이브](../../../../.github/workflows/normal-exp016-persist-reviewed-logs.yml) · [두 모델 보존](../../../../.github/workflows/normal-exp016-preserve-two-candidates.yml).

[관련 실험일지](../logs/크기편향_원인분석/README.md) · [CSV 및 원본 측정 자료](../results/크기편향_원인분석/README.md). `main`의 최종 모델·배포 코드는 변경하지 않았다.

## EXP유019 — 상단/하단 국소 획 ±1px 증강 재현용 원본 코드 (2026-11-17)

- [국소 위치 증강 학습·평가 원본](exp019_train_local_warp.py): 기존 EXP유017의 80%-Size·14%-Stroke+CoM 재현 후 B1(top)/B2(bottom) 각각 Train 20% 변형, MLP 3 Seed 학습.
- [최종 3 Seed 집계 및 Gate 검사](exp019_summarize_training.py): OfficialVal, 기존 2,560장, 신규 438장 성능과 7↔9, 0.8·1.2배 크기 Family Flip 비교.
- [통합 보고서](../logs/크기편향_원인분석/EXP유019_국소획위치증강_실측보고서.md) · [실측 CSV](../results/크기편향_원인분석/EXP유019_국소획위치증강/EXP019_Decision_Matrix.csv) · [모델 해시](../results/크기편향_원인분석/EXP유019_국소획위치증강/EXP019_model_sha256_manifest.json).

주의: **이 두 코드는 원래 로컬 환경에서 실행된 정확한 파일을 보존**하며 경로 상수가 `/mnt/data`를 가리킨다. GitHub Checkout만으로 즉시 실행되는 스크립트로 수정·검증한 것이 아니고, 6개 신규 joblib 가중치도 GitHub에는 없음. 신규 모델은 연구 목적 B1/B2 학습본이며 **둘 다 불채택**. 배포 모델·추론 파이프라인 변경 없음.
