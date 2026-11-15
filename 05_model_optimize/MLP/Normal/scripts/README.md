# MLPClassifier / Normal — 실행 코드

실제로 수행한 `.py`, `.ipynb` 등을 저장한다. 코드 파일별 입력 데이터·실행 명령·패키지 버전·Seed를 해당 실험 로그에 연결한다.

**시작점:** [기존 초기 MLPClassifier 코드](../../../../03_experiments/model_optimization/mlp.py). 초깃값을 수정하지 말고 이번 최적화 실험의 실행 코드는 여기에서 독립 버전으로 관리한다.

- **[Pixcel_Attribution.py](Pixcel_Attribution.py)**: 공식 Training 기반으로 기존 Normal MLP를 재학습·검증하고, 공식 Validation/Hand-made의 정상·오분류 이미지 98개에서 Integrated Gradients를 계산한다. 실행 결과는 [EXP유001](../logs/EXP유001_Pixcel_Attribution.md)에 기록. **추론 입력 변경이나 하이퍼파라미터 최적화는 하지 않음.**
- [재현 GitHub Actions](../../../../.github/workflows/mlp_pixcel_attribution.yml) — 실제 데이터·코드가 있는 저장소에서 실행하여 [결과](../results/Pixcel_Attribution/README.md)를 생성한다.

- **[Input_Sensitivity.py](Input_Sensitivity.py)**: 고정된 Normal MLP로 입력 변형 20조건+원본을 비교하고, 456,960개 예측 결과를 저장하는 실험. 코드 변경·재학습 모델을 다른 변형마다 변경하지 않음. [EXP유002 로그](../logs/EXP유002_Input_Sensitivity.md).
- **[plot_input_sensitivity.py](plot_input_sensitivity.py)**: EXP유002의 CSV에서 공식/손글씨 비교, 숫자/문자 상충, 작성자별 민감도 그래프 3개 재생성.

- **[Spatial_Response_Analysis.py](Spatial_Response_Analysis.py)**: EXP유002의 보관된 Normal MLP 가중치를 수정·재학습하지 않고, 공식/Hand-made 공간 분포·오류 교정 관련성·은닉층/IG 반응을 재현한다. [EXP유003 로그](../logs/EXP유003_Spatial_Response.md) · [결과](../results/Spatial_Response/README.md) · [수동 GitHub Actions](../../../../.github/workflows/mlp_spatial_response_stage4.yml).

- **[Automatic_Centering_Transformer.py](Automatic_Centering_Transformer.py)**: EXP유004의 레이블·작성자 정보 비사용 정렬 구현. 목표 좌표 (13.5,13.5), 전경 threshold=32, 정수 픽셀 이동·0 채움, Center of Mass / Bounding Box.
- **[CoM_Decision_Diagnostic.py](CoM_Decision_Diagnostic.py)**: **동일한 EXP유004의 두 번째 검증 단계**. 저장된 Normal/CoM MLP를 재학습하지 않고 Hand-made 원본/CoM 이동을 비교하며, 고정 경쟁 클래스(5/F, B/8)의 Signed IG를 산출한다. [통합 로그](../logs/EXP유004_Automatic_Centering.md) · [전체 산출물](../results/Automatic_Centering/README.md) · [성공 실행](https://github.com/ryu980920/Machine_Learning/actions/runs/43686466736). 원본 코드의 `review-only`는 원래 실행 당시의 상태를 뜻한다.
- **[Automatic_Centering_Experiment.py](Automatic_Centering_Experiment.py)**: 기준 MLP 체크포인트와 두 후보 MLP의 학습·공식 Val/Hand-made 평가를 수행하며, 원본 결과를 GitHub Actions 아티팩트로 보관한다. [EXP유004 정식 로그](../logs/EXP유004_Automatic_Centering.md) · [배포된 실제 결과](../results/Automatic_Centering/README.md) · [실행 워크플로](../../../../.github/workflows/mlp_exp004_automatic_centering_review.yml). 코드의 `review-only` 설명은 **원래 실행의 배포 전 상태**를 뜻하며, 이후 사용자 승인으로 실험 결과가 배포되었다.
