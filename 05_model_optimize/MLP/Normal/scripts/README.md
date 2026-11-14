# MLPClassifier / Normal — 실행 코드

실제로 수행한 `.py`, `.ipynb` 등을 저장한다. 코드 파일별 입력 데이터·실행 명령·패키지 버전·Seed를 해당 실험 로그에 연결한다.

**시작점:** [기존 초기 MLPClassifier 코드](../../../../03_experiments/model_optimization/mlp.py). 초깃값을 수정하지 말고 이번 최적화 실험의 실행 코드는 여기에서 독립 버전으로 관리한다.

- **[Pixcel_Attribution.py](Pixcel_Attribution.py)**: 공식 Training 기반으로 기존 Normal MLP를 재학습·검증하고, 공식 Validation/Hand-made의 정상·오분류 이미지 98개에서 Integrated Gradients를 계산한다. 실행 결과는 [EXP유001](../logs/EXP유001_Pixcel_Attribution.md)에 기록. **추론 입력 변경이나 하이퍼파라미터 최적화는 하지 않음.**
- [재현 GitHub Actions](../../../../.github/workflows/mlp_pixcel_attribution.yml) — 실제 데이터·코드가 있는 저장소에서 실행하여 [결과](../results/Pixcel_Attribution/README.md)를 생성한다.

- **[Input_Sensitivity.py](Input_Sensitivity.py)**: 고정된 Normal MLP로 입력 변형 20조건+원본을 비교하고, 456,960개 예측 결과를 저장하는 실험. 코드 변경·재학습 모델을 다른 변형마다 변경하지 않음. [EXP유002 로그](../logs/EXP유002_Input_Sensitivity.md).
- **[plot_input_sensitivity.py](plot_input_sensitivity.py)**: EXP유002의 CSV에서 공식/손글씨 비교, 숫자/문자 상충, 작성자별 민감도 그래프 3개 재생성.
