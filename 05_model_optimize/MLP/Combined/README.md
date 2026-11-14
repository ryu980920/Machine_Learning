# MLPClassifier — Combined 모델 최적화

**현재 상태: 실험 로그 미등록.** 새 파일을 기록할 때 실제 실행 결과와 계획을 구분한다.

## 연구 대상

- **Training:** 공식 MNIST·EMNIST Training + 분리된 Hand-made Training 데이터. 정확한 수량·작성자·Fold는 실험마다 기록.
- **Validation:** 같은 작성자의 학습/검증 중복을 피한 작성자 그룹 CV 또는 별도 신규 작성자 Validation. Fold마다 MLPClassifier 모델을 새로 학습하고, 각 Fold에서 보류 작성자에 대한 Confusion Matrix를 얻는다.
- **주의:** 아직 Combined 학습/Group CV Runner가 기존 [초기 모델 실행 코드](../../../03_experiments/model_optimization/README.md)에 구현된 것은 아니다. 실제 구현과 검증 내역을 반드시 함께 저장.

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
