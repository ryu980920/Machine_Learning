# Extra Trees — Normal 모델 최적화

**현재 상태** 개발 실험 완료. 실측 결과와 재현 절차는 아래에 있다.

## 연구 대상

- **Training:** 공식 MNIST·EMNIST 데이터만 사용. 기존 Baseline Train 76,800장·Validation 19,200장 분할을 기준으로 비교.
- **오류 분석:** 공식 Validation Confusion Matrix와 개발용 Hand-made Confusion Matrix 비교. Hand-made는 Normal Training에 편입하지 않는다.
- **기존 Baseline 실행:** [초기 모델 코드](../../../03_experiments/model_optimization/README.md)에서 `--model extra_trees --variant baseline` 선택.

## 매 실험의 순서

1. **Confusion Matrix와 오분류 사례 분석** — 사용 데이터/레이블, 주요 혼동과 비율 기록.
2. **오차 원인 가설 및 강의 개념 연결** — 이미지·모델 특성과 연결한 검증 가능한 가설.
3. **개선 조건 설정·실행** — 전처리, Training 증강, 데이터 구성, Hyperparameter의 변경 변수 분리.
4. **동일 개발 Validation으로 재평가** — Accuracy·Precision·Recall·F1, Confusion Matrix 및 학습 데이터 크기에 따른 Training / Validation 점수, 학습·추론 시간.
5. **결론 및 후속 실험** — 성공과 실패 모두 실험일지·CSV·그림에 연결.

## 저장 경로

- [logs/](logs/README.md): [공통 실험일지 양식](../../EXPERIMENT_TEMPLATE.md)을 복사해 개별 Markdown 저장
- [scripts/](scripts/README.md): 실행 Python·Notebook 및 사용 방법
- [results/](results/README.md): 지표 CSV, Confusion Matrix, Learning Curve, 실제 실행 증빙

**원칙:** 공식 Original Test 16,000장과 교수님 최종 2차 Test는 모델 튜닝에 사용하지 않는다. 교수님 1차 데이터는 최초 Test 후 성능이 부족할 때에만 Validation으로 전환한다. 기록에는 모델 버전과 데이터 누출 여부를 명시한다.

[상위 모델 안내](../README.md) · [05 전체 안내](../../README.md) · [연구 흐름도](../../../01_project/RESEARCH_FLOW.md).

## 실측 결과

- [보고서](results/Extra_Trees_Normal.pdf)
- [실험 설계](logs/experimental_protocol.md) · [판정 및 결과](logs/findings.md) · [실행 로그](logs/execution.log)
- [선정 설정](results/selection.json) · [후보 비교 CSV](results/experiment_summary.csv)
- [Fold 지표](results/fold_metrics.csv) · [클래스별 CSV](results/class_metrics.csv) · [학습곡선](results/learning_curve.csv)
- [입력 데이터 감사](results/data_audit.json) · [실행 환경](results/environment.json) · [재현 절차](../README.md)

## 후속 검토

- [작성자별 7→9 전이](../Combined/results/review/seven_to_nine_by_writer.csv)
- [예측 변경 샘플의 작성자·행 번호](../Combined/results/review/paired_error_examples.csv)
