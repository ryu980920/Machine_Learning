# MLPClassifier — 최적화 기록

- [Normal 최적화](Normal/README.md) — 공식 MNIST·EMNIST Training 기반
- [Combined 최적화](Combined/README.md) — 공식 Training + 별도 Hand-made 학습용 데이터 기반
- **Learning Curve 예시:** Iteration에 따른 Training Loss 및 가능하면 Validation 점수 (실제로 산출한 조건과 x축을 명시)
- **기존 학습 전 기본 코드:** [mlp.py](../../03_experiments/model_optimization/mlp.py)
- **공통 로그 양식:** [EXPERIMENT_TEMPLATE.md](../EXPERIMENT_TEMPLATE.md)

아직 이 폴더에서 최적화 실험을 완료한 것으로 표시하지 않는다.

---

## 2026-11-22 보고서 기반 최적화 개요

**담당: 유용성**

- [Normal 최적화 보고서 핵심 요약](../../04_reports/MLP_Normal_Optimization_Review_20261122.md) — 초기 팀 Hand-made 62.85% → 최종 92.07%, 공식 Validation 최종 96.54%. 최종 256-128 / L2 0.03 / LR .001 / Batch 200 / Size80%-Stroke14%-CoM.
- [Combined 최적화 보고서 핵심 요약](../../04_reports/MLP_Combined_Optimization_Review_20261122.md) — Writer-held-out OOF 81.2500% → 92.3568%, Macro F1 최종 0.923571. 최종 256 / L2 0.0001 / LR .001 / Batch 128 / Size30%-Stroke28%-CoM.

위 지표는 평가 구성이 서로 다르며, 3시드 평균 또는 Writer OOF 평균을 단일 앙상블 정확도로 해석하지 않는다. Original Test 및 교수님 독립 Test 점수는 포함되지 않는다. PDF 원본의 GitHub 등록 여부와 구분한다.

