# MLP / Combined — 실험 로그

연구 구조는 **Baseline → 초기분석 → 가설 B / 가설 A**로 관리하며, [Normal 연구의 폴더·번호 관리 방식](../../Normal/logs/README.md)을 따른다.

| 구분 | 실험 로그 | 목적 | 상태 |
|---|---|---|---|
| Baseline (원래 경로 유지) | [EXP유001 — Combined 4-Fold CV](EXP유001_Combined_MLP_4Fold_CV_Baseline.md) | 4명 OOF, Normal 대조 및 공식 Validation 비교 | **완료** |
| [초기분석](초기분석/README.md) | [EXP유002 — Error Analysis](초기분석/EXP유002_Error_Analysis.md) · [EXP유003 — Input Sensitivity](초기분석/EXP유003_Input_Sensitivity.md) | 실제 오류·공간/획 특징 진단, 고정 모델 입력 민감도 | **계획 / 미실시** |
| [가설 B](가설B/README.md) | 후속 실험 미등록 | 공간 분포·Size/Center·증강과 4-Fold 재학습 비교 | **계획** |
| [가설 A](가설A/README.md) | 후속 실험 미등록 | `0/D`, `8/B`의 픽셀·획 구분 기준 | **계획** |

**번호 규칙:** Combined의 기존 `EXP유001`은 유지하고, 초기분석은 `EXP유002`부터 연속 사용한다. 가설별 독립 번호 사용이 필요한 경우 Normal 방식처럼 해당 가설의 별도 번호 체계를 명시한다. 실행 ID `EXP-CMB-001` / `EXP-CMB-002` / `EXP-CMB-003`와 로그 표시 번호를 함께 기록해 충돌을 막는다.

**실험일지 기본 순서:** 문제 발견 → 원인 가설 → 실제 통제/재현 방법 → 실측 결과(CM·정량표·그림) → 해석 및 다음 실험. 미실시 실험은 **계획**으로만 작성하고 실제 결과처럼 표시하지 않는다.

결과·코드 연결: [기존 실측 결과](../results/EXP_CMB_001/README.md) · [실행 코드](../scripts/README.md) · [결과 폴더 총괄](../results/README.md).

[최적화 공통 양식](../../../EXPERIMENT_TEMPLATE.md) · [기존 상세 양식](../../../../03_experiments/EXPERIMENT_TEMPLATE.md).
