# MLP / Combined — 실험 로그

연구 구조는 **Baseline → 초기분석 → 가설 B / 가설 A**로 관리한다. [Normal의 폴더/번호 관리](../../Normal/logs/README.md)와 동일한 방식이다.

| 구분 | 실험 로그 | 목적 | 상태 |
|---|---|---|---|
| Baseline | [EXP유001 — Combined 4-Fold CV](EXP유001_Combined_MLP_4Fold_CV_Baseline.md) | Normal 대조 및 4명 작성자 보류 OOF | **완료** |
| [초기분석](초기분석/README.md) | [EXP유002 — Error Analysis](초기분석/EXP유002_Error_Analysis.md) | 0/D·8/B 오류, 공간·형태 특징, A/B 연구 우선순위 결정 | **완료: B 우선, A 보류** |
| [초기분석](초기분석/README.md) | [EXP유003 — Input Sensitivity](초기분석/EXP유003_Input_Sensitivity.md) | 모델 고정 10개 입력 변형, 작성자·클래스·공식 Validation 비교 | **완료: 얇게25% 후속 후보, 95% 축소는 작성자 편중** |
| [가설 B](가설B/README.md) | 후속 재학습 실험 미등록 | Size/Center·증강 효과 검증 | **후속 후보** |
| [가설 A](가설A/README.md) | 독립 후속 실험 미등록 | 의미 있는 획 형태·연결성의 기여 | **조건부 보류** |

**번호 규칙:** Combined의 기존 EXP유001을 유지하고 초기분석은 EXP유002, EXP유003 순서로 기록한다. 실행 ID는 별도로 EXP-CMB-001~003으로 보존한다. 가설별 독립 번호가 필요한 경우 Normal처럼 명시한다.

**실험일지 흐름:** 문제 발견 → 가설 → 통제 조건 → 실측 결과(원본·CM·그림) → 가설 판단 → 후속 연구. 미실시 실험은 계획으로만 표기한다.

[결과 폴더 안내](../results/README.md) · [실행 코드](../scripts/README.md) · [05 공통 양식](../../../EXPERIMENT_TEMPLATE.md).
