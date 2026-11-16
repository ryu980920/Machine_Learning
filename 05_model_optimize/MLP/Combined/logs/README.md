# MLP / Combined — 실험 로그

연구 구조는 **Baseline → 초기분석 → 가설 B / 가설 A**로 관리한다. [Normal의 폴더/번호 관리](../../Normal/logs/README.md)와 동일한 방식이다.

| 구분 | 실험 로그 | 목적 | 상태 |
|---|---|---|---|
| Baseline | [EXP유001 — Combined 4-Fold CV](EXP유001_Combined_MLP_4Fold_CV_Baseline.md) | Normal 대조 및 4명 작성자 보류 OOF | **완료** |
| [초기분석](초기분석/README.md) | [EXP유002 — Error Analysis](초기분석/EXP유002_Error_Analysis.md) | 0/D·8/B 오류, 공간·형태 특징, A/B 연구 우선순위 결정 | **완료: B 우선, A 보류** |
| [초기분석](초기분석/README.md) | [EXP유003 — Input Sensitivity](초기분석/EXP유003_Input_Sensitivity.md) | 모델 고정 10개 입력 변형, 작성자·클래스·공식 Validation 비교 | **완료: 얇게25% 후속 후보, 95% 축소는 작성자 편중** |
| [가설 B](가설B/README.md) | [EXP유001 — 사전등록](가설B/EXP유001_Geometry_Normalization.md) · [EXP유001 — 결과/판정](가설B/EXP유001_Geometry_Normalization_Results.md) | BBox Size vs CoM 중심, 4-Fold 재학습 및 입력/가중치 분리 | **완료: 역할·부작용 설명, 모델 채택 보류** |
| [가설 B](가설B/README.md) | [EXP유002 / EXP-CMB-005 — Size 0~30% × Stroke](가설B/EXP유002_Balanced_Scale_Stroke_Final_Results.md) | 저비율 증강 10조건, 120개 MLP 및 클래스별 오류 검증 | **실험·원인분석 완료, 모델 채택 보류** |
| [가설 B](가설B/README.md) | [EXP유003 / EXP-CMB-006 — Size 40~50% × Stroke](가설B/EXP유003_Size40_50_Stroke_Extension.md) | 추가 6조건, 72개 MLP 및 기존 16조건 비교 | **실험·비교 완료, 모델 채택 보류** |
| [가설 B](가설B/README.md) | [EXP유004 / EXP-CMB-007 — Size 60~100% × Stroke](가설B/EXP유004_Size60_100_Final_Results.md) · [OOF 상위 3개·오분류율](가설B/EXP유004_Size60_100_Top3_OOF_Misclassification_Review.md) | 추가 15조건·180개 MLP, 총 31조건·16클래스·240방향·크기별 확률/오분류율 | **실험·후속분석 완료, 모델 채택 보류** |
| [가설 A](가설A/README.md) | 독립 후속 실험 미등록 | 의미 있는 획 형태·연결성의 기여 | **조건부 보류** |

**번호 규칙:** Combined의 기존 EXP유001을 유지하고 초기분석은 EXP유002, EXP유003 순서로 기록한다. 실행 ID는 별도로 EXP-CMB-001~007로 보존한다. 가설 B EXP유001은 해당 가설 폴더 안의 번호이며, 기존 Combined Baseline EXP유001과 경로/실험 ID가 다르다. 가설별 독립 번호가 필요한 경우 Normal처럼 명시한다.

**실험일지 흐름:** 문제 발견 → 가설 → 통제 조건 → 실측 결과(원본·CM·그림) → 가설 판단 → 후속 연구. 미실시 실험은 계획으로만 표기한다.

[결과 폴더 안내](../results/README.md) · [실행 코드](../scripts/README.md) · [05 공통 양식](../../../EXPERIMENT_TEMPLATE.md).
