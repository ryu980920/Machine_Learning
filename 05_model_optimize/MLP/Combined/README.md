# MLPClassifier — Combined 모델 최적화

**현재 상태:** EXP-CMB-001 통제 Baseline **완료**(2026-11-15), 후속 원인 분석·입력 민감도·재학습 최적화 **미실시**.

Normal과 동일하게 연구 과정을 **초기분석 → 가설 B → 가설 A**로 관리하며, 실제 검증 순서에 따라 두 가설의 실험을 병행하거나 조정할 수 있다.

## 등록한 실험

| 구분 | 실험 | 상태 | 근거 |
|---|---|---|---|
| Baseline | **[EXP유001 — Combined MLP 4-Fold CV](logs/EXP유001_Combined_MLP_4Fold_CV_Baseline.md)** | **완료** | [실측 결과](results/EXP_CMB_001/README.md) · [코드](scripts/combined_mlp_writer_cv.py) |
| 초기분석 | **[EXP유002 — Error Analysis](logs/초기분석/EXP유002_Error_Analysis.md)** | **계획 / 미실시** | [실험 목적·통제·예정 산출물](logs/초기분석/README.md) |
| 초기분석 | **[EXP유003 — Input Sensitivity](logs/초기분석/EXP유003_Input_Sensitivity.md)** | **계획 / 미실시** | [실험 목적·통제·예정 산출물](logs/초기분석/README.md) |

## Baseline 결과 — Hand-made OOF 2,560장

| 지표 | Normal 대조군 | Combined |
|---|---:|---:|
| Accuracy | 62.5781% | **81.2500%** |
| Macro F1 | 0.623046 | **0.812043** |
| 정답 수 | 1,602 | **2,080** |

Combined는 각 Fold에서 **공식 Training 76,800장 + 보류 작성자를 제외한 Hand-made 1,920장**으로 MLP를 새로 학습하고, 보류 작성자 640장씩을 검증했다. 이 OOF 결과는 **4개 별도 학습 모델을 합한 개발 평가**로, 단일 최종 모델 또는 독립 Test 정확도가 아니다.

**남은 문제:** `0↔D` 오류 59장, `8↔B` 오류 53장, 41·43번 작성자 성능 편차와 특정 Fold의 공식 개발 Validation 저하. 이미지 특징과 판단 근거는 후속 연구 과제이며 아직 새로운 실측 원인으로 확정하지 않았다.

## 연구 폴더 및 로그 흐름

- **[logs/](logs/README.md)** — Baseline, [초기분석](logs/초기분석/README.md), [가설 B](logs/가설B/README.md)(위치·크기·정렬), [가설 A](logs/가설A/README.md)(유사 문자 판단 기준).
- **[scripts/](scripts/README.md)** — 재현 Python, 공통 전처리, CM 렌더러, 실행 환경·Actions. 후속 실험 코드는 실행 준비가 되었을 때 추가.
- **[results/](results/README.md)** — 실험별 실제 CSV/PNG/모델·재현 메타데이터. [EXP-CMB-001](results/EXP_CMB_001/README.md)은 보존하고, [Error_Analysis](results/Error_Analysis/README.md) 및 [Input_Sensitivity](results/Input_Sensitivity/README.md)는 현재 **빈 계획 폴더(README만 존재)**이다.

매 실험은 **문제 발견(CM) → 원인 가설 → 개선 방법·통제 조건 → 실측 결과 → 가설 판단 → 후속 연구**를 따른다. 실제 결과 그림은 해당 로그에도 삽입하고, 표·그림 수치는 원본 파일과 일치해야 한다.

## 데이터 누출 방지 및 재현 원칙

- Hand-made 작성자 **41/42/43/44 Leave-One-Writer-Out 4-Fold** 유지. 보류 작성자의 이미지는 해당 Fold Training·증강·전처리 통계 추정에 사용하지 않는다.
- 공식 Training/개발 Validation의 원래 Group 분할 유지. 공식 Original Test 16,000장·교수님 최종 Test를 튜닝에 사용하지 않는다.
- 진단 실험은 **기존 EXP-CMB-001의 저장된 가중치/OOF 예측을 우선 재활용**한다. 새 학습이 필요한 개선 실험은 기존 결과를 덮어쓰지 않고 별도의 실험 ID와 결과 폴더를 사용한다.
- 이미 게시된 CM은 Viridis Count 및 행 정규화(%)를 사용한다. 분석의 원본 근거는 [EXP-CMB-001 manifest](results/EXP_CMB_001/experiment_manifest.json)와 [OOF 예측](results/EXP_CMB_001/OOF_predictions.csv)에 보존한다.

[Normal 연구 구조 참고](../Normal/logs/README.md) · [05 공통 양식](../../EXPERIMENT_TEMPLATE.md) · [연구 흐름도](../../../01_project/RESEARCH_FLOW.md).
