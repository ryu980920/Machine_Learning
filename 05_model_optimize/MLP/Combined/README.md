# MLPClassifier — Combined 모델 최적화

**현재 상태:** [EXP-CMB-001 Baseline](logs/EXP유001_Combined_MLP_4Fold_CV_Baseline.md) 완료, [EXP-CMB-002 오분류 원인 분석](logs/초기분석/EXP유002_Error_Analysis.md) **실행·검증 완료**, [EXP-CMB-003 입력 민감도](logs/초기분석/EXP유003_Input_Sensitivity.md) **실행·검증 완료**; 실제 새 모델 재학습은 아직 미실시.

Normal과 같은 **초기분석 → 가설 B / 가설 A** 연구 구조를 사용한다. 현재 EXP-CMB-002의 연구적 결론은 **가설 B(공간·크기·중심·명암) 우선 검증**, **가설 A의 독립 획 연결 구조 실험은 조건부 보류**다. 이는 다음 개선 실험을 선택할 근거이며, 새로운 개선 모델이 완성되었다는 의미가 아니다.

## 등록된 실험

| 구분 | 실험 | 상태 | 근거 |
|---|---|---|---|
| Baseline | [EXP유001 — Combined 4-Fold CV](logs/EXP유001_Combined_MLP_4Fold_CV_Baseline.md) | **완료** | [기존 실측](results/EXP_CMB_001/README.md) · [실행 코드](scripts/combined_mlp_writer_cv.py) |
| 초기분석 | [EXP유002 — Error Analysis](logs/초기분석/EXP유002_Error_Analysis.md) | **완료** | [실측 결과·실제 이미지](results/Error_Analysis/README.md) |
| 초기분석 | [EXP유003 — Input Sensitivity](logs/초기분석/EXP유003_Input_Sensitivity.md) | **실행 완료** | [원본 실측](results/Input_Sensitivity/README.md) |

## Baseline 결과 — Hand-made OOF 2,560장

| 지표 | Normal 대조 | Combined |
|---|---:|---:|
| Accuracy | 62.5781% | **81.2500%** |
| Macro F1 | 0.623046 | **0.812043** |
| 정답 수 | 1,602 | **2,080** |

각 Combined Fold는 공식 Training 76,800장과 **보류 작성자를 제외한 Hand-made 1,920장**으로 새로 학습한다. OOF는 4개 모델의 개발 평가이며 단일 최종 모델/독립 Test 성능이 아니다.

EXP-CMB-002에서 **원본 2,560장 예측을 재현**하고, 오류와 정답의 공간·잉크 특징을 실제로 분석했다. 작성자 43번 8→B에서 너비·잉크 비율, 41번 D→0에서 x 중심 차이를 확인했지만, 전체 작성자·클래스에 동일한 처리를 해야 한다는 증거는 없다. 이어 [EXP유003](logs/초기분석/EXP유003_Input_Sensitivity.md)에서 기존 모델을 고정한 10가지 단일 입력 변형을 실측했다. **95% 축소는 OOF +2.6172%p(43번 작성자에 집중, 공식 Validation 하락), 획 얇게25%는 +1.0547%p**였다. 일률적 Center 이동은 악화했고, 후속 재학습 후보는 **획 얇게25%**로 선별했다. 입력만 변형한 진단 결과이지 새 모델의 개선치는 아니다.

## 연구 경로 및 원칙

- [logs/](logs/README.md): Baseline, [초기분석](logs/초기분석/README.md), [가설 B](logs/가설B/README.md), [가설 A](logs/가설A/README.md).
- [scripts/](scripts/README.md): 실행·재현·픽셀 전처리 코드.
- [results/](results/README.md): 원본 수치/CSV/이미지/모델/실행 이력. Baseline 폴더는 수정하지 않는다.

**데이터 누출 통제:** 작성자 41–44 Leave-One-Writer-Out 유지. Fold별 학습 작성자와 보류 작성자를 겹치지 않는다. 공식 Original Test·교수님 최종 Test를 실험 조건 선택에 사용하지 않는다. 개발용 Hand-made 반복 탐색의 과적합 가능성을 기록한다.

[05 공통 양식](../../EXPERIMENT_TEMPLATE.md) · [연구 흐름도](../../../01_project/RESEARCH_FLOW.md).
