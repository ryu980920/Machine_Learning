# MLPClassifier — Combined 모델 최적화

**현재 상태:** [EXP-CMB-001 Baseline](logs/EXP유001_Combined_MLP_4Fold_CV_Baseline.md), [EXP-CMB-002 오분류 분석](logs/초기분석/EXP유002_Error_Analysis.md), [EXP-CMB-003 입력 민감도](logs/초기분석/EXP유003_Input_Sensitivity.md), [EXP-CMB-004 Size/CoM 정규화](logs/가설B/EXP유001_Geometry_Normalization_Results.md) **완료**. 이어 **[EXP-CMB-005~007 Size × Stroke 증강의 총 31조건 비교](logs/가설B/EXP유004_Size60_100_Final_Results.md)**와 **[상위 3개 오분류율 검토](logs/가설B/EXP유004_Size60_100_Top3_OOF_Misclassification_Review.md)**까지 실측·기록·`main` 병합 완료. **최종 학습 Pipeline/모델 선정은 보류.**

Normal과 같은 **초기분석 → 가설 B / 가설 A** 연구 구조를 사용한다. 현재 EXP-CMB-002의 연구적 결론은 **가설 B(공간·크기·중심·명암) 우선 검증**, **가설 A의 독립 획 연결 구조 실험은 조건부 보류**다. 이는 다음 개선 실험을 선택할 근거이며, 새로운 개선 모델이 완성되었다는 의미가 아니다.

## 등록된 실험

| 구분 | 실험 | 상태 | 근거 |
|---|---|---|---|
| Baseline | [EXP유001 — Combined 4-Fold CV](logs/EXP유001_Combined_MLP_4Fold_CV_Baseline.md) | **완료** | [기존 실측](results/EXP_CMB_001/README.md) · [실행 코드](scripts/combined_mlp_writer_cv.py) |
| 초기분석 | [EXP유002 — Error Analysis](logs/초기분석/EXP유002_Error_Analysis.md) | **완료** | [실측 결과·실제 이미지](results/Error_Analysis/README.md) |
| 초기분석 | [EXP유003 — Input Sensitivity](logs/초기분석/EXP유003_Input_Sensitivity.md) | **실행 완료** | [원본 실측](results/Input_Sensitivity/README.md) |
| 가설 B | [EXP유001 — Geometry Normalization](logs/가설B/EXP유001_Geometry_Normalization_Results.md) | **실험·설명 대조 완료** | [사전등록 기준](logs/가설B/EXP유001_Geometry_Normalization.md) · [실측 결과](results/Geometry_Normalization/README.md) |
| 가설 B | [EXP유002 / CMB-005 — Size 0~30% × Stroke](logs/가설B/EXP유002_Balanced_Scale_Stroke_Final_Results.md) | **120모델·10조건 완료** | [원본 10조건](results/Balanced_Scale_Stroke/full_comparison/SUMMARY.md) |
| 가설 B | [EXP유003 / CMB-006 — Size 40~50% × Stroke](logs/가설B/EXP유003_Size40_50_Stroke_Extension.md) | **72모델·6조건 완료** | [원본 16조건](results/Balanced_Scale_Stroke_Extension/full_0to50_comparison/SUMMARY.md) |
| 가설 B | [EXP유004 / CMB-007 — Size 60~100% × Stroke](logs/가설B/EXP유004_Size60_100_Final_Results.md) | **180모델·15조건 완료, 총 31조건 비교** | [31조건 실측](results/Balanced_Scale_Stroke_HighFraction/full_0to100_comparison/SUMMARY.md) · [상위 3개 오분류율](logs/가설B/EXP유004_Size60_100_Top3_OOF_Misclassification_Review.md) |

## Baseline 결과 — Hand-made OOF 2,560장

| 지표 | Normal 대조 | Combined |
|---|---:|---:|
| Accuracy | 62.5781% | **81.2500%** |
| Macro F1 | 0.623046 | **0.812043** |
| 정답 수 | 1,602 | **2,080** |

각 Combined Fold는 공식 Training 76,800장과 **보류 작성자를 제외한 Hand-made 1,920장**으로 새로 학습한다. OOF는 4개 모델의 개발 평가이며 단일 최종 모델/독립 Test 성능이 아니다.

EXP-CMB-002에서 **원본 2,560장 예측을 재현**하고, 오류와 정답의 공간·잉크 특징을 실제로 분석했다. 작성자 43번 8→B에서 너비·잉크 비율, 41번 D→0에서 x 중심 차이를 확인했지만, 전체 작성자·클래스에 동일한 처리를 해야 한다는 증거는 없다. 이어 [EXP유003](logs/초기분석/EXP유003_Input_Sensitivity.md)에서 기존 모델을 고정한 10가지 단일 입력 변형을 실측했다. **95% 축소는 OOF +2.6172%p(43번 작성자에 집중, 공식 Validation 하락), 획 얇게25%는 +1.0547%p**였다. 일률적 Center 이동은 악화했고, 후속 재학습 후보는 **획 얇게25%**로 선별했다. 입력만 변형한 진단 결과이지 새 모델의 개선치는 아니다.

## EXP-CMB-004 정규화 방법 비교

- **BBox Size-only OOF 80.7813%:** 문자 Accuracy **94.5833%**, 숫자 **72.5000%**. 크기 표준화는 특정 클래스 판단을 개선하면서 다른 클래스 오답을 증가시켰다.
- **CoM-only OOF 85.8594%:** 숫자 **89.3125%**, 문자 **80.1042%**. CoM 위치 정렬의 효과는 **정렬된 데이터로 모델을 새로 학습할 때** 크게 나타났다.
- **Size+CoM OOF 88.9453%:** 숫자 **86.6875%**, 문자 **92.7083%**로 각 단일 방법의 상충이 완화되었다. 하지만 공식 Validation은 **네 Fold 모두 감소**했다.
- **통과 기준은 숫자 점수 이상이 아닌 연구 설명 가능성**으로 사전등록했으며, 원인 관련 지표·오답 교정/신규·기존 MLP 입력만 변경 vs 재학습을 근거로 판단했다. 결과를 최종 모델이라고 표시하지 않는다.

## EXP-CMB-005~007 — Size × Stroke 증강 후속 연구

- **기본 MLP 하이퍼파라미터 고정**, Seed 42/43/44 및 4-Writer Held-out OOF 검증. CMB-005/006 기존 16조건과 CMB-007 신규 15조건을 합친 **총 31조건**의 실측 비교.
- OOF 정확도 상위 조합은 **30%-28% 90.716% / 30%-14% 90.690% / 30%-0% 90.573%**이며, 각각의 **오분류율은 9.284% / 9.310% / 9.427%**.
- [전체 31조건·16클래스·240방향 원본 분석](logs/가설B/EXP유004_Size60_100_Final_Results.md)에서 고비율 증강의 오류 재분배와 안정성 한계를 기록했다. [후속 오분류율 분석](logs/가설B/EXP유004_Size60_100_Top3_OOF_Misclassification_Review.md)에서 8/B/D 오류와 입력 크기별 평균 계열 확률을 분리해 검토했다.
- Original Test·교수님 Test·Normal 신규 손글씨 480장을 이 Combined 실험의 모델 선택에 사용하지 않았다. **로그/코드/실측은 `main`에 반영했으나 최종 모델 교체는 하지 않음.**

## 연구 경로 및 원칙

- [logs/](logs/README.md): Baseline, [초기분석](logs/초기분석/README.md), [가설 B](logs/가설B/README.md), [가설 A](logs/가설A/README.md).
- [scripts/](scripts/README.md): 실행·재현·픽셀 전처리 코드.
- [results/](results/README.md): 원본 수치/CSV/이미지/모델/실행 이력. Baseline 폴더는 수정하지 않는다.

**데이터 누출 통제:** 작성자 41–44 Leave-One-Writer-Out 유지. Fold별 학습 작성자와 보류 작성자를 겹치지 않는다. 공식 Original Test·교수님 최종 Test를 실험 조건 선택에 사용하지 않는다. 개발용 Hand-made 반복 탐색의 과적합 가능성을 기록한다.

[05 공통 양식](../../EXPERIMENT_TEMPLATE.md) · [연구 흐름도](../../../01_project/RESEARCH_FLOW.md).
