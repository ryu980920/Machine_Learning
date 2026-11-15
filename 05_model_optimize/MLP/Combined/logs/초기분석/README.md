# MLP / Combined — 초기분석

## 검증 목적

기존 [EXP유001 Combined Baseline](../EXP유001_Combined_MLP_4Fold_CV_Baseline.md)의 Hand-made OOF Accuracy **81.2500%**에서 남은 오답 **480장**을 조사하고, 개선 방법을 선택하기 전 실제 오분류 원인을 탐색한다.

- **가설 A:** 숫자·문자 사이의 비슷한 획·연결 구조를 MLP가 어떻게 구별하는가? 특히 `0↔D`, `8↔B`의 판단 기준은 무엇인가?
- **가설 B:** 작성자별 입력의 절대 위치·크기·획 굵기·밝기 차이가 모델의 분류 결과를 바꾸는가?

## 실험 목록

| 구분 | 실험 | 현재 상태 | 목적 |
|---|---|---|---|
| 선행 | [EXP유001 — Combined 4-Fold CV Baseline](../EXP유001_Combined_MLP_4Fold_CV_Baseline.md) | **완료** | Normal vs Combined 학습 데이터 편입 효과 및 오류 확인 |
| 초기분석 1 | [EXP유002 — Error Analysis](EXP유002_Error_Analysis.md) | **계획 / 미실시** | OOF 오류·작성자·공간/명암·픽셀 판단 근거 진단 |
| 초기분석 2 | [EXP유003 — Input Sensitivity](EXP유003_Input_Sensitivity.md) | **계획 / 미실시** | 기존 모델 고정, 단일 입력 변형과 교정·신규 오류 비교 |

**현재까지의 실측 결과**는 EXP-CMB-001에 한정된다. 위의 EXP유002·003 문서는 향후 실험 설계이며, 새로운 분석·재학습 또는 개선 성능을 보고하는 로그가 아니다.

결과에 따라 [가설 A](../가설A/README.md)(획·픽셀 분류 근거) 또는 [가설 B](../가설B/README.md)(공간 분포·Size/Center·증강)로 후속 연구를 나눈다. 공식 Original Test와 교수님 최종 Test는 사용하지 않는다.
