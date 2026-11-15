# EXP유003 — Combined Input Sensitivity (실험 계획)

> **실험 ID:** EXP-CMB-003 · **상태: 계획 수립 / 미실시**. 이번 문서는 기존 모델을 이용한 진단 실험 계획이다. 새로운 예측 결과, 성능 변화 또는 최적화 결과는 아직 없다.

## 1. 연구 질문

Hand-made Training을 추가한 Combined MLP도 Normal처럼 위치·크기·획 굵기·밝기 변화에 민감한가? Combined 학습은 입력 변화에 따른 오답 교정과 신규 오류를 어떻게 바꾸는가?

선행 자료: [EXP-CMB-001](../EXP유001_Combined_MLP_4Fold_CV_Baseline.md), [EXP유002 — Error Analysis](EXP유002_Error_Analysis.md), [Normal Input Sensitivity](../../../Normal/logs/초기분석/EXP유002_Input_Sensitivity.md).

## 2. 통제 조건

- `EXP_CMB_001/trained_models.zip`에서 **Normal 1개와 Combined 보류 작성자별 4개**를 로드한다. **새 학습 0회**, 모든 가중치 고정.
- 41번 이미지는 41번을 보류한 Combined 모델로만 평가한다. 나머지도 같은 원칙을 적용한다.
- 공식 개발 Validation과 Hand-made의 입력/인덱스/레이블, 모델 구조, Fold, 클래스 순서를 고정하고, **입력 변수 하나씩만** 변경한다.
- 기존 OOF 2,560개 예측 및 원본 CM·Accuracy가 먼저 재현되어야 한다. 공식 Original Test·교수님 최종 Test를 사용하지 않는다.

## 3. 입력 변형 후보

| 항목 | 비교 조건(원본 대비) | 고정할 조건 |
|---|---|---|
| 위치 | 좌·우·상·하 각각 1px 이동 | 밝기, 굵기, 크기 조작 없음 |
| 크기 | 중심 기준 95%, 105% | 위치/밝기/굵기 조작 없음 |
| 굵기 | 얇게·굵게 단일 수준 (정확한 수치 실행 전 지정) | 위치/크기/밝기 조작 없음 |
| 밝기 | 원본 대비 90%, 110% | 위치/크기/굵기 조작 없음 |

기존 Normal 실험의 변환 구현을 확인하되, 픽셀 경계 0 채움·보간·클리핑·획 손실을 명시한다. 처음부터 복수 전처리를 동시에 적용하지 않는다.

## 4. 평가 및 판단 계획

- 전체 OOF Accuracy, Macro Precision/Recall/F1 및 숫자·문자/작성자·클래스별 변화
- Prediction Flip Rate, 오답→정답, 정답→오답, 오답→다른 오답 분리
- `0↔D`, `8↔B` 오류쌍과 고정 경쟁 클래스 Logit Margin 변화
- **동일한 변형을 EXP-CMB-001의 Normal 대조군에도 적용**해 Combined가 민감도를 줄였는지 비교
- 공식 개발 Validation, 전처리 및 추론 시간과 이미지 클리핑 비율 기록

**이 단계는 원인 진단이다.** 특정 변형의 정확도가 높더라도 이를 그대로 배포용 전처리로 채택하지 않는다. 다음 단계의 Size/Center 재학습은 [가설 B](../가설B/README.md)에서 별도 통제 실험으로 다룬다.

## 5. 산출물 계획 (미생성)

[Input_Sensitivity 결과 폴더 안내](../../results/Input_Sensitivity/README.md) — 실제 CSV·그림·모델 변경은 아직 없다. 실행 후 조건별 지표, paired prediction/transition CSV, 작성자·클래스별 표, Viridis CM, 변화 그래프, 메타데이터와 실행 시간 기록을 추가한다.
