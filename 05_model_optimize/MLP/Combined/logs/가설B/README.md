# Combined — 가설 B: 공간 분포·중심 정렬·데이터 증강

## 검증 목적

작성자별 중심·크기·밝기·획 굵기의 차이가 Combined MLP의 잔존 오분류와 관련되는지 확인하고, 전처리·학습 증강의 개별 효과를 4-Fold OOF로 검증한다.

- 위치 ±1px와 Size 변화만으로 예측이 뒤집히는가?
- Size 정규화와 Center 정렬을 **각각** 적용하면 작성자 보류 OOF가 달라지는가?
- 동일한 전처리를 Training과 Validation에 적용하여 재학습할 때 효과가 바뀌는가?
- 학습용 Hand-made 비중 또는 Training-only Augmentation을 바꿀 경우 성능과 공식 Validation을 함께 유지하는가?

## 연구 흐름 (모두 계획)

1. [초기분석 EXP유002 — Error Analysis](../초기분석/EXP유002_Error_Analysis.md) — 오류 특징 비교 (**미실시**).
2. [초기분석 EXP유003 — Input Sensitivity](../초기분석/EXP유003_Input_Sensitivity.md) — 기존 모델 고정 단일 변수 실험 (**미실시**).
3. 추후 Size-only, Center-only, Size+Center 재학습 비교 — 동일 4-Fold·공식 Train 76,800장+Hand-made 1,920장, Baseline 원본 유지 (**미실시**).
4. 추후 Hand-made Training 비중·증강 및 Forward/Reverse Verification — 별도 통제 실험으로 구분 (**미실시**).

참고: [Normal 가설 B](../../../Normal/logs/가설B/README.md). CoM의 Normal 결과를 Combined의 개선 성능으로 간주하지 않는다.

## 평가 원칙

전체 Accuracy만 보지 않고 Macro F1, 숫자/문자, 작성자별 성능, `0↔D`·`8↔B`, 공식 개발 Validation, 학습/추론 시간, Iteration을 비교한다. 보류 작성자와 공식 Test는 학습·조정에 사용하지 않는다.
