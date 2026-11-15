# MLP / Combined — 초기분석

## 검증 목적

[EXP유001 Combined Baseline](../EXP유001_Combined_MLP_4Fold_CV_Baseline.md) Hand-made OOF Accuracy **81.2500%**에서 남은 480개 오답을 분석하여 **가설 A(유사 숫자/문자 픽셀·획 구조)**, **가설 B(공간·크기·중심·명암)**의 연구 우선순위를 근거로 결정한다.

## 실험 목록

| 구분 | 실험 | 상태 | 판단 |
|---|---|---|---|
| 선행 | [EXP유001 — Combined 4-Fold CV Baseline](../EXP유001_Combined_MLP_4Fold_CV_Baseline.md) | **완료** | Normal 62.5781% → Combined 81.2500% |
| 초기분석 1 | [EXP유002 — Error Analysis](EXP유002_Error_Analysis.md) | **실험 완료** | **가설 B 우선 검증**; 가설 A 독립 획 구조 실험은 현재 근거 부족하여 보류 |
| 초기분석 2 | [EXP유003 — Input Sensitivity](EXP유003_Input_Sensitivity.md) | **계획 / 미실시** | 기존 모델 가중치 고정 상태에서 가설 B 검증 예정 |

EXP유002는 기존 OOF 예측 재현, 원본 이미지·구조 특징, 작성자/실제 클래스 통제, 모델 픽셀 판단 및 강도/Bounding Box 보존 픽셀 교환 대조까지 완료했다. **실험 목적은 A/B의 후속 연구 가치 결정이었으며, 새 학습이나 개선된 Accuracy를 주장하지 않는다.**

후속 진단에서는 라벨을 모르는 상태에서 적용 가능한 동일 입력 변형을 단일 변수로 검증한다. 실제 수정·재학습이 필요한 시점까지 기존 [EXP-CMB-001 결과](../../results/EXP_CMB_001/README.md)는 변경하지 않는다.

[가설 B](../가설B/README.md) · [가설 A](../가설A/README.md).
