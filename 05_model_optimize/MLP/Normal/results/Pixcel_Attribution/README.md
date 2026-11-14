# EXP유001 — Pixcel_Attribution 실제 산출물 (Normal MLP 1·2단계)

이 폴더의 파일은 **Normal MLP Baseline의 분류 결과 및 입력 픽셀 기여도 분석**을 GitHub Actions에서 재실행한 실제 산출물이다. 3단계 입력 변형, 전처리 도입, 모델 최적화는 포함하지 않는다.

- [실험 로그](../../logs/EXP유001_Pixcel_Attribution.md)
- [Python 재현 코드](../../scripts/Pixcel_Attribution.py)
- [성공한 GitHub Actions 실행 기록](https://github.com/ryu980920/Machine_Learning/actions/runs/43628471307)

## 실행 검증

| 항목 | 재실행 결과 |
|---|---:|
| 학습 | 공식 Training 내부 Train **76,800장**, 기존 MLP 동일 설정 |
| 공식 Validation | **19,200장**, Accuracy **97.3177083%** |
| 기존 공식 CM와 차이 | **L1=0** |
| Hand-made 개발용 | **2,560장**, Accuracy **62.8515625%** |
| 기존 Hand-made 개별 예측과 차이 | **0 / 2,560** |
| 실제 추출·IG 계산 | 공식 **42장** + Hand-made **56장** = **98장** |
| Test 데이터 사용 | **없음** |

## 파일 목록

| 파일 | 내용 |
|---|---|
| [selected_cases.csv](selected_cases.csv) | 98개 분석 사례의 정확한 출처(작성자·인덱스), 정답·예측, 확률, IG 합·잔차 |
| [handmade_all_predictions.csv](handmade_all_predictions.csv) | 2,560개 Hand-made 전체 재추론 결과 |
| [selected_pixels_and_IG.npz](selected_pixels_and_IG.npz) | 98개 이미지의 실제 28×28 배열과 IG 행렬 (selected_cases.csv와 같은 순서) |
| [run_metadata.json](run_metadata.json) | 데이터 SHA, 모델 설정, 버전, 재현 검증, completeness residual |
| [SUMMARY.md](SUMMARY.md) | Actions가 생성한 실행 요약 |
| [review_grids/](review_grids/) | 총 17개 이미지: 원본 손글씨·Signed IG·Absolute IG 비교 |
| [MLP_Pixcel_Attribution_handmade_examples.png](MLP_Pixcel_Attribution_handmade_examples.png) | 주요 Hand-made 오분류 3건(5→F, B→8, 9→4) |
| [MLP_Pixcel_Attribution_6_examples.png](MLP_Pixcel_Attribution_6_examples.png) | 공식 Validation 3건 + Hand-made 3건 비교 |

## 해석 주의

- IG는 **검은 이미지** 기준 **128-step midpoint** 근사다. 오분류는 `예측 logit − 실제 logit`의 기여도를, 정상 분류는 정답 logit의 기여도를 계산했다.
- Signed IG에서 빨강은 목표값 증가, 파랑은 감소 방향이다. **검은 배경 픽셀의 IG 기여도가 0인 것**은 기준점 선택과 관계있으므로, 획의 기여도 집중만으로 모델의 *형태 학습*을 증명하지 않는다.
- 이 표본은 오류 유형을 중심으로 선정했다. **전체 손글씨 오류의 통계적 대표 표본이 아니다.**
- 입력 위치·굵기·명암·크기를 변화시켜 실제 예측이 얼마나 달라지는지는 **다음 실험에서 별도 평가**한다.
