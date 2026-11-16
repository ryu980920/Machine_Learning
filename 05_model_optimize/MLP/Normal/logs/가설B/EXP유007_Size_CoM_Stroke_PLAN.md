# Normal MLP · 가설 B · EXP유007 — Size→CoM 이후 Stroke 추가 효과 (사전등록)

> EXP유006 실제 측정 결과를 확인한 후 시작하는 조건부 후속 실험이다.
> 실행 결과를 확인하기 전에는 완료나 성능 향상이라고 표시하지 않는다.

## 선행 결과 및 이유
- EXP유006에서 CoM-only 3-seed 평균 Hand-made 전체 77.695%, 문자 66.493%, 공식 Val 97.052%.
- Size→CoM은 전체 86.953%, 문자 91.771%, 공식 Val 96.061%.
- 따라서 Size의 문자 개선 효과는 확인됐으나 **공식 Validation 약 0.99%p 하락**이 남았다.
- 이번 실험은 Size→CoM을 고정한 상태에서 Stroke를 추가하면 오분류가 어떻게 변하는지 확인한다.

## 정확한 조건 (동일 학습 세팅과 seed)
- A: Size→CoM 대조군 — 다시 학습한다.
- B: Size→CoM→Stroke(강도 0.30)→CoM.
- C: Size→CoM→Stroke(강도 0.60)→CoM.
- 각 조건 Seed 42·43·44, 총 9개 신규 MLP.
- MLPClassifier(random_state=seed), 나머지 기본 설정, 28×28 /255.0.
- Official Train 76,800장만 학습, Official Validation 19,200장 및
  기존 41~44 Hand-made 2,560장은 개발 평가 전용.
- Original Test 16,000장과 교수님 Test는 절대 사용하지 않는다.

## 전처리 기준
- Size: Combined EXP-CMB-004의 BBox 긴 변 20px 등방성 bilinear affine scaling,
  threshold 32, 0-filled padding.
- CoM: threshold 32, 목표 (13.5,13.5), 반올림한 정수 이동.
- Stroke: 기존 가설 A EXP유004 브랜치의 검증된 StrokeWidthNormalizer 소스 그대로 복사.
  threshold 64, Skeleton + Distance Transform으로 두께 측정,
  공식 Training에 있는 획 굵기의 20%·80% 분위수 기준 dilation/erosion,
  원본과 변형을 alpha=0.30 또는 0.60으로 혼합.
- Stroke 정규화 기준은 **Size→CoM 변환된 Official Training만으로 fit**.
  Validation과 Hand-made 레이블 및 이미지 분포는 사용하지 않는다.
- Stroke 이후 CoM을 다시 적용해 획 변경에 따른 중심 좌표 재이동을 통제한다.
- 연결요소/Euler 수가 변하면 Stroke 변형 강도를 완화하는 기존 topology guard 유지.

## 평가
- Official Validation, Hand-made 전체·숫자·문자·4명 작성자 Accuracy/Macro F1.
- B↔8, D↔0, 5↔F 양방향 혼동, 같은 이미지별 오답 교정과 신규 오답.
- 3 seeds에서 개선 방향 일관성, 공식 Val 손실 악화/회복 여부.
- 공식 Validation 97.30% 목표 미달 시 최종 모델 후보 채택은 보류할 수 있다.
- 기존 4명은 반복 분석한 개발 데이터이므로 외부 작성자 일반화 주장은 금지한다.

## 저장·배포
- 코드: 05_model_optimize/MLP/Normal/scripts/exp007_size_com_stroke_followup.py
- 참조 구현: 05_model_optimize/MLP/Normal/scripts/stroke_normalization.py
- 워크플로: .github/workflows/normal-hypb-exp007-size-com-stroke.yml
- Artifact에만 결과 저장, GitHub main 변경/자동 결과 commit 없음.
