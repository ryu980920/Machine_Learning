# STEP 0 — Baseline 복원 및 검증

## 배경 및 목적
최종 Validation 1,280장의 기존 SVC/KNN 모델을 그대로 복원. 새로운 후보의 점수를 기존 저장값과 정확히 대조하는 기준을 확보한다.

## 입력과 검증
- 학습 원본은 기존 `final-validation-development-sources.zip`, 평가 원본은 `Final_Validation_Private_Evaluation_Kit_20261126.zip`. 기존 평가 ZIP `Final_10_Model_Validation_Results.zip`에서 원본 예측/클래스 지표/CM을 추출.
- SVC 학습 82,080, KNN 학습 113,920. 원본 체크포인트 SHA256 SVC `801e0b242fdd2ae3fd6a3e22e8816fd0f138b61be310513fca7a241a9ed48311`, KNN `ff046fffafc27cf19bb0a3e6a997b55c2af42b9494c556d6851c3ea06a5df6d9`. 직접 파일 해시 대조 완료.
- Evaluation NPZ 해시 `73de49ac0a0e05eba5099a0a8c31b332ac8abec3fd7eebbc69eb875210770f31`. 16 classes × 80 images.
- scikit-learn 1.8.0, NumPy 2.3.5, Python 3.13.5; 원본 CoM→Deskew→Pixel/HOG 20:80.
- 공식 Original Test, 교수님 Test 미사용. 신규 1,280장은 이번 실험에서는 개발용으로 전환하여 후보 평가에만 사용하고 학습, 증강, PCA fitting에 사용하지 않음.

## 실측 재현
|모델|기존 정답|재현 정답|개별 예측 불일치|
|---|---:|---:|---:|
|SVC Combined|1199|1199|0|
|KNN Combined|1182|1182|0|

기존 공식 개발 정답은 SVC 18967/19200, KNN 18856/19200 (fit manifest); SVC vectorized 공식 cross-check도 18967. 기존 학습/추론 시간은 기존 환경 측정값으로, 이번 실험의 탐색 시간과 동일한 측정치로 취급하지 않는다.

## 판정
PASS — 원본 1,280개 개별 예측 완전 일치. 기존 모델과 평가 원본을 수정하지 않음.
