# Normal MLP · 가설 B · EXP유006 — BBox Size / CoM Center 비교 (사전등록)

> 상태: 실행 코드 등록 / 결과 검증 전. **실측 결과로 해석하지 않는다.**

## 연구 질문
Combined EXP-CMB-004에서 Size와 CoM이 서로 다른 숫자/문자 오류 방향에 작용했다.
Normal MLP에서도 Size-only와 Size→CoM을 독립적으로 학습하면,
CoM-only에서 관찰한 문자 인식 하락과 B↔8, D↔0 혼동을 완화하는가?

## 고정 비교
- A: CoM-only (신규 동일 조건 학습 대조)
- B: BBox Size-only (신규)
- C: BBox Size → CoM (신규)
- 기존 Normal 원본 62.8516%와 기존 CoM 77.1094%는 역사적 참조치이며
  새로운 3-seed 비교에서는 A/B/C를 같은 학습 예산으로 재학습해 비교한다.
- **Stroke 정규화는 이 실험에 포함하지 않는다.**
  Size-only 또는 Size→CoM의 추가 효과가 실제 확인된 뒤 별도 후속 실험에서 검토한다.

## 데이터
- 공식 v3 NPZ의 원래 Training 96,000장 내부에서 기존 StratifiedGroupKFold
  첫 Fold: Official Train 76,800 / Official Dev Val 19,200.
- Hand-made 현재 표준 작성자 41·42·43·44 각 640장, 합계 2,560장.
- 손글씨는 MLP 학습·Size 기준 선정에 사용하지 않으며, 반복 탐색된 개발용 평가 데이터다.
- **공식 Original Test 16,000장 및 교수님 Test 미사용**.

## 정확한 전처리 구현
- Combined의 exp_cmb004_geometry_normalization.py에서 image_geometry,
  apply_size 함수를 직접 사용해 구현 차이를 제거한다.
- 크기 기준: Training 이미지의 BBox 긴 변 중앙값 **20px** (실행 시 재검증).
- 전경 threshold=32/255, BBox 중심을 고정한 등방성 affine resize,
  SciPy bilinear order=1, zero padding, 원본 강도 0..255.
- CoM: Normal의 검증된 center_batch(X, "mass") 재사용.
  threshold=32, 무게중심 목표 (13.5,13.5), 정수 픽셀 이동, zero fill.
- 변환 순서 C: **Size 먼저, CoM 나중**.
- 각 조건/seed에서 Training과 Validation/Hand-made 입력에 완전히 동일한 변환 적용.

## 모델·실행
- Pipeline(PixelScaler255(), MLPClassifier(random_state=seed)),
  기타 MLP 파라미터는 sklearn 기본 설정. Seeds **42,43,44**.
- 총 **3개 조건 × 3 seeds = 9개 신규 MLP** 학습.
- seed 42 CoM은 기존 EXP유004의 공식 Val 정답 18,608장 및
  Hand-made 정답 1,974장을 재현하는지 검증.
- GitHub Actions 결과는 오직 Artifact로 수집하며 자동 commit 또는 main 변경하지 않는다.

## 평가·판정
1. 공식 Validation과 Hand-made 전체 Accuracy/Macro F1, 숫자·문자,
   작성자 41~44, 16개 클래스별 Recall을 분리한다.
2. B↔8, D↔0, 5↔F 오분류를 양방향으로 보고하고,
   A 대비 B/C의 같은 이미지 오답→정답 및 정답→오답을 전량 대조한다.
3. 크기 정규화가 긴 변 분산을 낮췄는지, CoM이 중심 오차를 줄였는지,
   보간으로 인해 잉크량과 가장자리 손실이 발생했는지 함께 본다.
4. Size의 추가 효과는 C 대 A의 **문자 오분류 개선·전체 성능·공식 Val 손실**,
   seed 반복성과 작성자별 반례를 함께 고려해 판정한다.
5. Size가 문자 개선에 유의미한 단서를 주더라도 공식 Val 손실이 크면
   최종 Pipeline 채택은 보류하고 후속 Stroke 결합의 연구 후보로만 사용한다.
6. 모든 수치는 기존 4명 개발 데이터 기반. 신규 작성자 독립 검증 전에는
   일반화 개선으로 주장하지 않는다.

## 실행 위치
- 코드: 05_model_optimize/MLP/Normal/scripts/exp006_size_com_comparison.py
- Workflow: .github/workflows/normal-hypb-exp006-size-com.yml
- 리뷰 Artifact: Normal-HypB-EXP006-complete-review 및 9개 개별 모델 Artifact.
- 본 실험 브랜치: exp/normal-b-exp006-size-com.
