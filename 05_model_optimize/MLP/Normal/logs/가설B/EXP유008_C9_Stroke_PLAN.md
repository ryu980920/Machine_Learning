# Normal MLP · 가설 B · EXP유008 — 가설 A의 C9에 Stroke 추가 (사전 실험 계획)

**실험 상태:** 사전등록 / 실행 전. EXP유007과 별도 실험으로 관리한다.

## 질문
가설 A EXP유004의 C9(Size 정규화 + CoM)를 정확히 재현한 후, StrokeWidthNormalizer를 추가하면 문자·숫자·공식 Validation 성능은 어떻게 달라지는가?

## 방법
- 조건 1: **C9 그대로** — 가설 A EXP유004 Stage2의 C9_BBoxLongside20 구현을 직접 재사용한 MLP.
- 조건 2: **C9 → Stroke 0.30 → CoM 재정렬**.
- 조건 3: **C9 → Stroke 0.60 → CoM 재정렬**.
- 같은 모델구조와 학습 예산을 위해 각 조건에서 **Seed 42·43·44**, 각 200 iterations, hidden 100 ReLU/Adam, tol=0.0, n_iter_no_change=201. 3조건 × 3 Seed = 총 9개 MLP.
- 공식 Train 76,800장만 학습/전처리 기준 산출, 공식 Dev Validation 19,200장과 이미 여러 번 분석한 Hand-made 2,560장 개발 평가.
- **공식 Original Test 16,000장·교수님 Test 미사용.** Hand-made를 학습하거나 기준값 결정에 사용하지 않는다.

## C9 원본 재현
- 원본 코드 출처: exp/normal-a-exp004-stroke-normalization-20261116 브랜치의 exp_a004_followup.py, exp_a004_official_mlp.py, width_normalization.py. 세 파일의 원본 코드를 수정하지 않고 실험 브랜치에 복제한다.
- C9: 전경 threshold=64, BBox 긴 변의 Official Train 중앙값 20px, scale = clip(20 / observed_bbox_longside, 0.75, 1.25); BBox 중심 기준 bilinear 등방성 OpenCV affine warp; 연결요소/Euler number가 바뀌면 최대 5회 강도 완화; 최종 CoM은 threshold=32, 목표 (13.5,13.5), zero-filled 정수 이동.
- 실제 C9 코드는 크기 목표를 공식 Train만으로 fit하고, 각 이미지를 변환 후 CoM으로 정렬한다.
- C9 기준 모델의 Seed별 역사적 정답 수를 원본 Stage2 결과 CSV와 비교하여 재현 여부를 별도 기록한다. 일치하지 않으면 그 차이와 실행환경을 명시하고 같은 실행 내 대조만 해석한다.

## Stroke 추가
- 앞선 Normal 가설 A에서 사용한 동일 stroke_normalization.py 사용(이미 main의 EXP유007에서 사용).
- Stroke 기준은 **C9 처리된 Official Train 76,800장으로만 새로 fit**.
- Skeleton + Distance transform으로 획 굵기 추정, threshold=64, Training q20/q80에서 벗어난 획을 dilation/erosion 후 alpha=0.30 또는 0.60으로 혼합.
- Topology guard와 CoM 최종 재정렬 사용.
- C9 단독의 전처리·학습 조건과 Stroke 추가 조건의 차이는 Stroke 처리 여부와 강도로 고정한다.

## 주요 평가
- Official Validation Accuracy/Macro F1 및 Hand-made 전체·숫자·문자·작성자별 Accuracy.
- 16개 클래스별 Recall과 B↔8, D↔0, 5↔F 방향별 혼동.
- 같은 이미지에서 C9 단독 대비 교정/신규 오답.
- 원본 C9 입력·출력 재현, C9+Stroke 변환 후 전체 형태 변화·잉크 질량 감소 기록.
- 세 Seed 평균과 각 Seed 변동; 공식 Val 97.30% 기준 충족 여부를 함께 본다.

**채택 기준:** Hand-made 문자와 숫자가 함께 개선되더라도 공식 Validation에 손실이 있으면 최종 Pipeline 교체는 하지 않는다. Hand-made는 반복 사용된 개발용 데이터이므로 신규 필체 일반화 증거로 해석하지 않는다.

## 산출물과 배포 원칙
- 실험 브랜치 exp/normal-b-exp008-c9-stroke에서만 코드·사전계획·결과 로그 작성.
- 워크플로 .github/workflows/normal-hypb-exp008-c9-stroke.yml의 Actions Artifact에 9개 학습모델 및 실제 CSV·그림·메트릭 저장(30일).
- GitHub main은 사용자가 추가 승인하기 전까지 건드리지 않는다.
