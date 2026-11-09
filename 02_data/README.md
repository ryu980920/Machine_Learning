# 02_data — 데이터셋 및 구현 코드

이 폴더는 **최종 학습·평가 데이터 / 구현 코드 / 개인 Hand-made 데이터**의 세 하위 폴더로 관리한다.

| 폴더 | 내용 | 사용 목적 |
|---|---|---|
| [datasets/](datasets/) | 최종 MNIST·EMNIST v3 **112,000장 NPZ**, 검증 보고서 | 공식 분할 기반 모델 개발 데이터 |
| [notebooks/](notebooks/) | 데이터 생성 Notebook, v3 재구성 코드, Hand-made 통일 변환 코드 | 구성 과정 재현 |
| [Hand_made/](Hand_made/) | 팀원별 독립 손글씨 NPZ **4개, 각 640장** | **Normal/Combined 학습 비교 및 독립 평가용으로 분할** |

## 1. 최종 MNIST·EMNIST 데이터

| 파일 | 수량·특징 |
|---|---|
| [MNIST_EMNIST_official_split_112000.npz](datasets/MNIST_EMNIST_official_split_112000.npz) | Training **96,000장**, Test **16,000장** = **112,000장**. 숫자 0–9, 대문자 A–F (16클래스), 28×28 Grayscale |
| [ML_04_v3_verification_report.md](datasets/ML_04_v3_verification_report.md) | v3 중복 및 증강 수량 검증 자료 |

- 현재 NPZ는 **최종 v3**이며 Training 내부와 Training/Test 사이의 **완전 동일 픽셀 중복이 각각 0쌍**이다. 이는 모든 유사 이미지(near-duplicate)를 제거했다는 의미는 아니다.
- 최종 증강 **5,349장**: 회전 2,141장, 획 두껍게 1,604장, 가늘게 1,604장. D 클래스 내부 중복 1쌍은 **원본 1장을 +5° 회전본으로 대체**해 해결했으므로 전체 이미지 수는 증가하지 않았다.
- 최종 파일 SHA-256: `0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879`
- [W01 이선형 통합 실험 기록](../03_experiments/W01_2026-11-02_11-08/logs/EXP이001_중복검증_증강개선.md)에 이전 버전(v2)과 최종 수정(v3)을 구분해 기록했다.
- **최초 공식 분할 Notebook과 최종 v3는 동일한 버전이 아니다.** 이전 데이터셋 기록인 [EXP곽001](../03_experiments/W01_2026-11-02_11-08/logs/EXP곽001_MNIST_EMNIST_데이터셋_구성.md)의 수치와도 혼용하지 않는다.

## 2. Hand-made 데이터 4개 — 현재 표준 (42번 새 원본 교체, 2026-11-09)

41·43·44번은 원본 JPG(28×28)에서, 새 42번은 재제출 ZIP 내부 원본 NPZ의 `X`에서 **중간 밝기값을 보존하는 표준 Grayscale NPZ**를 구성했다. 기존 0/1 Binary NPZ는 **현재 main 브랜치에서 대체**되었다. 기존 파일은 이전 Git 커밋 이력에 남으며, 원본 개인 ZIP은 저장소에 올리지 않는다.

| 파일 | 데이터 크기 | 픽셀 표현 | 처리 |
|---|---|---|---|
| [ml26_41.npz](Hand_made/ml26_41.npz) | 640 × 28 × 28 | 0–255, 256종 | 원본 JPG 밝기 보존 + 90° 반시계 회전 |
| [ml26_42.npz](Hand_made/ml26_42.npz) | 640 × 28 × 28 | 0–255, **256종** | **재제출 원본 X를 그대로 사용. 흑백 반전·90° 회전 없음** |
| [ml26_43.npz](Hand_made/ml26_43.npz) | 640 × 28 × 28 | 0–255, 256종 | **원본 JPG에서 Grayscale 재생성** (과거 Binary NPZ 사용 안 함) |
| [ml26_44.npz](Hand_made/ml26_44.npz) | 640 × 28 × 28 | 0–255, 256종 | **255 - X** 반전 + 90° 반시계 회전 |

총 **2,560장**, 작성자당 **16개 클래스 × 각 40장**. 네 파일을 합치지 않고 개별 보관한다. **기존 42번은 고유 밝기값이 4종인 사실상 Binary 데이터였으나, 새로 제출된 `ml26_42(2).zip`의 Grayscale 원본으로 교체했다.** 새 42번은 256종의 밝기값을 가지고 90° 회전 및 흑백 반전을 적용하지 않았다. 이 교체 이전에 기록된 실험 3 성능은 **구 42번 데이터** 기준으로, 새 42번에 대한 성능으로 해석하면 안 된다.

### 공통 NPZ 양식 — 실험 코드 호환

| Key | Shape | dtype | 내용 |
|---|---|---|---|
| `img` | `(640, 28, 28)` | `uint8` | **Grayscale 0–255**, 검은 배경·밝은 글자 |
| `label` | `(640,)` | `<U1` | `'0'`–`'9'`, `'A'`–`'F'` |
| `label_id` | `(640,)` | `uint8` | 0–9, A–F는 10–15 |

- 저장 순서: `0_01 → 1_01 → … → F_01 → 0_02 → … → F_40`. 640장의 순서·정답 배열은 기존과 동일하다.
- **현재 표준의 전처리:** 41번 반시계 90° 회전, 43번 변경 없음, 44번 흑백 반전(`255-X`) 후 반시계 90° 회전. **새 42번은 제출 ZIP 내부 NPZ의 `X` 픽셀을 그대로 사용하며 반전·회전·임계값 이진화·Blur·중심 정렬·증강을 적용하지 않았다.** 제출 JPG와 NPZ `X` 사이의 JPEG 재인코딩에 따른 차이는 최대 2 밝기 단위이므로, 42번에서는 정보 손실을 피하기 위해 `X`를 기준으로 사용한다. 과거 42번 반전·회전 이력은 실험 3 로그에 남겨 둔다.
- **과거 실험 3 성능은 삭제하지 않았다.** [EXP유003 실험일지](../03_experiments/W02_2026-11-09_11-15/logs/EXP유003_Handmade_방향보정_성능평가.md)에 구 42번 사용 당시의 최초 평가(19.88–23.12%)와 방향 보정 후 평가(53.24–62.93%)를 보존했다. **새 42번 예측을 포함한 비교 결과는 EXP유002 8절에 추가했으며, 4명 모두에 대한 완전 신규 추론 재실행과는 구별한다.**
- 네 파일 모두 클래스별 40장·레이블·NPZ 키 및 크기를 확인했다. 새 42번은 제출 ZIP 내부 `X`와 현재 표준 NPZ `img`의 바이트 배열이 동일하며 원본 JPG와의 차이는 최대 2 밝기 단위이다. 최신 **SHA-256과 중간 밝기 비율**은 [GRAYSCALE_VERIFICATION.md](Hand_made/GRAYSCALE_VERIFICATION.md)를 참고한다.
- **새 원본 기준 실험 결과:** [EXP유002 8절](../03_experiments/W02_2026-11-09_11-15/logs/EXP유002_5개모델_Normal_Baseline.md)에 5개 모델 Rotation 전·후, [EXP유004](../03_experiments/W02_2026-11-09_11-15/logs/EXP유004_42번_Grayscale_원본재제출_비교.md)에 구/신 42번 원본 비교를 기록했다.
- **중요 — 실험 3:** 손글씨 `img`에는 이미 0–255 값이 들어 있다. **추가로 `*255`를 수행하지 않는다.** 실험 2에서 저장한 SVC·MLP·KNN의 `PixelScaler255` Pipeline은 그대로 사용한다.
- 이후 Combined Dataset 실험에서는 손글씨 중 **학습에 사용한 이미지와 평가용 이미지를 분리**한다. 시험용 보류 작성자 등 사용 분할을 기록한다.

## 3. 코드 및 재현성

- [02_official_split_dataset.ipynb](scripts/02_official_split_dataset.ipynb): 공식 분할 **초기 데이터셋**을 생성하는 Notebook. 이를 실행하는 것만으로 v3가 재생성되는 것은 아니다.
- [rebuild_ml04_v3.py](scripts/rebuild_ml04_v3.py): **공식 분할 초기 NPZ**를 검증된 v3로 업데이트. 원본 SHA와 결과 SHA가 맞는 경우에만 교체하도록 구성.
- [convert_hand_made.py](scripts/convert_hand_made.py): 기존 `ML-Project.zip`에서 41·43·44번을 재구성하고, **추가로 필요한 새 42번 ZIP**에서 `X`를 직접 가져와 네 표준 NPZ를 재생성·검증한다. 42번에 반전·회전을 적용하지 않으며, 새 ZIP이 없으면 구 42번 데이터로 실수로 덮어쓰지 않도록 실행을 중단한다. 원본 ZIP은 저장소에 보관하지 않는다.

원본 ZIP이 있는 로컬 컴퓨터에서 변환 실행:

```bash
pip install numpy pillow
python 02_data/scripts/convert_hand_made.py --source /path/to/ML-Project.zip \
  --writer42-replacement '/path/to/ml26_42(2).zip' --output-dir 02_data/Hand_made
# 기존 개별 41·43·44번 ZIP을 사용하는 경우
python 02_data/scripts/convert_hand_made.py --input-dir /path/to/zips \
  --writer42-replacement '/path/to/ml26_42(2).zip' --output-dir 02_data/Hand_made
```

## 4. 운영 원칙

- 결과 이미지 및 NPZ를 GitHub에서 비교할 때는 **Grayscale v4 (42번 원본 교체, 2026-11-09)** 여부, 원본 이미지 수, 픽셀 범위(0–255), 정답 레이블, 파일 SHA-256을 확인한다. 이전 Binary v1은 과거 커밋에 보존되어 있다.
- **Hand-made는 학습에 사용할 수 있다.** Normal/Combined 비교에서는 Hand-made 중 학습용과 평가용을 **겹치지 않게 분리**하고, 비교하는 모델에 동일한 평가 조건을 사용한다. 교수님이 배포하는 평가 데이터는 **1차/2차 단계별 별도 지침**을 따른다.
- 데이터·코드 변경은 [03_experiments 주차별 기록](../03_experiments/README.md)에 근거를 남기고, 최종 NPZ 버전과 연결한다.

## 5. 최신 데이터 활용 및 신규 수집 계획 (2026-11-10)

새롭게 논의한 데이터 구분은 [HANDMADE_SPLIT_PLAN.md](HANDMADE_SPLIT_PLAN.md)에 기록한다.

- 현재 기존 팀원 4명 × 640장 = **Hand-made 2,560장**을 모두 탐색적 평가에 사용한 상태. 향후 **Combined 학습 추가 후보**지만, 실제 효과는 Normal/Combined 비교 실험으로 검증하고 채택 여부를 결정한다.
- 신규 **Hand-made Validation 640장(8명×80장)** 및 신규 **Hand-made Test 640장(다른 8명×80장)**은 수집 권장안이며 **수집 전·규모 미확정**이다. 기존 학습 작성자와도 겹치지 않게 한다.
- 개발 단계에는 신규 Validation을 반복 사용하고, 최종 모델·전처리 설정 고정 후 신규 Hand-made Test와 공식 Test 16,000장을 별도로 평가한다.
- 위쪽 문서의 "새 42번을 이용한 실험 3 재평가 미수행"은 **4명 전체에 대한 모델의 완전 신규 추론 재실행이 없었다**는 구분이다. 새 42번을 실제 추론한 결과와 기존 41·43·44번 예측을 결합한 분석은 [EXP유002 8절](../03_experiments/W02_2026-11-09_11-15/logs/EXP유002_5개모델_Normal_Baseline.md)에 이미 기록돼 있다.
