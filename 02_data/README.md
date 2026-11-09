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
| [MNIST_EMNIST_official_split_112000.npz](datasets/MNIST_EMNIST_official_split_112000.npz) | Training **96,000장**, Test **16,000장** = **112,000장**. 숫자 0~9, 대문자 A~F (16클래스), 28×28 Grayscale |
| [ML_04_v3_verification_report.md](datasets/ML_04_v3_verification_report.md) | v3 중복 및 증강 수량 검증 자료 |

- 현재 NPZ는 **최종 v3**이며 Training 내부와 Training/Test 사이의 **완전 동일 픽셀 중복이 각각 0쌍**이다. 이는 모든 유사 이미지(near-duplicate)를 제거했다는 의미는 아니다.
- 최종 증강 **5,349장**: 회전 2,141장, 획 두껍게 1,604장, 가늘게 1,604장. D 클래스 내부 중복 1쌍은 **원본 1장을 +5° 회전본으로 대체**해 해결했으므로 전체 이미지 수는 증가하지 않았다.
- 최종 파일 SHA-256: `0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879`
- [W01 이선형 통합 실험 기록](../03_experiments/W01_2026-11-02_11-08/logs/EXP-이-001_중복검증_증강개선.md)에 이전 버전(v2)과 최종 수정(v3)을 구분해 기록했다.
- **최초 공식 분할 Notebook과 최종 v3는 동일한 버전이 아니다.** 이전 데이터셋 기록인 [EXP-곽-001](../03_experiments/W01_2026-11-02_11-08/logs/EXP-곽-001_MNIST_EMNIST_데이터셋_구성.md)의 수치와도 혼용하지 않는다.

## 2. Hand-made 데이터 4개 — Grayscale 표준 및 방향 보정 (2026-11-09)

팀원 원본 JPG(28×28)에서 **중간 밝기값을 보존하는 Grayscale NPZ**로 4개 파일을 다시 생성했다. 기존 0/1 Binary NPZ는 **현재 main 브랜치에서 대체**되었다. 기존 파일은 이전 Git 커밋 이력에 남으며, 원본 개인 ZIP은 저장소에 올리지 않는다.

| 파일 | 데이터 크기 | 픽셀 표현 | 처리 |
|---|---|---|---|
| [ml26_41.npz](Hand_made/ml26_41.npz) | 640 × 28 × 28 | 0~255, 256종 | 원본 JPG 밝기 보존 + 90° 반시계 회전 |
| [ml26_42.npz](Hand_made/ml26_42.npz) | 640 × 28 × 28 | 0~255, **4종** | **255 - X** 반전 + 90° 반시계 회전 |
| [ml26_43.npz](Hand_made/ml26_43.npz) | 640 × 28 × 28 | 0~255, 256종 | **원본 JPG에서 Grayscale 재생성** (과거 Binary NPZ 사용 안 함) |
| [ml26_44.npz](Hand_made/ml26_44.npz) | 640 × 28 × 28 | 0~255, 256종 | **255 - X** 반전 + 90° 반시계 회전 |

총 **2,560장**, 작성자당 **16개 클래스 × 각 40장**. 네 파일을 합치지 않고 개별 보관한다. 42번은 최초 JPG 자체가 사실상 이진 이미지라 반전해도 중간 밝기가 거의 없다.

### 공통 NPZ 양식 — 실험 코드 호환

| Key | Shape | dtype | 내용 |
|---|---|---|---|
| `img` | `(640, 28, 28)` | `uint8` | **Grayscale 0~255**, 검은 배경·밝은 글자 |
| `label` | `(640,)` | `<U1` | `'0'`~`'9'`, `'A'`~`'F'` |
| `label_id` | `(640,)` | `uint8` | 0~9, A~F는 10~15 |

- 저장 순서: `0_01 → 1_01 → … → F_01 → 0_02 → … → F_40`. 640장의 순서·정답 배열은 기존과 동일하다.
- **임계값 이진화, 이미지 확대/축소, 중심 정렬, 증강은 적용하지 않았다.** 원본 JPG 밝기값을 보존하면서 최초 Grayscale 통일 단계에서 **42·44번 흑백 반전**, 실험 3의 오류 발견 후 **41·42·44번을 반시계 방향 90° 회전**했다. 43번은 회전하지 않았다.
- **방향 보정 전 성능은 삭제하지 않았다.** [EXP-유-004 실험일지](../03_experiments/W02_2026-11-09_11-15/logs/EXP-유-004_Handmade_방향보정_성능평가.md)에 5개 모델의 초기 평가(19.88~23.12%)와 방향 보정 후 평가(53.24~62.93%)를 모두 기록했다.
- 네 파일 모두 원본 JPG 픽셀/레이블을 재확인하고 NPZ 재읽기 검증, 재생성 바이트 일치 검증을 마쳤다. 정확한 **SHA-256과 작성자별 중간 밝기 비율**은 [GRAYSCALE_VERIFICATION.md](Hand_made/GRAYSCALE_VERIFICATION.md)를 참고한다.
- **중요 — 실험 3:** 손글씨 `img`에는 이미 0~255 값이 들어 있다. **추가로 `*255`를 수행하지 않는다.** 실험 2에서 저장한 SVC·MLP·KNN의 `PixelScaler255` Pipeline은 그대로 사용한다.
- 이후 Combined Dataset 실험에서는 손글씨 중 **학습에 사용한 이미지와 평가용 이미지를 분리**한다. 시험용 보류 작성자 등 사용 분할을 기록한다.

## 3. 코드 및 재현성

- [02_official_split_dataset.ipynb](notebooks/02_official_split_dataset.ipynb): 공식 분할 **초기 데이터셋**을 생성하는 Notebook. 이를 실행하는 것만으로 v3가 재생성되는 것은 아니다.
- [rebuild_ml04_v3.py](notebooks/rebuild_ml04_v3.py): **공식 분할 초기 NPZ**를 검증된 v3로 업데이트. 원본 SHA와 결과 SHA가 맞는 경우에만 교체하도록 구성.
- [convert_hand_made.py](notebooks/convert_hand_made.py): 개인별 원본 JPG ZIP 또는 팀 제출 `ML-Project.zip`에서 **흑백 반전 및 90° 방향 보정을 포함한 0~255 Grayscale NPZ 4개를 재생성·검증**한다. 원본 ZIP은 저장소에 보관하지 않는다.

원본 ZIP이 있는 로컬 컴퓨터에서 변환 실행:

```bash
pip install numpy pillow
python 02_data/notebooks/convert_hand_made.py --source /path/to/ML-Project.zip --output-dir 02_data/Hand_made
# 개별 ZIP 4개가 있는 경우
python 02_data/notebooks/convert_hand_made.py --input-dir /path/to/zips --output-dir 02_data/Hand_made
```

## 4. 운영 원칙

- 결과 이미지 및 NPZ를 GitHub에서 비교할 때는 **Grayscale v3 (방향 보정, 2026-11-09)** 여부, 원본 이미지 수, 픽셀 범위(0~255), 정답 레이블, 파일 SHA-256을 확인한다. 이전 Binary v1은 과거 커밋에 보존되어 있다.
- **Hand-made는 학습에 사용할 수 있다.** Normal/Combined 비교에서는 Hand-made 중 학습용과 평가용을 **겹치지 않게 분리**하고, 비교하는 모델에 동일한 평가 조건을 사용한다. 교수님이 배포하는 평가 데이터는 **1차/2차 단계별 별도 지침**을 따른다.
- 데이터·코드 변경은 [03_experiments 주차별 기록](../03_experiments/README.md)에 근거를 남기고, 최종 NPZ 버전과 연결한다.
