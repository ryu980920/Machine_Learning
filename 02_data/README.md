# 02_data — 데이터셋 및 구현 코드

이 폴더는 **최종 학습·평가 데이터 / 구현 코드 / 개인 Hand-made 테스트 데이터**의 세 하위 폴더로 관리한다.

| 폴더 | 내용 | 사용 목적 |
|---|---|---|
| [datasets/](datasets/) | 최종 MNIST·EMNIST v3 **112,000장 NPZ**, 검증 보고서 | 공식 분할 기반 모델 개발 데이터 |
| [notebooks/](notebooks/) | 데이터 생성 Notebook, v3 재구성 코드, Hand-made 통일 변환 코드 | 구성 과정 재현 |
| [Hand_made/](Hand_made/) | 팀원별 독립 손글씨 NPZ **4개, 각 640장** | **테스트 전용** |

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

## 2. Hand-made 테스트 데이터 4개

| 파일 | 데이터 크기 | 클래스 | 참고 |
|---|---|---|---|
| [ml26_41.npz](Hand_made/ml26_41.npz) | 640 × 28 × 28 | 16클래스 × 40장 | JPG와 기존 별도 형식 NPZ에서 43 형식으로 통일 |
| [ml26_42.npz](Hand_made/ml26_42.npz) | 640 × 28 × 28 | 16클래스 × 40장 | JPG에서 생성, **흑백 극성 반전** |
| [ml26_43.npz](Hand_made/ml26_43.npz) | 640 × 28 × 28 | 16클래스 × 40장 | 사용자가 제공한 **원본 NPZ 바이트 그대로 보존** |
| [ml26_44.npz](Hand_made/ml26_44.npz) | 640 × 28 × 28 | 16클래스 × 40장 | JPG와 기존 별도 형식 NPZ에서 43 형식으로 통일, **흑백 극성 반전** |

총 **2,560장**이며, 네 데이터셋을 합치지 않고 개별 NPZ로 보관한다. 모두 **학습·모델 선택·튜닝에는 사용하지 않는 독립 테스트 전용**이다.

### 공통 NPZ 양식 — `ml26_43.npz` 기준

| Key | Shape | dtype | 내용 |
|---|---|---|---|
| `img` | `(640, 28, 28)` | `uint8` | 0/1 이진 이미지, **0=검은 배경, 1=흰 글자** |
| `label` | `(640,)` | `<U1` | `'0'`~`'9'`, `'A'`~`'F'` |
| `label_id` | `(640,)` | `uint8` | 숫자 0~9, 문자 A~F는 10~15 |

- 원본 JPG는 모두 **28×28**이다. 클래스별 40장씩, 저장 순서는 `0_01 → 1_01 → … → F_01 → 0_02 → … → F_40`이다.
- JPG 밝기값 **127 초과 → 1**로 이진화하였다. 41·43은 검은 배경에 밝은 글자이며 42·44는 흰 배경에 어두운 글자라서, 42·44만 반전하여 **네 파일 모두 검은 배경=0, 밝은 글자=1**로 통일했다.
- 43의 원본 NPZ는 전체 640장의 JPG 이진화 결과와 픽셀별로 일치함을 확인한 뒤 변경하지 않았다.
- 4개 NPZ 모두 `img`, `label`, `label_id` **세 Key만 포함**하며, Shape·dtype·레이블·모든 이미지의 0/1 픽셀을 검증했다.
- **주의:** 최종 MNIST·EMNIST NPZ의 픽셀 범위 및 데이터 스케일과 Hand-made NPZ의 0/1 값은 다르다. 모델에 넣을 때는 **Training에서 정한 일관된 추론 입력 변환 규칙**을 적용해야 한다.

## 3. 코드 및 재현성

- [02_official_split_dataset.ipynb](notebooks/02_official_split_dataset.ipynb): 공식 분할 **초기 데이터셋**을 생성하는 Notebook. 이를 실행하는 것만으로 v3가 재생성되는 것은 아니다.
- [rebuild_ml04_v3.py](notebooks/rebuild_ml04_v3.py): **공식 분할 초기 NPZ**를 검증된 v3로 업데이트. 원본 SHA와 결과 SHA가 맞는 경우에만 교체하도록 구성.
- [convert_hand_made.py](notebooks/convert_hand_made.py): 개인별 원본 ZIP `ml26_41.zip`~`ml26_44.zip`에서 43과 같은 형식으로 4개 NPZ를 생성·검증. 원본 ZIP은 저장소에 보관하지 않는다.

원본 ZIP이 있는 로컬 컴퓨터에서 변환 실행:

```bash
pip install numpy pillow
python 02_data/notebooks/convert_hand_made.py --input-dir /path/to/zips --output-dir 02_data/Hand_made
```

## 4. 운영 원칙

- 결과 이미지 및 NPZ를 GitHub에서 비교할 때는 버전, 이미지 개수, 이미지 값(0/1과 0~255), 정답 레이블을 확인한다.
- 테스트 전용 Hand-made 및 교수 배포 평가 데이터는 **학습, 데이터 증강용 원본, 준지도 의사 레이블 학습, 튜닝 데이터로 사용하지 않는다.**
- 데이터·코드 변경은 [03_experiments 주차별 기록](../03_experiments/README.md)에 근거를 남기고, 최종 NPZ 버전과 연결한다.
