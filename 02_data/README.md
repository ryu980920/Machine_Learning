# 02_data — 데이터셋 및 구현 코드

이 폴더는 **최종 학습·평가 데이터 / 구현 코드 / 개인 Hand-made 데이터**의 세 하위 폴더로 관리한다.

| 폴더 | 내용 | 사용 목적 |
|---|---|---|
| [datasets/](datasets/) | 최종 MNIST·EMNIST v3 **112,000장 NPZ**, 검증 보고서 | 공식 분할 기반 모델 개발 데이터 |
| [scripts/](scripts/) | 데이터 생성 Notebook, v3 재구성 코드, Hand-made 통일 변환 코드 | 구성 과정 재현 |
| [Hand_made/](Hand_made/) | 팀원별 독립 손글씨 NPZ **4개, 각 640장** | **개발용 오류 분석·Normal/Combined 실험, 학습·검증 중복 방지** |

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

## 4. 교수님 강조 — 데이터 품질·전처리·증강의 역할

**전원 참여:** 유용성·김다연·이선형·곽지호 모두 공식 Original과 Hand-made 개발 데이터의 오분류 및 분포 차이를 분석하고, 각자 **중심·크기·기울기·획 굵기·명암·극성·노이즈·잘림·레이블 등**에서 검증할 가설을 정한다. 개선 과제를 특정 한 사람에게만 맡기지 않는다. 개인별 집중 분야는 [병렬 작업계획](../01_project/TEAM_PARALLEL_WORKPLAN.md) 참조.

1. **입력 표준화(과거 실험):** 41·43·44번의 방향·극성을 맞춰 표준 Grayscale NPZ를 만든 것은 **기존 데이터 정비 결과**로 보존한다. 새로운 외부 Test마다 파일을 사람이 수동 회전·수정하라는 의미가 아니다.
2. **자동 전처리 Pipeline(향후 구현):** 학습/추론 모두 같은 정책으로 중심 정렬·기울기·크기·명암 등을 자동 보정하는 방법을 비교. 특정 작성자 ID나 Test 정답을 조건으로 다르게 처리하지 않는다.
3. **학습용 데이터 증강:** Training에만 회전·이동·스케일·획 굵기 변형 적용, 원본·증강본이 학습과 Validation에 갈라지지 않도록 기록. 무조건 수량을 늘리는 대신 *지금 부족한 필체 특징*을 채우는지를 확인한다.
4. **Cleaning vs Pipeline:** 잘못된 레이블·깨진 이미지 등 개발 데이터의 Cleaning과 모든 입력에 동작하는 자동 Pipeline을 구분한다. 전처리 `fit`은 Training에 한정한다.

## 5. 최신 데이터 분할 및 Test 운영

- **공식 구성:** MNIST·EMNIST **112,000장 = 공식 Training 96,000장 + 공식 Original Test 16,000장**. 교수님 설명의 MNIST 숫자 60,000/10,000 및 EMNIST A–F 각 6,000/1,000과 일치한다. 현재 Baseline은 공식 Training 96,000장을 **76,800 학습·19,200 Validation**으로 내부 분할했으며 기존 수치 재현을 위해 고정한다.
- **기존 Hand-made 2,560장:** 교수님 설명에 따라 개발 과정에서 **학습·Validation·오류 분석에 사용 가능**. 다만 학습에 사용한 이미지나 작성자를 같은 실험의 독립 Validation/Test라고 부르지 않는다.
- **신규 Hand-made Validation 640장:** 8명×80장 수집 **권장안·미수집·미확정**. **신규 독립 Hand-made Test 640장은 선택 사항**이며 교수님 외부 Test를 대체하는 필수가 아니다.
- **교수님 1차·2차 배포 데이터:** 둘 다 **Test 전용**. 최종 모델·Pipeline을 고정한 후 1차에서 전체 성능만 확인. 충분하면 종료, 미흡하면 1차 이미지/특정 오류를 분석하지 않고 기존 개발 자료로만 다시 최적화 → 2차 Test 후 종료.
- 교수님 Test 원본 이미지는 임의로 수동 수정하지 않는다. **이미 개발 단계에서 확정한 자동 전처리 Pipeline은 Test 추론 과정에 적용 가능**. 배포 지침이 허용한 최소 Cleaning 외에는 데이터 자체를 수정하지 않는다.
- **Normal vs Combined**는 같은 모델 설정·전처리·평가 데이터에서 학습용 Hand-made 추가 여부만 달리해 측정한다. 성능 개선 결과를 보고 실제 추가 채택을 결정한다.

자세한 분할·누출 방지 조건: [HANDMADE_SPLIT_PLAN.md](HANDMADE_SPLIT_PLAN.md) · [연구계획](../01_project/RESEARCH_PLAN.md).
