# EXP-CMB-001 — Combined MLP Baseline 및 Writer-Group 4-Fold CV

**상태: 실험 실행을 위한 코드와 자동 실행 작업을 별도 브랜치에서 준비. 결과는 실측 확인 전까지 미확정.**

## 1. 원본 데이터와 실험 목표

- 저장소: `ryu980920/Machine_Learning`, 공식 최종 v3와 현행 새 42번 NPZ 사용
- `02_data/datasets/MNIST_EMNIST_official_split_112000.npz`: 공식 96,000 Training 중 그룹 기준 76,800 Train / 19,200 개발 Validation
- `02_data/Hand_made/ml26_41.npz` ~ `ml26_44.npz`: 4명 각 640장, 16개 클래스별 40장
- 공식 Original Test 16,000장 / 교수님 1·2차 Test는 절대 읽지 않음
- 모델: GitHub `03_experiments/model_optimization/mlp.py`와 동일한 **`MLPClassifier(random_state=42)`**; 픽셀 스케일 `X/255.0`. 초기 Baseline에서 다른 하이퍼파라미터/전처리/증강 변경 금지.

## 2. CV 계획

| Fold (보류 작성자) | Combined에 더하는 작성자 | Combined Train | 손글씨 Validation |
|---|---|---:|---:|
| 41 | 42, 43, 44 | 76,800 + 1,920 | 640 |
| 42 | 41, 43, 44 | 76,800 + 1,920 | 640 |
| 43 | 41, 42, 44 | 76,800 + 1,920 | 640 |
| 44 | 41, 42, 43 | 76,800 + 1,920 | 640 |

Normal MLP는 공식 Training 76,800장만으로 한 번 학습하고 동일한 보류 작성자 640장에 적용하는 대조군이다. 각 Combined Fold는 독립적으로 다시 학습한다. 전체 OOF 2,560장 결과는 **개발용 4개 모델의 예측을 합친 지표**이며 하나의 최종 독립 Test 모델의 점수가 아니다.

## 3. 실행 위치 / 방법 (GitHub 저장소 최상위 폴더에서)

원본 파일은 **GitHub 레포에 이미 있으므로 재업로드할 필요 없음**. 이 패키지의 `combined_mlp_writer_cv.py`, `pixel_transform.py`, `EXP_CMB_001_requirements.txt`, `test_reproducibility.py`를 다음 경로로 옮겨 놓는다:

`05_model_optimize/MLP/Combined/scripts/`

```bash
python -m pip install -r 05_model_optimize/MLP/Combined/scripts/EXP_CMB_001_requirements.txt
python 05_model_optimize/MLP/Combined/scripts/test_reproducibility.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python 05_model_optimize/MLP/Combined/scripts/combined_mlp_writer_cv.py \
  --official 02_data/datasets/MNIST_EMNIST_official_split_112000.npz \
  --handmade-dir 02_data/Hand_made \
  --output exp_cmb_001_preflight --check-only
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python 05_model_optimize/MLP/Combined/scripts/combined_mlp_writer_cv.py \
  --official 02_data/datasets/MNIST_EMNIST_official_split_112000.npz \
  --handmade-dir 02_data/Hand_made \
  --output exp_cmb_001_results
```

동일한 경로에 재실행 시 기존 결과를 덮어쓰지 않도록 별도의 새 `--output` 경로를 지정한다. **이 명령은 로컬 저장소 또는 GitHub Codespaces에서 실행해야 원본 데이터에 접근 가능하다.** GitHub Actions에서 실행하려면 `.github/workflows/combined-mlp-cv.yml`을 저장소에 별도 승인 후 등록하고 Actions 탭에서 해당 워크플로를 수동 실행한다. GitHub 워크플로는 저장소를 변경하지 않고 결과를 Artifacts로 업로드한다.

## 4. 자동 저장되는 재현 자료

| 산출물 | 검증 목적 |
|---|---|
| `input_hashes.json` | 공식 v3와 네 작성자 원본 NPZ SHA-256 확인 |
| `official_split_indices.npz` | 공식 96,000장에 대한 원본 행 번호 train_idx(76,800)/val_idx(19,200) 및 각 소스 그룹 ID |
| `training_validation_manifest.json` | MLP 기본 하이퍼파라미터, 4개 Fold의 학습 작성자/보류 작성자, 클래스별 수량, 분할 알고리즘, 누출 검사 |
| `writer_fold_assignment.csv` | 작성자별 학습/검증 배치와 이미지 개수 |
| `trained_models/*.joblib` | Normal 1개, Combined 4개 모델 재현용 체크포인트. **신뢰한 파일만 로드** |
| `OOF_predictions.csv` | 2,560장 작성자·작성자내 원본 인덱스·정답·Normal/Combined 예측 및 정오 판정 |
| `OOF_comparison.csv`, `fold_comparison.csv` | 전체/작성자별 Accuracy, Macro F1 및 시간 비교 |
| `OOF_errors.csv`, `combined_errors_only.csv` | 모든 실측 오분류 사례의 원본 인덱스 |
| `normal_OOF_confusion_matrix.csv/png`, `combined_OOF_confusion_matrix.csv/png` | 16×16 OOF Confusion Matrix |
| `normal_OOF_per_class.csv`, `combined_OOF_per_class.csv` | 16개 클래스별 Precision·Recall·F1 |
| `loss_curves.csv`, `MLP_loss_curves.png` | Normal과 Combined 4-Fold 각각 실제 MLP Loss Curve |
| `fit_times_and_iterations.csv`, `run_summary.json` | 학습시간, 반복 횟수, 최종 손실, 실행환경, 데이터 버전 및 실제 점수 |
| `fold_*_normal_confusion_matrix.csv`, `fold_*_combined_confusion_matrix.csv` | 보류 작성자별 모델 오분류 행렬 |

## 5. 현 실행환경의 검증 범위

- GitHub 원본의 `mlp.py`, `common.py`, 실험 2 `run_experiment2.py`를 읽고 초기 모델·스케일·공식 분할 알고리즘을 맞춤.
- `test_reproducibility.py`는 합성 데이터에서 모델 기본 설정, 스케일러, 그룹 분할 비중복, 계산/CSV/PNG 생성 동작을 테스트한다.
- 실제 원본 GitHub NPZ를 이 채팅의 Python 컨테이너로 이전하지 못했으므로 **4-Fold 실제 학습 결과·성능 수치는 아직 없음**. 합성 데이터 테스트 결과를 실측값으로 사용하면 안 된다.
- 파일은 실험용 코드만 승인된 별도 브랜치에 커밋. 실험 결과는 GitHub 레포에 커밋하지 않고 사용자에게 먼저 보고한다.

## 저장 정책 (2026-11-15 승인)

- 최종 결과 공유 위치(승인 후): `05_model_optimize/MLP/Combined/logs/` 및 `results/`.
- 이번에는 실행 코드만 실험 브랜치에 반영한다. 학습 결과 CSV·PNG·모델 체크포인트는 어떤 브랜치에도 자동 커밋하지 않는다.
- GitHub Actions에서 생성한 임시 Artifacts는 결과 보고와 검증 자료 인출에만 사용한다. 사용자에게 결과를 보고하고 별도 승인받기 전에는 레포 결과 경로에 배포하지 않는다.
