# MLPClassifier / Combined — 재현 코드

| 파일 | 용도 |
|---|---|
| [combined_mlp_writer_cv.py](combined_mlp_writer_cv.py) | SHA-256 확인, 공식 Group Train/Validation 분할, Normal 재학습 + Combined 작성자 4-Fold CV, 예측·Loss·모델 저장 |
| [pixel_transform.py](pixel_transform.py) | 기존 W02와 동일한 `PixelScaler255`: `X/255.0` |
| [test_reproducibility.py](test_reproducibility.py) | 모델/그룹분할/CSV/그래프 함수에 대한 합성 입력 점검 |
| [render_cm_viridis.py](render_cm_viridis.py) | 게시된 `OOF_predictions.csv`에서 **Viridis CM 4개 PNG만** 재생성 (모델 재학습 없음) |
| [EXP_CMB_001_requirements.txt](EXP_CMB_001_requirements.txt) | 실험 Python 패키지 버전 고정 |

## 실험 결과의 CM만 재생성

저장소 루트에서 실행:

```bash
python -m pip install -r 05_model_optimize/MLP/Combined/scripts/EXP_CMB_001_requirements.txt
python 05_model_optimize/MLP/Combined/scripts/render_cm_viridis.py
```

16클래스 순서는 `0123456789ABCDEF`, 색상 범위는 Count `0–160`, 행 정규화 `0–100%`. 0 셀 수치만 표시 생략하고 원본 예측은 바꾸지 않는다.

## 전체 학습 다시 실행

실험 원본은 [GitHub Actions Run #43622998640](https://github.com/ryu980920/Machine_Learning/actions/runs/43622998640)이다. [GitHub Actions workflow](../../../../.github/workflows/exp-cmb-001-mlp-combined.yml)는 재실행이 필요할 때만 수동으로 시작하도록 유지한다.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python 05_model_optimize/MLP/Combined/scripts/combined_mlp_writer_cv.py \
  --official 02_data/datasets/MNIST_EMNIST_official_split_112000.npz \
  --handmade-dir 02_data/Hand_made \
  --output ./EXP_CMB_001_new_run
```

새 학습은 매번 새 출력 디렉터리를 사용한다. 공식 Original Test 및 교수님 Test는 읽지 않는다. **추가 실험의 결과와 기존 EXP-CMB-001 결과는 혼용하지 않는다.**

[실험로그](../logs/EXP유001_Combined_MLP_4Fold_CV_Baseline.md) · [원본 결과](../results/EXP_CMB_001/README.md).
