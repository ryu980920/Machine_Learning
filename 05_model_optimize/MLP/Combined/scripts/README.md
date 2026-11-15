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

## 후속 Combined 실험의 GitHub Actions 재사용 절차

기존 성공 경로는 **실험 브랜치 → Draft PR → GitHub Actions 수동 실행 → Artifact 검증 → 승인 후 main 병합**이다. 대용량 NPZ 파일은 GitHub 저장소에 둔 채 Actions runner가 체크아웃하여 읽으므로 외부 Python 환경에 파일을 직접 옮기지 않는다.

1. 최신 `main`에서 `exp/mlp-combined-<실험ID>` 실험 브랜치를 생성한다. 기존 `EXP_CMB_001`의 결과 파일을 새 실험으로 덮어쓰지 않는다.
2. 현재 [실행 워크플로](../../../../.github/workflows/exp-cmb-001-mlp-combined.yml)를 기반으로 실행할 스크립트·요구 패키지·출력 경로·Artifact 이름을 **새 실험 ID에 맞춰 명시적으로 변경**한다. 기존 워크플로를 수정하지 않고 실행하면 EXP-CMB-001만 재실행된다.
3. 수정한 코드와 워크플로를 실험 브랜치에 커밋하고 **Draft PR**을 만들어 변경 파일과 실행 기록을 연결한다. 초안 PR이 열렸다는 것만으로 학습이 자동 시작되지는 않는다.
4. GitHub **Actions → EXP-CMB-001 Combined MLP writer-group CV → Run workflow**에서 실행 브랜치를 고르고 수동 실행한다. 이 워크플로는 `main`에 등록된 뒤 Actions 목록에서 이용 가능하다.
5. Actions에서 **합성 데이터 점검 → 입력 SHA-256/분할 사전점검 → 4-Fold 학습·평가 → Artifact 업로드**의 단계별 성공 여부를 확인한다. 오류가 발생하면 로그를 확인하고 원인 수정 후 재실행한다.
6. 완료된 Run의 `EXP-CMB-001-cv-outputs` Artifact를 내려받아 `metrics_summary.csv`, OOF 예측, 학습 Loss, 재현 manifest, CM, 학습 모델이 해당 실험에서 생성됐는지 확인한다. 후속 실험은 수정한 Artifact 이름과 경로를 따른다.
7. 검증된 결과만 `logs/`, `results/<새 실험 ID>/`에 등록한다. 공식 Original Test 및 교수님 Test를 학습/튜닝에 사용하지 않으며, Writer OOF와 독립 최종 Test를 구분한다.
8. **사용자 검토·승인 후** PR을 Ready for review로 전환하고 `main`에 병합한다. 기존 `main`과 변경이 겹치면 충돌부터 해결한다.

**시간 절약 원칙:** 학습은 Actions에서 한 번 실행하고 Artifact를 기준으로 CSV·이미지를 정리한다. CM 색상 변경·결과표 정리는 저장된 OOF 예측으로 다시 생성하고 학습을 재실행하지 않는다. GitHub Actions Run ID, 브랜치, 커밋 SHA 및 Artifact ID를 새 실험일지에 기록한다.

## 후속 분석·가설별 코드 배치

[초기분석 로그](../logs/초기분석/README.md)의 **EXP유002 Error Analysis**와 **EXP유003 Input Sensitivity**는 현재 계획 단계다. 실행 코드는 새 실험 ID로 생성하며, 완료된 [EXP-CMB-001 실행 코드](combined_mlp_writer_cv.py)를 덮어쓰지 않는다. [가설 A](../logs/가설A/README.md)·[가설 B](../logs/가설B/README.md)도 동일한 `scripts/`에서 독립 코드로 관리한다.

각 코드에는 모델/입력 SHA 확인, 작성자 보류 Fold 검증, 공식 Original Test 및 교수님 Test 미사용, 출력 경로와 실행 버전을 기록한다. 변경이 필요 없는 분석은 저장된 OOF·모델에서 수행해 Baseline 학습을 반복하지 않는다.


## EXP-CMB-002 — 오류 원인 검증 (기존 모델 고정)

- [exp_cmb002_evidence.py](exp_cmb002_evidence.py): EXP-CMB-001 OOF/모델/이미지 해시 및 2,560장 예측 재검증, 7개 공간 특징, IG, 실제 이미지 Grid와 기존 오류 전환표.
- [exp_cmb002_followup.py](exp_cmb002_followup.py): 형태/공간 질량 11개 지표, 작성자별 대조, 밝기·Bounding Box 보존 픽셀 교환 및 오류쌍 임계값 안정성.
- [EXP유002 정식 실험일지](../logs/초기분석/EXP유002_Error_Analysis.md) · [실측 원본](../results/Error_Analysis/README.md).
- [성공 실행 #43743082289](https://github.com/ryu980920/Machine_Learning/actions/runs/43743082289). 기존 **Normal 1개 + Combined 4개 저장 가중치만 사용**, 신규 MLP 학습 없음.

이번 분석의 작성자·클래스별 통계는 연구용이며, 실제 추론에서는 작성자 ID·정답을 이용해 전처리를 변경하지 않는다. 원본 EXP-CMB-001 실행·결과는 그대로 유지한다.

## EXP-CMB-003 — 모델 고정 입력 민감도 연구

- [exp_cmb003_input_sensitivity.py](exp_cmb003_input_sensitivity.py): Normal 1개 + Combined 4개 원본 체크포인트/OOF 해시 확인, 10개 입력 변형의 Accuracy·Macro F1·교정/신규 오답·Logit Margin·공식 개발 Validation 통제 비교. **새 학습 없음**.
- [EXP유003 완성 실험일지](../logs/초기분석/EXP유003_Input_Sensitivity.md) · [실측 CSV/이미지](../results/Input_Sensitivity/README.md) · [최종 성공 Actions #43745629093](https://github.com/ryu980920/Machine_Learning/actions/runs/43745629093).
- 기존 EXP-CMB-001 Baseline 훈련·모델 및 EXP-CMB-002 분석은 변경하지 않는다. 입력 단일 변형은 연구용 진단이며, 공식 Original Test·교수님 Test는 사용하지 않는다.
