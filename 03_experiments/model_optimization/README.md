# 팀원용 학습 전 모델 5개 — 기본 설정 재현 및 직접 최적화

기존 [실험 2 코드](../W02_2026-11-09_11-15/scripts/experiment2_baseline/run_experiment2.py)에서 확인한 **학습 전 기본 설정**을 공유한다. 학습 완료된 `.joblib`을 제공하지 않으며, 실행할 때 모델을 새로 생성·학습한다.

## 사용법

저장소 루트에서 다음을 실행한다. 원본 실험 2는 Python 3.13.5 / scikit-learn 1.8.0 / numpy 2.3.5 환경에서 수행했다. 가능하면 Python·패키지 버전과 CPU 스레드 수를 일치시킨다.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python 03_experiments/model_optimization/run.py --model svc --variant baseline

# 5개 전부 원래 기본 설정으로 재학습
python 03_experiments/model_optimization/run.py --model all --variant baseline

# svc.py 의 build_candidate()만 수정하고 개선 시도
python 03_experiments/model_optimization/run.py --model svc --variant candidate
```

모델 선택: `svc`, `hgb`, `mlp`, `extra_trees`, `knn`, `all`. 데이터 기본 경로: `02_data/datasets/MNIST_EMNIST_official_split_112000.npz`. 별도 경로면 `--data 경로`를 사용한다.

- `common.py`: 원본과 동일한 데이터 SHA-256 및 **StratifiedGroupKFold(5, shuffle=True, random_state=42) 첫 Fold**, Train 76,800장·Validation 19,200장을 검사한다. 원본+증강본이 양쪽에 갈라지지 않도록 그룹을 유지한다.
- `build_baseline()`: 초기 기본 설정을 **수정하지 않는다**. `build_candidate()`: 모델·스케일링 설정을 독립적으로 수정하는 곳. 초기 상태에서는 둘이 같다.
- SVC·MLP·KNN은 기존 `PixelScaler255`로 `/255` 처리. HGB·Extra Trees는 원본 0–255.
- 결과 저장: `outputs/{baseline|candidate}/{모델}/metrics.json`, `confusion_matrix.csv`, `per_class.csv`. 파일을 자동 커밋하지 않고 실제 실험일지에 연결한다.
- **공식 Test 16,000장, Hand-made 2,560장에 대해서는 이 스크립트를 실행해도 학습·평가하지 않는다.** 후속 Combined 실험은 별도 설계 대상이다.

## 기존 실험 2 결과 (신규 실행값 아님)

| 모델 | Accuracy | Macro F1 |
|---|---:|---:|
| SVC | 97.6667% | 0.976551 |
| HGB | 97.4635% | 0.974542 |
| MLP | 97.3177% | 0.973032 |
| Extra Trees | 96.9688% | 0.969541 |
| KNN | 96.2917% | 0.962791 |

실측 근거: [EXP-유-003 CSV](../W02_2026-11-09_11-15/results/EXP-유-003_5models_comparison.csv).

**검증 상태:** 이 코드는 기존 분할·기본 모델 정의·스케일링을 옮겨 구현했다. 5개 모델을 이 새 코드로 실제로 재학습하여 위 지표와 동일한지 확인하는 절차는 **별도 실행 전까지 미검증**이다. 결과가 다르면 `metrics.json`, 패키지 버전, 입력 해시, 모델 설정을 확인한다.

모든 팀원은 **오류·데이터 특성 분석 → Pipeline/증강 가설 → 같은 Validation 비교 → 결과·근거 → 다음 작업**을 포함해 [실험일지 양식](../EXPERIMENT_TEMPLATE.md)에 기록하고, 공통 Validation을 임의로 바꾸지 않는다.

## 교수님 강조에 따른 후속 확장 사항 (2026-11-10, 아직 미구현)

**이 폴더의 현행 `common.py`·`run.py`는 Original 데이터의 Baseline/Candidate 훈련·Validation 전용이다.** 실제 Hand-made Combined 로더, 이미지 중심/크기/기울기/명암 자동 Pipeline, Training Augmentation 기능은 **아직 구현되지 않았고, 다음 연구 단계의 개발 대상**이다. `build_candidate()` 함수만 바꾼다고 손글씨가 자동으로 Combined 학습되거나 이미지가 정렬되지는 않는다.

- **4명 모두** 개발용 Hand-made 오류·원본/손글씨 이미지 특징을 분석하고 Pipeline/증강 실험을 직접 수행한다. 팀원별 구체적인 모델·업무 배정은 팀장이 따로 정한다.
- **Normal vs Combined 통제 실험:** 기본 모델·전처리·Validation 인덱스를 고정해 손글씨 학습 추가 효과만 먼저 분리 평가한다.
- **데이터 최적화:** 추론 시 고정 자동 Pipeline은 모델 일부; 회전·이동·크기·획 굵기 Augmentation은 **Training만**; 세부 방법이 정확도를 올리는지 개발용 Validation에서 실측한다.
- **모델 최적화:** 동일 Validation으로 후보별 Accuracy·Macro F1·모델 학습시간·Pipeline 처리시간·추론시간 비교. Epoch가 없는 모델에 가짜 Epoch Curve를 만들지 않는다. Notebook에 실제 실행·출력·학습곡선을 담아 보고한다.
- **교수님 Test:** 1·2차 모두 독립 평가용. 학습·오류 분석·맞춤 전처리 설계에 절대 사용하지 않는다. 전체 성능 확인만 하며 1차 충분 시 종료, 미흡하면 개발용 데이터로만 개선 후 2차를 평가한다.
- **결과 파일 주의:** 현재 Runner에서 같은 `--model`, `--variant`, `--output` 조합을 반복 실행하면 `metrics.json` 등이 **덮어써질 수 있다**. 반복 실험은 `--output`에 각자 고유 경로 또는 `EXP유001` 등 실행 식별자를 지정해 파일을 보존한다. 실행 환경·수치의 완전 재현은 아직 검증되지 않았다.

**추가 손글씨·CV 후속 구현:** 교수님은 새 Hand-made를 학습용으로 추가하는 방법을 허용했다. 기존 2,560장만으로 Combined 학습·검증을 병행하려면 실제 작성자 기준 Group CV가 적절할 수 있다. **현재 `run.py`에는 Combined 학습이나 작성자 Group CV 기능이 구현되어 있지 않다.** 신규 작성자 Validation을 따로 확보하면 기존 2,560장 내부 CV는 필수가 아니다.
