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

최적화 담당자는 **문제 → 원인/가설 → 시도 → 결과/근거 → 다음 작업**을 [실험일지 양식](../EXPERIMENT_TEMPLATE.md)에 기록하고, 공통 Validation을 임의로 바꾸지 않는다.
