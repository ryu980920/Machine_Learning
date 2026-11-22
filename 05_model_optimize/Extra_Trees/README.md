# Extra Trees — 최적화 기록

- [Normal 최적화](Normal/README.md) — 공식 MNIST·EMNIST Training 기반
- [Combined 최적화](Combined/README.md) — 공식 Training + 별도 Hand-made 학습용 데이터 기반
- **Learning Curve 예시:** 학습 데이터 크기에 따른 Training / Validation 점수 (실제로 산출한 조건과 x축을 명시)
- **기존 학습 전 기본 코드:** [extra_trees.py](../../03_experiments/model_optimization/extra_trees.py)
- **공통 로그 양식:** [EXPERIMENT_TEMPLATE.md](../EXPERIMENT_TEMPLATE.md)

아직 이 폴더에서 최적화 실험을 완료한 것으로 표시하지 않는다.

## Extra Trees 최적화 실측 결과

- [Normal 실험일지](Normal/logs/findings.md) · [Normal PDF](Normal/results/Extra_Trees_Normal.pdf)
- [Combined 실험일지](Combined/logs/findings.md) · [Combined PDF](Combined/results/Extra_Trees_Combined.pdf)

### 재실행

저장소 루트에서 실행한다. 기준 데이터와 팀 공통 MLP 전처리 코드를 사용한다. Python 3.12를 권장한다. 아래 경로는 새 결과를 저장하는 별도 디렉터리다. 같은 모드·환경에서 중단된 실행을 재개할 때 같은 출력 경로를 사용하면 완료한 조건을 재사용한다.

```bash
extra=05_model_optimize/Extra_Trees
python -m pip install -r "$extra/Normal/scripts/requirements.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python "$extra/Normal/scripts/run_extra_trees.py" --mode normal --out /tmp/et-normal --jobs 4
python "$extra/Combined/scripts/run_combined.py" --out /tmp/et-combined --jobs 4
```

학습 전에 입력 데이터만 검증하려면 `--check-only`를 추가한다. 결과 CSV·그래프·PDF는 기존 결과 폴더를 입력으로 다시 생성할 수 있다. 아래 예시는 Normal이며 Combined도 같은 방법으로 경로와 모드를 바꿔 실행한다. PDF 렌더링에는 시스템의 Pango와 Noto Sans CJK KR 글꼴이 필요하다.

```bash
source_dir="$extra/Normal/results"
python "$extra/Normal/scripts/aggregate.py" --mode normal --source "$source_dir" --out /tmp/et-normal-figs
python "$extra/Normal/scripts/render_report.py" --mode Normal --source "$source_dir" \
  --results /tmp/et-normal-figs --output /tmp/Extra_Trees_Normal.pdf
```

실험별 `summary.json`에는 설정과 모든 평가 지표가 있고, `predictions.npz`에는 각 공식 Validation Fold 및 작성자별 예측이 있다. 공식 Validation 예측 순서는 `official_split_indices.npz`의 val_idx를 따른다. 작성자 예측 순서는 원본 NPZ 행 순서다. 모델 가중치는 저장하지 않았으며 선택한 설정과 학습 코드를 제공한다.
