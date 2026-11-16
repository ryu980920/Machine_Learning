# EXP유011 — 재현 방법

- **실제 공통 실행 코드:** [exp011_012_causal_probes.py](../scripts/exp011_012_causal_probes.py)
- **원래 학습한 9개 MLP:** [EXP유006 성공한 Actions](https://github.com/ryu980920/Machine_Learning/actions/runs/43785390739)
- **EXP유011/012 성공한 Actions:** https://github.com/ryu980920/Machine_Learning/actions/runs/43801194796

기존 EXP유006의 `normal-exp006-com-42` · `normal-exp006-size-42` · `normal-exp006-size_com-42` 등 **3개 모델×3개 Seed의 Artifact**를 다운로드한다. 원본 모델·Manifest를 변경하지 않고 `--models-root`의 각 하위 폴더에 배치한다.

```bash
python 05_model_optimize/MLP/Normal/scripts/exp011_012_causal_probes.py \
  --seed 42 \
  --models-root /path/to/exp006-original-models \
  --out /tmp/exp011-012-seed42
```

43,44도 같은 방법으로 각각 실행한다. 해당 스크립트는 **EXP유011과 EXP유012를 함께 계산하는 원본 공통 실행 코드**다. 두 결과는 각각 `cross_metrics.csv` / `paired_effects.csv` 및 `scale_class_response.csv` 등으로 분리 저장된다.

기존 EXP유006 체크포인트 SHA256 및 대각선 예측 정답 수가 일치하지 않으면 실행을 실패 처리한다. 재학습·공식 Test 접근은 하지 않는다.
