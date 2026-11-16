# EXP유012 — 재현 방법

- **실제 공통 실행 코드:** [exp011_012_causal_probes.py](../scripts/exp011_012_causal_probes.py)
- **원래 학습된 9개 체크포인트:** [EXP유006 성공한 Actions](https://github.com/ryu980920/Machine_Learning/actions/runs/43785390739)
- **실제 성공한 크기 개입 분석:** https://github.com/ryu980920/Machine_Learning/actions/runs/43801194796

이 실험은 EXP유011과 **동일한 고정 모델·공통 코드**를 사용한다. 기존 9개 체크포인트와 Manifest를 받아 `--models-root`에 배열하고 Seed 42,43,44 각각 실행한다.

```bash
python 05_model_optimize/MLP/Normal/scripts/exp011_012_causal_probes.py \
  --seed 42 \
  --models-root /path/to/exp006-original-models \
  --out /tmp/exp011-012-seed42
```

EXP유012의 원본 결과는 `scale_class_response.csv`(이미지·Seed·모델별 확률), `scale_geometry.csv`(크기·위상 교란)로 생성된다. 총괄 집계는 [exp010_012_aggregate.py](../scripts/exp010_012_aggregate.py)를 사용하며 정확한 전체 실행 절차는 [Actions YAML](https://github.com/ryu980920/Machine_Learning/blob/main/.github/workflows/normal-exp010-012-mechanism.yml)에 있다.

**주의:** 스케일 1.0은 원본 Hand-made→CoM이며, EXP유006의 Size→CoM 평가 입력과 다르다. 새 학습/독립 Test 사용 없음.
