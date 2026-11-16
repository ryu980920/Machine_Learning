# EXP유010 — 재현 방법

- **실제 실행 코드:** [exp010_geometry_distribution.py](../scripts/exp010_geometry_distribution.py)
- **전처리 원본:** [exp006_size_com_comparison.py](../scripts/exp006_size_com_comparison.py) / Combined의 `exp_cmb004_geometry_normalization.py`
- **성공한 GitHub Actions 워크플로:** [normal-exp010-012-mechanism.yml](https://github.com/ryu980920/Machine_Learning/blob/main/.github/workflows/normal-exp010-012-mechanism.yml)
- **원본 성공 Run:** https://github.com/ryu980920/Machine_Learning/actions/runs/43801194796

저장소 루트에서 Python 3.13과 `05_model_optimize/MLP/Normal/scripts/EXP007_requirements.txt`의 패키지를 설치한 뒤 실행한다.

```bash
python 05_model_optimize/MLP/Normal/scripts/exp010_geometry_distribution.py --out /tmp/exp010-geometry
```

**전제:** 저장소 기존 Official Train/Val과 Hand-made 데이터 경로를 유지한다. 공식 Original Test/교수님 Test를 사용하지 않는다. **모델 학습이 없는 기하학적 계측 실험**이다.

모든 실제 원본 CSV는 위 Actions의 `exp010-geometry`와 종합 Artifact에 있다(보존 기간 정책 적용).
