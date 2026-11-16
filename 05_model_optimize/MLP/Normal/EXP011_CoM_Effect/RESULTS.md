# EXP유011 — 실측 결과 요약

**원본:** [통합 실험일지](README.md) · [12개 교차조건 실측 CSV](summary.csv) · [성공한 Actions](https://github.com/ryu980920/Machine_Learning/actions/runs/43801194796)

고정된 기존 모델을 사용하여 모델 학습과 평가 입력 전처리를 분리했다.

| 학습 모델 | 평가 입력 | 숫자 | 문자 |
|---|---|---:|---:|
| Size | Size | 36.4583% | 94.2014% |
| Size | Size→CoM | 73.8125% | 69.7222% |
| Size→CoM | Size→CoM | 84.0625% | 91.7708% |

Size 모델 고정 시 입력만 CoM으로 이동하면 숫자 **+37.3542%p**, 문자는 **−24.4792%p** 변화한다. 동일 Size→CoM 입력에서 학습 모델을 바꾸면 숫자는 추가로 **+10.2500%p** 높아진다. 반대 순서의 분해는 다르므로 한 효과의 보편적인 독립 기여율로 해석하지 않는다.

**실행 검증:** EXP유006의 9개 체크포인트 해시 및 공식 Validation/Hand-made 정답 수가 일치했고 재학습은 수행하지 않았다.

## 전수 결과 및 검증 정보

원본 종합 Artifact에는 `exp011_accuracy_matrix.csv`, `exp011_paired_summary.csv`, `exp011_center_geometry_by_outcome.csv`, `all_cross_metrics.csv`, `all_paired_effects.csv`, `all_pair_errors.csv`, `all_size_center_outcomes.csv`가 있다. `summary.csv`는 **3개 학습조건×4개 평가조건의 공식 Validation 전체/Hand-made 숫자/문자 3-Seed 평균**만 추린 파일이다.
