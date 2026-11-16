# EXP유012 — 실측 결과 요약

**원본:** [통합 실험일지](README.md) · [15개 그룹·배율 조합 실측 CSV](summary.csv) · [성공한 Actions](https://github.com/ryu980920/Machine_Learning/actions/runs/43801194796)

기존 CoM 모델을 고정하고 원본 Hand-made의 크기를 바꾼 뒤 CoM을 재정렬한 결과:

| 배율 | 숫자 Accuracy | 문자 Accuracy |
|---|---:|---:|
| 0.8× | 75.3125% | 25.0000% |
| 1.0× | 84.4167% | 66.4931% |
| 1.2× | 19.0625% | 95.3472% |

크기가 커질수록 문자 쪽 점수가 높아지는 강한 경향을 보였다. 반면 Size→CoM 모델에서는 같은 배율에 반대 방향의 반응이 나타난다. **단, 이미지 확대/축소는 획 굵기와 보간 픽셀·질량도 함께 바꾸므로 순수한 BBox 크기 하나의 인과 효과로 단정하지 않는다.**

6개 핵심 클래스 960장 중 5개 배율 모두 topology 지표가 유지된 **714장**에 대해서도 반응 방향을 확인했다. 동일 이미지를 여러 Seed에서 반복 평가한 횟수를 독립 표본으로 세지 않는다.

## 전수 결과 및 검증 정보

원본 종합 Artifact에는 `exp012_scale_pair_margins.csv`, `exp012_scale_group_accuracy.csv`, `exp012_geometry_confound_checks.csv`, `all_scale_response.csv`, `all_scale_geometry.csv`가 있다. 현재 폴더의 `summary.csv`는 **모델 3종 × 5개 배율의 Hand-made 숫자/문자 평균 정확도 축약본**이다.
