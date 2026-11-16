# EXP유010 — 실측 결과 요약

**원본:** [통합 실험일지](README.md) · [기계 판독용 CSV](summary.csv) · [실제 성공한 Actions](https://github.com/ryu980920/Machine_Learning/actions/runs/43801194796)

- 원본 Official Train 문자의 BBox 긴 변 중앙값: B/D/F **24px**; Hand-made 문자는 **20px**.
- 숫자 8/0/5의 긴 변 중앙값은 Official Train과 Hand-made 모두 **20px**.
- Size 적용 후 B/D/F의 긴 변 중앙값은 두 집단 모두 **20px**.
- Training–Hand-made 긴 변 W1 거리: B **3.6321→0.0190px**, D **3.3921→0.0163px**, F **3.4571→0.0288px**.
- 단, 크기 정규화 알고리즘이 긴 변을 맞추기 때문에 W1 감소 자체가 인과 증명은 아니다. 가로폭·잉크량·형태 차이는 일부 남는다.

## 전수 결과 및 검증 정보

성공한 Actions 원본 Artifact의 `exp010-geometry`에는 `geometry_summary.csv`, `geometry_distance.csv`, `handmade_geometry_per_image.csv`, `manifest.json`이 있다. `Normal-EXP010-012-reviewed-causal-results`에는 `exp010_geometry_core_summary.csv`가 포함돼 있다. 현재 폴더의 `summary.csv`는 **선택한 6개 핵심 클래스의 검증 가능한 축약본**이며 전체 클래스 원본을 대체하지 않는다.
