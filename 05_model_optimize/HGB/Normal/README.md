# HistGradientBoosting Normal — 실험 완료

**범위:** 공식 Training만 학습. Train 76,800 / Validation 19,200 (StratifiedGroupKFold 첫 Fold, 시드 42). Hand-made 2,560장(작성자 41~44)은 개발용 평가에만 사용. 공식 Test 미사용.

| 실험 | 환경 | 변경 | 공식 Val | Hand-made | 판정 |
|---|---|---|---|---|---|
| N1 | Colab | 원본 픽셀 기준 모델 | 97.77% | 57.19% | 기준 |
| N2 | Colab | CoM 정렬 | 97.82% | 73.28% | 채택 |
| N3 | Colab | + 크기 30%·획 28% 증강 | 97.58% | 83.44% | 채택 |
| N4 | Colab | 학습률 0.2 (2만 장 서브셋 선택) | 88.86% | 44.34% | 기각 (26회 붕괴) |
| N5 | Colab | + 투영 56 + HOG 324 | 98.50% | 88.75% | N7로 교체 |
| N6 | Colab | N3 + `max_features=0.5` | 97.57% | 81.88% | 기각 |
| D1~D3 | 별도 | 분기 gain 히트맵, 오분류 사례, 작성자별 글자 기하 | — | — | 원인 분석 |
| 특징 분리 | 별도 | 픽셀+투영 / 픽셀+HOG | 97.91 / 98.39% | 77.38 / 90.66% | 투영 기각 |
| N7 | 별도 | 픽셀 + HOG (3시드) | 98.31% | 90.98 ± 0.34% | N9로 교체 |
| N8 | 별도 | N7 + 정규화 투영 (3시드, 사전 등록) | — | 시드 42 붕괴 | 기각 |
| **N9** | 별도 | N7 + `l2_regularization=1.0` (3시드, 사전 등록) | **98.28%** | **91.43 ± 0.25%** | **최종** |
| N10 | 별도 | N9 + `max_iter=500` (사전 등록) | 98.38% | 91.37% (시드 42) | 기각 |
| N11 | 별도 | N9 + HOG 4×4 (3시드, 사전 등록) | 98.36% | 91.63% | 기각 |

**최종 Normal 모델 N9:** 학습 전처리 = Train 30% 크기(긴 변 16~24px)·획 28% 증강 → CoM(임계 32, 중심 13.5). 평가 전처리 = CoM만. 특징 = 픽셀 784 + HOG 324(7×7 셀, 9방향, 2×2 블록). `HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0, max_iter=300, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)`.

- 보고서: [logs/HGB_Normal_최적화_보고서.pdf](logs/)
- 사전 등록 기준: [logs/](logs/) `PREREG_*.md`
- 전체 수치: [results/HGB_Normal_results.csv](results/HGB_Normal_results.csv) · 학습곡선 `lc_*.csv` · 그림 [results/figures/](results/figures/)
- 코드: [scripts/](scripts/README.md)
