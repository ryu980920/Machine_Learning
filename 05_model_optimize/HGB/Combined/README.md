# HistGradientBoosting Combined — 실험 완료

**범위:** 공식 Train 76,800장 + 보류 작성자를 제외한 3명의 Hand-made 1,920장으로 학습하고, 보류 작성자 640장으로 평가. 작성자 41·42·43·44를 바꿔 4번 학습한 예측을 모아 2,560장 OOF 계산. 전처리·특징은 Normal 결과를 출발점으로 사용. 공식 Test 미사용.

| 실험 | 환경 | 변경 | Hand-made OOF | 판정 |
|---|---|---|---|---|
| C1 | Colab | N3 전처리, 픽셀 | 86.41% | 기준 (0↔D 오류 방향만 바뀜) |
| C2 | 별도 | 픽셀+투영+HOG | 89.80% | 작성자 41 Fold 53회 붕괴 |
| C3 | 별도 | 픽셀+HOG (3시드) | 92.63 ± 1.42% | 시드 43 작성자 41 Fold 붕괴 |
| C4 | 별도 | C3 + Hand-made 학습분에도 크기·획 증강 (사전 등록) | 93.32% (시드 42) | 기각 (기준 ①·④ 미충족) |
| C5 | 별도 | C3 + `n_iter_no_change=30` (사전 등록) | 첫 Fold 37.66% | 중단 (가설 반증 → 잎 값 폭발 진단) |
| **C6** | 별도 | C3 + `l2_regularization=1.0` (3시드, 사전 등록) | **93.39 ± 0.47%** | **최종** |

C6: 12개 Fold 모두 209~284회 안정 학습, 작성자 41/42/43/44 = 89.3 / 93.6 / 97.5 / 93.1%, 공식 Validation Fold 평균 98.26%, Macro F1 0.9339.

- 보고서: [logs/HGB_Combined_최적화_보고서.pdf](logs/)
- 사전 등록 기준: [logs/](logs/) `PREREG_*.md`
- 전체 수치: [results/HGB_Combined_results.csv](results/HGB_Combined_results.csv) · 잎 값 진단 `diag41_*.json` · 학습곡선 `lc_*.csv` · 그림 [results/figures/](results/figures/)
- 코드: [scripts/](scripts/README.md) — 특징 캐시(`Ftr.npy` 등)는 Normal `run_all.py n5`가 `Normal/scripts/out/`에 만든 것을 사용
