# EXP-CMB-004 — BBox Size vs CoM Center: 실측 비교 요약

**통과 기준:** 사전등록 문서에 따라 특정 점수 이상이 아니라 어느 방법이 어떤 오류에 왜 효과적인지 **근거와 반례를 함께 설명**하는 것. 이 자동 보고서는 숫자만 제시하며 최종 연구 판정은 별도 실험일지에 작성한다.

## 1. 전체 OOF (16 classes, 4 heldout writers)

| Method | Accuracy | Macro F1 | Δ correct vs baseline | Corrected | Newly wrong | Writers improved | Official Val worst Δpp |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 81.250% | 0.8120 | 0 | 0 | 0 | 0 | — |
| bbox_center | 80.312% | 0.8036 | -24 | 197 | 221 | 1 | -1.896 |
| bbox_size | 80.781% | 0.8096 | -12 | 246 | 258 | 1 | -0.719 |
| com_center | 85.859% | 0.8593 | 118 | 273 | 155 | 4 | -1.448 |
| size_then_com | 88.945% | 0.8898 | 197 | 343 | 146 | 4 | -1.328 |

## 2. 주요 오답 방향별 Count

| Method | 0→D | D→0 | 8→B | B→8 | 합계 |
|---|---:|---:|---:|---:|---:|
| bbox_center | 43 | 35 | 40 | 36 | 154 |
| bbox_size | 33 | 6 | 48 | 4 | 91 |
| com_center | 25 | 35 | 15 | 35 | 110 |
| size_then_com | 13 | 11 | 30 | 14 | 68 |

## 3. Mechanism: 입력에서 실제로 변화한 특징

| Method | Official train BBox long-side std before→after | Hand-made mean CoM offset before→after | Mean image mass change (Hand-made pooled by writers) |
|---|---|---|---:|
| bbox_center | 2.015→2.015 | 1.390→1.385px | 0.0000 |
| bbox_size | 2.015→0.115 | 1.390→1.347px | 0.0957 |
| com_center | 2.015→1.974 | 1.390→0.380px | 0.0001 |
| size_then_com | 2.015→0.115 | 1.390→0.380px | 0.0957 |

## 4. Interaction (descriptive: 4 separate retrained models)

| Group | E − Size − CoM + Baseline (accuracy percentage points) |
|---|---:|
| all | +3.555 |
| digits | +4.438 |
| letters | +2.083 |
| 0 | +11.250 |
| D | +1.875 |
| 8 | +4.375 |
| B | -6.875 |

### Interpretations that still need to be made

- Size vs CoM must be interpreted using error-direction and writer-level transitions together with the actual size/center displacement and margin changes, not Accuracy rank alone.
- Mean input mass change during scaling is NOT pure clipping loss: bilinear resizing changes area and intensity mass. Interpretation must separate scaling from cropping.
- The combined-condition contrast is descriptive of retrained pipelines, NOT a clean causal estimate that holds model weights fixed.
- Hypothesis evidence is developmental OOF on the same four previously examined writers; new-writer external verification is still missing.
- Original Test and professor Test were never accessed.

See: writer_comparison.csv, group_comparison.csv, main_error_pairs.csv, mechanism_by_stratum.csv, geometry_comparison.csv, factorial_interaction.csv and each method subfolder's actual PNG/CSV/models.
