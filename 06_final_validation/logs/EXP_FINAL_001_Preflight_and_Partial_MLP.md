# EXP_FINAL_001 — data gate and partial MLP benchmark (2026-11-26)

Status: **2/10 final full-data models trained and evaluated**. Eight models are not measured. Do not treat this as completed model ranking.

## Verified data
- Official split: 76,800 Train / 19,200 development Validation; exact source hash verified by [source audit](https://github.com/ryu980920/Machine_Learning/actions/runs/45775698001).
- New Handmade Validation: 1,280, 320 per collector, 80 per class. Original pixel hashes unique; blank 0. Exact matches and raw cosine >=0.995 pairs against official Train/Val and existing 2,560 handmade all zero. Collector IDs do not prove individual writer disjointness. Raw new set not published in this public repo.
- All 10 configurations passed small-data fit/predict smoke checks. Not full-data reproducibility for other eight.
- Frozen MLP Normal seed42 checkpoint replay reproduced 18,551/19,200. New full fit 18,558/19,200; augmentation 61,440 selected, 57,642 modified, 3,798 rejected matched historical log. Exact refit equivalence not demonstrated.
- Official-only 30%/28% augmentation verified against original prepareraw array.
- Training runtime: Python 3.13.5, scikit-learn 1.8.0, numpy 2.3.5, scipy 1.17.0, scikit-image 0.26.0, AMD EPYC 9V74, 2 computational threads, sequential. HGB's reported 1.9.1 is a known version difference.

## Actual validation results (1280 each, seed42)
| Model | Training rows | Training time s | Preprocess s | Fit s | Inference ms/img | Accuracy | Macro Precision | Macro F1 | Correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP Normal | 76800 | 106.070839 | 14.781871 | 91.288968 | 0.300327 | 88.90625% | 89.22615% | 88.92341% | 1138 |
| MLP Combined | 79360 | 67.387035 | 5.696096 | 61.690940 | 0.301356 | 86.015625% | 86.64575% | 85.98624% | 1101 |
| SVC Normal/Combined | — | — | — | — | — | — | — | — | — |
| KNN Normal/Combined | — | — | — | — | — | — | — | — | — |
| HGB Normal/Combined | — | — | — | — | — | — | — | — | — |
| Extra Trees Normal/Combined | — | — | — | — | — | — | — | — | — |

Training time includes train preprocessing and fit, excludes checkpoint storage and evaluation. Inference is end-to-end one image at a time, 16 warmup, 5 complete passes. MLP Combined full fit uses all four old collection cohorts, not the older OOF fold model. SVC Normal fit was interrupted due to execution length; no incomplete timing used. Remaining models not started beyond smoke tests. No ongoing local process or pending promise of background results. Full benchmark should rerun all 10 on one stable runtime if machine changes.

The new 1280 results are not used to retune either MLP or any other model. Normal and Combined are independently optimized recipes, so this difference is not an isolated causal effect of adding handwriting. Main remains unchanged.
