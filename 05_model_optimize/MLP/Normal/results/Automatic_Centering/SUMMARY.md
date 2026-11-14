# EXP유004 — Automatic_Centering (review-only, measured results)

**No official Test, no professor Test, and no Handmade training or hyperparameter selection.**
Same 76,800 Official Training examples and unchanged MLPClassifier(random_state=42).
The current Normal model is the previously validated frozen Stage 3 checkpoint; the two centering models are freshly fit.
Training and inference use the SAME centering method in each new Pipeline.
Preset geometric target = (13.5,13.5) pixel coordinates; threshold=32; integer-pixel translation, no interpolation.

## Official Val and Handmade — paired accuracy / corrections
| Dataset | Model | Accuracy | Δ baseline (pp) | Corrected | Newly wrong | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| OfficialVal | Normal | 97.3177% | +0.000 | 0 | 0 | 0.9730 |
| OfficialVal | Center_of_Mass | 96.9167% | -0.401 | 245 | 322 | 0.9690 |
| OfficialVal | Bounding_Box_Center | 97.3073% | -0.010 | 260 | 262 | 0.9729 |
| Handmade | Normal | 62.8516% | +0.000 | 0 | 0 | 0.6241 |
| Handmade | Center_of_Mass | 77.1094% | +14.258 | 533 | 168 | 0.7704 |
| Handmade | Bounding_Box_Center | 61.6016% | -1.250 | 337 | 369 | 0.6204 |

## Alignment distortion and consistency
| Data | Center | Mean center error before→after | Images with ink loss | Mean lost ink fraction |
|---|---|---|---:|---:|
| OfficialTrain | mass | 1.055→0.381px | 6288/76800 | 0.00030 |
| OfficialTrain | bbox | 1.088→0.163px | 31/76800 | 0.00000 |
| OfficialVal | mass | 1.055→0.383px | 1617/19200 | 0.00032 |
| OfficialVal | bbox | 1.083→0.162px | 4/19200 | 0.00000 |
| Handmade | mass | 1.390→0.380px | 700/2560 | 0.00012 |
| Handmade | bbox | 0.276→0.258px | 2/2560 | 0.00000 |

## Generalization/interpretation safeguards
- This four-writer Handmade set has already been repeatedly inspected in EXP유001-003. Improvements here are **exploratory**, NOT evidence of unseen-writer Test accuracy.
- Official Val was also used in historical research; do not claim completely untouched test performance.
- Since the baseline is already strong on official Validation, inspect the domain gap and class/writer strata, not a tiny official accuracy gain alone.
- Centering can shift, clip or reshape which input pixels the MLP sees. Here only integer translation with zero-fill is used: measure ink clipping separately.
- Class labels/writer IDs are for evaluation ONLY. Neither centering transformer uses them.
- EXP유004 does not determine a final deployed preprocessing policy; await review and independent writer evaluation.
- Historical baseline Official CM L1 difference: 0.

## Files
- exp004_group_metrics.csv: official and Handmade full/class/writer/category paired measures.
- exp004_predictions.csv.gz: per-image predictions (OfficialVal + Handmade) for all 3 models.
- exp004_confusions.csv: 16×16 confusion entries for all models and both domains.
- exp004_alignment_stats.csv: target-centering accuracy, translation and stroke clipping.
- exp004_typical_confusions.csv: selected 5/F, B/8, 0/D confusion changes.
- accuracy_comparison.png: performance chart.
- exp004_run_metadata.json: SHA256 hashes, software versions, training times and fit boundaries.
- *_{mass,bbox}.joblib: trained candidate Pipelines, user review artifact only.
