# Normal MLP Input_Sensitivity — Stage 3 result

- Source: 19,200 official Validation + 2,560 Handmade.
- Original accuracy: official 97.317708%; Handmade 62.851562%.
- Historical reproduction check: official CM L1=0; Handmade prediction discrepancies=0.
- Prior 98 focus samples with changed predictions in this refit: 0.
- Model identity: same-data, same-hyperparameter independent refit; results are paired within THIS refit.
- Any discrepancy from Stage 1-2 must not be presented as an identical-weight comparison.
- Controlled modifications: 8 shift, 4 brightness, 4 stroke, 4 scale = 20, plus original.
- Total scored rows: 456,960; focus examples: 106 (98 previous + 8 extra).
- No official Test; no Handmade fitting, refit only once on Normal official Training.

## Highest observed prediction flip conditions (each dataset)

| Dataset | Transform | Flip rate | Accuracy delta(pp) | Corrected wrong | Newly wrong |
|---|---|---:|---:|---:|---:|
| OfficialVal | shift_U2 | 32.86% | -30.24 | 138 | 5945 |
| OfficialVal | shift_L2 | 31.03% | -28.40 | 150 | 5602 |
| OfficialVal | shift_D2 | 27.39% | -24.63 | 167 | 4896 |
| OfficialVal | shift_R2 | 18.39% | -15.68 | 175 | 3186 |
| OfficialVal | scale_110 | 13.57% | -11.41 | 143 | 2333 |
| Handmade | shift_U2 | 59.73% | -29.77 | 135 | 897 |
| Handmade | shift_L2 | 55.08% | -29.69 | 76 | 836 |
| Handmade | shift_D2 | 44.88% | -4.53 | 333 | 449 |
| Handmade | shift_R2 | 40.94% | -0.78 | 355 | 375 |
| Handmade | scale_110 | 40.12% | -14.92 | 176 | 558 |

## How to read these results

- Flip rate: original predicted class differs from transformed prediction.
- Correction rate: among originally incorrect inputs, fraction now correct.
- Induction rate: among originally correct inputs, fraction now incorrect.
- Accuracy delta: corrected count minus newly wrong count divided by all examples.
- A high flip rate is sensitivity, not automatically an improvement.
- Brightness clipping is measured explicitly. Thickness/scale transforms can alter discriminative strokes.
- The comparison cannot by itself prove the model learned semantics rather than pixel correlations.
- This is a diagnostic, not a selected preprocessing policy or a final performance gain.

## Output index

- stage3_predictions.csv.gz: ALL sample-level predictions (gzip-compressed CSV).
- stage3_sensitivity_summary.csv: 21 conditions x 2 datasets.
- stage3_per_class.csv: per-class metrics.
- stage3_per_writer.csv: writer-level metrics.
- stage3_error_pairs.csv: selected confusion-pair changes.
- stage3_focus_106.csv: 106 original/focus images x 21 conditions.
- stage3_example_grid.png and stage3_probability_curves.png: focused before/after views.
- stage3_run_metadata.json: environment, conditions, reproducibility checks.
- stage3_baseline_prediction_differences.csv: changed historical Handmade predictions, if any.
- normal_mlp_refit_stage3.joblib: frozen weights for this run, in artifact only.
