# EXP유004 follow-up — CoM decision diagnostic (review only)

- The same 2,560 Handmade files (not independent unseen-writer Test).
- No new training, no Test, no hyperparameter tuning, no model deployment.
- Compare Normal_raw vs Normal_CoM: input shift with identical weights.
- Compare Normal_CoM vs CoMModel_CoM: weights difference on identical centered images.
- CoMModel_raw is **off the new model training distribution**, diagnostic only.
- IG uses signed margin true-class logit minus fixed rival class (5-F or B-8).
- Red IG region supports true class *relative to rival*, blue supports rival.
- A pixel IG map is not evidence of human-like shape understanding or proof of causality.

| Group | Normal/raw | Normal/CoM shift | CoM-model/raw (diagnostic) | CoM-model/CoM |
|---|---:|---:|---:|---:|
| all | 62.85% | 70.90% | 64.84% | 77.11% |
| digits_0-9 | 56.31% | 82.00% | 72.44% | 83.19% |
| letters_A-F | 73.75% | 52.40% | 52.19% | 66.98% |
| true_5 | 44.38% | 78.75% | 56.25% | 73.75% |
| true_B | 40.00% | 31.87% | 26.25% | 33.12% |

## Paired contrasts (causal controlled manipulations on the same samples)
### digits_0-9
| Comparison | Accuracy delta (pp) | Corrected | New errors |
|---|---:|---:|---:|
| Movement_same_normal_weights | +25.687 | 455 | 44 |
| New_weights_same_centered_pixels | +1.188 | 108 | 89 |
| Actual_pipeline_delta | +26.875 | 479 | 49 |
### letters_A-F
| Comparison | Accuracy delta (pp) | Corrected | New errors |
|---|---:|---:|---:|
| Movement_same_normal_weights | -21.354 | 29 | 234 |
| New_weights_same_centered_pixels | +14.583 | 185 | 45 |
| Actual_pipeline_delta | -6.771 | 54 | 119 |

## Selected pairs

- True 5, baseline F → CoM model 5: 4 illustrated examples.
- True B, baseline B → CoM model 8: 3 illustrated examples.

## Files
- four_way_summary.csv, contrasts.csv: exact paired group results
- four_way_predictions.csv: same-image 2×2 controls for all 2,560
- pairwise_margins.csv: true 5-vs-F and B-vs-8 logit margins
- selected_ig_diagnostics.csv: per-example signed IG and completeness
- IG_5_wrong_F_to_5.png; IG_B_correct_to_wrong_8.png: actual image/IG grids
- metadata.json: inputs, checkpoint hashes, no-Test confirmation
