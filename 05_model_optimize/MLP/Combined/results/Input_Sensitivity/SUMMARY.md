# EXP-CMB-003 — Frozen Combined MLP input sensitivity: automatic metrics

**Status:** experimentally executed; final research interpretation in EXP유003 log.
Frozen Normal 1 + Combined 4, original 2,560 Hand-made predictions revalidated.
Official Original Test/Professor Test not loaded, no training.

| Condition | OOF Accuracy | Δpp | Corrected | Newly wrong | Flip rate | McNemar exact p | BH q (10) |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 81.250% | +0.000 | 0 | 0 | 0.00% | 1 | — |
| shift_L1 | 71.172% | -10.078 | 98 | 356 | 22.54% | 1.891e-35 | 1.891e-34 |
| shift_R1 | 78.633% | -2.617 | 155 | 222 | 18.05% | 0.0006564 | 0.0009376 |
| shift_U1 | 71.445% | -9.805 | 131 | 382 | 24.06% | 1.733e-29 | 8.664e-29 |
| shift_D1 | 76.680% | -4.570 | 146 | 263 | 20.12% | 7.73e-09 | 1.546e-08 |
| scale_095 | 83.867% | +2.617 | 132 | 65 | 9.57% | 2.074e-06 | 3.456e-06 |
| scale_105 | 73.945% | -7.305 | 95 | 282 | 17.77% | 1.231e-22 | 4.103e-22 |
| stroke_thin_25 | 82.305% | +1.055 | 50 | 23 | 3.40% | 0.002118 | 0.002648 |
| stroke_thick_25 | 75.625% | -5.625 | 98 | 242 | 16.21% | 3.392e-15 | 8.479e-15 |
| brightness_090 | 81.250% | +0.000 | 6 | 6 | 0.51% | 1 | 1 |
| brightness_110 | 80.977% | -0.273 | 8 | 15 | 1.25% | 0.21 | 0.2334 |

**Gate result:** SCREEN_RETRAIN_CANDIDATES

Screening candidates are not finished models; further 4-Fold Training/Validation evaluation would be a separate study.
No criterion met does not imply no geometry effect: it means these global single-variable inference-time interventions are not a ready-to-deploy solution.
Consult writer_metrics.csv, pair_confusion_changes.csv, focus_margin_diagnostics.csv and official_validation_metrics.csv.
