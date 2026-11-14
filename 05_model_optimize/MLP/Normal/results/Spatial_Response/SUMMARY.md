# EXP유003 Spatial Distribution & MLP Response — Actual execution summary

- Frozen Stage 3 MLP reused, no fitting. Official Test NOT read.
- Official Train 76,800 (raw 72,521, augmented 4,279), Validation 19,200, Handmade 2,560.
- Historical predictions verified: CM-L1=0, Handmade mismatch=0.
- Fixed geometry threshold=32; robustness thresholds=[32, 64, 128].
- Paired input analyses: 130,560 records, same MLP; focus 106 images, IG maps 72.

## Measured geometry (all classes together, exploratory only)
| Data | Center X median | Center Y median | Bounding box width median | height median |
|---|---:|---:|---:|---:|
| OfficialTrain | 13.871 | 13.925 | 18.000 | 20.000 |
| OfficialVal | 13.870 | 13.928 | 18.000 | 20.000 |
| Handmade | 12.983 | 13.357 | 15.000 | 20.000 |

## Paired same-image response and errors
| Dataset | Change | Accuracy | Corrected | Newly wrong | Hidden relative L2 | Mean class-distance change |
|---|---|---:|---:|---:|---:|---:|
| OfficialVal | shift_L1 | 93.271% | 161 | 938 | 0.376 | +0.491 |
| OfficialVal | shift_R1 | 95.214% | 184 | 588 | 0.367 | +0.473 |
| OfficialVal | shift_D1 | 93.411% | 191 | 941 | 0.407 | +0.536 |
| OfficialVal | scale_090 | 93.141% | 162 | 964 | 0.368 | +1.320 |
| OfficialVal | scale_095 | 97.010% | 109 | 168 | 0.192 | +0.068 |
| Handmade | shift_L1 | 46.875% | 96 | 505 | 0.415 | +0.515 |
| Handmade | shift_R1 | 65.391% | 269 | 204 | 0.406 | -0.113 |
| Handmade | shift_D1 | 65.039% | 273 | 217 | 0.440 | -0.064 |
| Handmade | scale_090 | 65.664% | 301 | 229 | 0.358 | +0.628 |
| Handmade | scale_095 | 66.680% | 170 | 72 | 0.185 | +0.070 |

## Interpretation restrictions
- All geometry and prediction associations are observational, even when actual perturbations are applied.
- The perturbations simultaneously change many pixel coordinates and sometimes erase/reshape strokes.
- Class-reference geometry uses TRUE labels for audit only; it is not an allowable deployment rule.
- Grouped augmentation source samples are not independent; do not claim significance from raw counts.
- Integrated Gradients compare the SAME true-vs-original-competitor logit objective within each example.
- Zero-image IG baseline makes dark-background attribution zero by construction.
- This experiment did not select a normalizing transformation or improve a trained model.

See CSV/PNG/NPZ files for exact counts, classes, writers and fixed-model activations.
