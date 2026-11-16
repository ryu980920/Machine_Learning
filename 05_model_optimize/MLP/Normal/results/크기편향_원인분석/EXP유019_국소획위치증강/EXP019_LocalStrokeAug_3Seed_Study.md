# EXP유019 (local-only) — 상단·하단 국소 획 ±1px 증강 검증

- Official Train v3 SHA256: `0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879`
- Existing EXP유017 pipeline 80% size (16·18·20·22·24px), stroke 14%, CoM, MLP (256,128), alpha=0.03, LR=0.001, ReLU/Adam, max_iter=200, early_stopping=False
- B1: train-only 20% top x-shift ±1px; B2: same 20% bottom x-shift ±1px; label balanced 16 classes, left/right balanced per class, preserve topology thresholds 96/128, no new train rows
- Validation and HM never augmented; repeated HM2560/new438 used as diagnostic, not independent test or for tuning.
- Baseline = original EXP017 frozen checkpoint, not retrained; B1/B2 freshly fit same 76,800 train rows and same sklearn hyperparameters and seeds.

## 3-seed mean accuracies (%)
| Dataset | Baseline | B1 top | B2 bottom |
|---|---:|---:|---:|
| officialval | 96.5365% | 96.3142% | 96.7830% |
| handmade2560 | 92.0703% | 92.2266% | 92.2266% |
| new480 | 83.0556% | 83.2639% | 84.3750% |
| new438 | 87.8995% | 87.8995% | 89.0411% |

## 7→9 and 9→7 (3-seed total, correlated across same original images)
| Dataset | Variant | True 7 count | 7→9 / 3-seed | True 9 count | 9→7 / 3-seed |
|---|---|---:|---:|---:|---:|
| officialval | baseline | 1253 | 11/3759 | 1190 | 22/3570 |
| officialval | B1_top | 1253 | 11/3759 | 1190 | 29/3570 |
| officialval | B2_bottom | 1253 | 12/3759 | 1190 | 19/3570 |
| handmade2560 | baseline | 160 | 27/480 | 160 | 5/480 |
| handmade2560 | B1_top | 160 | 29/480 | 160 | 7/480 |
| handmade2560 | B2_bottom | 160 | 30/480 | 160 | 6/480 |
| new480 | baseline | 30 | 27/90 | 30 | 3/90 |
| new480 | B1_top | 30 | 21/90 | 30 | 2/90 |
| new480 | B2_bottom | 30 | 29/90 | 30 | 2/90 |
| new438 | baseline | 22 | 27/66 | 26 | 0/78 |
| new438 | B1_top | 22 | 21/66 | 26 | 0/78 |
| new438 | B2_bottom | 22 | 29/66 | 26 | 0/78 |

## OfficialVal 0.8/1.2 scale family-flip gate
- **baseline**: OfficialVal 96.5365%, HM2560 92.0703%, OfficialVal 7→9 11, 9→7 22; **gates passed**
- **B1_top**: OfficialVal 96.3142%, HM2560 92.2266%, OfficialVal 7→9 11, 9→7 29; **rejected: OfficialVal 9→7 increased, B scale family flip >20%**
- **B2_bottom**: OfficialVal 96.7830%, HM2560 92.2266%, OfficialVal 7→9 12, 9→7 19; **gates passed**

## Conclusion boundaries
This is a retrospective controlled training-data intervention. Model performance may vary by seed; all three seeds and pair errors must be reviewed together. This study does not establish the causal shortcut nor demonstrate independent writer generalization. It does not alter GitHub main or release models.
