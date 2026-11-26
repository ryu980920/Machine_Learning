# Final Handmade Validation — 10 frozen models (2026-11-26)

Status: source audit and dataset gate passed; final full-data benchmark not yet reported. Do not merge into main before result review.

## Approved protocol

Mode B: official training 76,800 images, Combined uses all existing 2,560 Handmade samples from collection groups 41–44. The 1,280 new images are evaluation only; collector labels are not verified identities of individual handwriting contributors. Original Test and Professor Test are not accessed. Prior extra 480 and screened 438 are not added to training.

Final outputs: ten standalone models, Training Time (augmentation + feature extraction + fit), Inference Time (batch 1 end-to-end, warmup then 5 passes), accuracy, macro precision, macro F1. Seed 42; common CPU runtime, two threads, sequential fit. Do not tune after seeing new validation scores.

## Frozen recipes

| Model | Normal | Combined | Final training rows N/C |
|---|---|---|---:|
| MLP | EXP017 slot1_alpha_up | CMB011/base, 4 cohorts | 76800 / 79360 |
| SVC | EXP007 hybrid | CMB005 extra20 per D/B class per cohort | 76800 / 82080 |
| KNN | EXP014 targeted10 / EXP015 k3 | CMB005 pixel20 HOG80 + repeat_mixed_50 | 107520 / 113920 |
| HGB | N9 | C6 | 76800 / 79360 |
| Extra Trees | T02_feat18 | T02_feat18 | 76800 / 79360 |

Main source: 476672f224a3cb5e7b652af4841626bdced46c69. Extra Trees exact source: ec30f7fd4a97de9f8336df2eef6a52b394c810c0. SVC/KNN four-cohort augmentation extends the original three-cohort rules; fold-offset sentinel 0 preselected before new validation. Common scikit-learn 1.8.0 differs from HGB report's stated 1.9.1; exact re-fit parity is not assumed.

## Actual source and input audit

- Source export and original development split verification: https://github.com/ryu980920/Machine_Learning/actions/runs/45775698001.
- Official source SHA-256: 0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879.
- Local combined new NPZ SHA-256: 73de49ac0a0e05eba5099a0a8c31b332ac8abec3fd7eebbc69eb875210770f31; 1280 unique pixel hashes, 80 samples per class, 320 per collector, zero empty samples and exact duplicates.
- Cross-source exact pixel and raw cosine >=0.995 checks: zero against official training 76800, official development validation 19200, and existing Handmade 2560.
- Local smoke checks succeeded for all ten configurations (NOT full-data fit or new-set performance).
- Existing frozen MLP Normal seed42 checkpoint matched SHA and reproduced 18551/19200 official development predictions in the local runtime. Official-only 30%-size/28%-stroke array reproduced exactly from the original augmentation function.
- The new raw validation data remains local, not published in this public repository. Audit threshold does not prove absence of shifted/cropped duplicates or actual writer identity overlap.

No new performance claims are made by this document. No changes to the deployed model or main.
