# Experimental protocol, reproducibility, and limitations

## Data split

- Classification task: MNIST digits 0–9 + EMNIST capital letters A–F (16 classes).
- Official Training: 76,800. Official Validation: 19,200. Official Test: 16,000 (excluded).
- Hand-made dataset: 4 writers (41–44), 640 samples each, 2,560 total; 40 samples per class per writer.
- Normal: official Training only; evaluate Official Validation and Hand-made development data.
- Combined: four writer-heldout folds. In each fold, train on official Training 76,800 + other three writers' Hand-made 1,920, and evaluate held-out writer's 640. Augment **training writers only**.
- In Combined experiments, official Validation 19,200 was also checked on each fold. These 4 validation scores must not be presented as 4 independent datasets.

## Algorithms

- SVC, RBF kernel, C=1.0, gamma='scale'; original baseline source: `03_experiments/model_optimization/svc.py`.
- Center-of-mass normalization (CoM) → deskew → features.
- HOG only: 324 dimensions. Hybrid: Pixel 20% + HOG 80%, 1,108 dimensions.
- Hand-made augmentation: size and stroke-thickness perturbations; for 100% add 1,920 training-writer examples per fold. Targeted D/B adds 120 more training-writer examples (3 writers × 2 classes × 20).

## What is and is not proven

- Pilot experiments (EXP-003 through EXP-006) used 8,000 official Training and 3,200 official Validation; their scores are *not* directly comparable to full-data outcomes.
- Full-data Normal validation was done in EXP-007 (HOG and Hybrid), both using 76,800 official Training.
- All Combined model-selection trials train using **the full 76,800 official Training** within each writer fold.
- Repeated tuning on the same 4 Hand-made writers makes the writer-heldout OOF dataset a **development/selection** dataset, not a fresh independent Test.
- Final Official Test and professor-provided hidden Test have not been evaluated here.
- No learning curve is claimed: a few training sizes or feature choices are not a measured learning curve.
- Precise per-fold prediction files, confusion matrices, JSON metrics and runtime logs are stored on the experiment machine, not reproduced in this package. Copy them without editing to preserve provenance.

## GitHub source dependencies

The experiments import repository modules from `03_experiments/model_optimization/` and `05_model_optimize/KNN/Normal/scripts/` / `KNN/Combined/scripts/`. Keep the existing repo structure. Run from the repo root under the working `ml-latest` environment.
