# SVC (RBF) — MNIST + EMNIST 16-class optimization

> Completed optimization experiments recorded through 2026-11-22. Baseline, pilot and full-data results are distinguished. **Official Test (16,000 images) was not used in these SVC optimization experiments.**

- [Normal optimization](Normal/README.md): official Training only, 76,800 samples for final full-data experiments.
- [Combined optimization](Combined/README.md): 76,800 official Training + writer-heldout Hand-made Training.
- [Comparison table](results_summary.csv) and [development accuracy plot](development_accuracy.png).
- [Experimental protocol and limitations](EXPERIMENT_PROTOCOL.md).
- The per-experiment Python scripts and original result CSV/JSON artifacts live in the corresponding `Normal/` and `Combined/` directories. This page provides a navigable summary. Summary tables and plots are derived from logged runs; the original result artifacts remain the primary evidence.

## Final candidate results

| Model | Official Validation | Hand-made development | Macro F1 (Hand-made) |
|---|---:|---:|---:|
| SVC Normal baseline | 97.6667% | 61.5625% | 0.604038 |
| SVC Normal HOG full | 98.2656% | 91.7188% | 0.915031 |
| SVC Normal hybrid full | **98.8698%** | 91.6016% | 0.914145 |
| SVC Combined hybrid, writer-heldout | 98.86–98.88% by fold | 93.8672% | 0.938581 |
| Combined +100% Hand-made augmentation | 98.80–98.85% by fold | 94.9219% | 0.949203 |
| Combined +100% augmentation + targeted D/B | 98.79–98.85% by fold | **95.0781%** | **0.950784** |

**Interpretation:** Normal hybrid was selected for higher official Validation accuracy. Combined targeted D/B achieved the best repeatedly-used Hand-made writer-heldout development OOF score. It is **not an independent held-out test result**, due to repeated model selection on the same writers.
