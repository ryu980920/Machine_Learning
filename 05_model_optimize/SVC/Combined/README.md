# SVC Combined — full-data writer-heldout optimization

**All Combined folds use official Training 76,800.** Hand-made training writers=3 (1,920 original); held-out writer=1 (640). Four folds cover writers 41–44, resulting in 2,560 out-of-fold *development* predictions. Official Test excluded.

| Experiment | Feature / augmentation | Total training per fold | OOF correct / 2,560 | OOF accuracy | OOF Macro F1 |
|---|---|---:|---:|---:|---:|
| EXP-CMB-001 | Hybrid20:80, none | 78,720 | 2,403 | 93.8672% | 0.938581 |
| EXP-CMB-002 | HOG only, none | 78,720 | 2,395 | 93.5547% | 0.935035 |
| EXP-CMB-003 (25%) | Hybrid, 480 augmented | 79,200 | 2,414 | 94.2969% | 0.942904 |
| EXP-CMB-003 (50%) | Hybrid, 960 augmented | 79,680 | 2,424 | 94.6875% | 0.946852 |
| EXP-CMB-003 (100%) | Hybrid, 1,920 augmented | 80,640 | 2,430 | 94.9219% | 0.949203 |
| EXP-CMB-004 | Saved-result error analysis, no training | — | 2,430 | 94.9219% | 0.949203 |
| **EXP-CMB-005** | **Hybrid, 1,920 augmented + D/B targeted 120** | **80,760** | **2,434** | **95.0781%** | **0.950784** |

**Selected Combined development candidate:** CoM → Deskew → Pixel20%:HOG80%, RBF SVC C=1 gamma='scale', plus base and targeted D/B training-writer augmentation.

### Targeted augmentation diagnosis

- EXP-CMB-004 (100% augmented) had 130 wrong; dominant confusions D→0 (24), B→8 (13), B→0 (8). D recall=83.13%, B recall=84.38%.
- EXP-CMB-005 120 targeted samples: D recall 84.38%, B recall 85.00%, 0 remains 96.25%, 8 rises from 93.75% to 94.37%. Net +4 correct vs 100% control (5 corrected, 1 new error).
- Official Validation accuracy at EXP-CMB-005 by fold: 98.8490%, 98.7917%, 98.8073%, 98.7969%. This is slightly below the Normal Hybrid 98.8698% and is a potential trade-off.

### Reproducibility

- Scripts: [scripts/](scripts/)
- Outputs should be copied intact from the working PC's [results/](results/); each experiment creates its own directory and CSV/JSON artifacts.
- All Hand-made OOF scores are **repeated-use model-selection scores**, not unbiased new-writer generalization estimates.
