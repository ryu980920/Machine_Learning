# SVC Normal — experiments completed

**Scope:** official Training only. Official Test was not used. Full training=76,800; full Validation=19,200. Hand-made evaluation=2,560 development samples.

| Experiment | Scale | Change / observation |
|---|---|---|
| EXP-001 | Full | Original SVC baseline: official 97.6667%, Hand-made 61.5625%. |
| EXP-002 | Full predictions | Analysis: prominent 5→F, B→8, D→0 errors. |
| EXP-003 | Pilot (8,000/3,200) | CoM increases Hand-made from 55.8203% to 72.8125%; initial unguarded full CoM attempt stalled and was stopped. |
| EXP-004 | Pilot (8,000/3,200) | CoM+Deskew: official 95.9375%, Hand-made 76.7969%. |
| EXP-005 | Pilot (8,000/3,200) | HOG 89.3750% Hand-made, Hybrid 89.2188%; hybrid official 97.9375%. |
| EXP-006 | Pilot (8,000/3,200) | 9 combinations of C/gamma; C=1 and gamma='scale' best on development metrics. |
| EXP-007 HOG | **Full** | Official 98.265625% (18,867/19,200), Hand-made 91.71875% (2,348/2,560), F1 0.915031, fit 87.89s. |
| EXP-007 Hybrid | **Full** | **Official 98.8698%**, Hand-made 91.6015625% (2,345/2,560), F1 0.914145, fit 224.85s. |

**Selected Normal model for official Validation:** CoM → Deskew → Pixel20%:HOG80% → RBF SVC (C=1, gamma='scale'). HOG-only is faster and 3 samples better on Hand-made development, but lower on official Validation.

- Scripts: [scripts/](scripts/)
- Original experiment result files must be copied from the working PC to [results/](results/). Summary numbers here come from recorded terminal outputs.
- Pilot vs full comparison is intentionally labeled; do not describe a pilot result as full-data validation.
