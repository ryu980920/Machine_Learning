# EXP-CMB-002 evidence — automatic execution results (research review)

**Status:** analysis completed, final A/B determination pending interpretation and controls.

- Original OOF revalidated: 2,560 Hand-made images, Combined 2,080 correct (81.25%).
- Corrected 540, newly wrong 62, both wrong 418.
- Focused true classes 0/D/8/B: 640 images.
- Fixed-fold model IG diagnostics: 89 distinct images, 178 model-condition rows.
- Stratified writer+class permutation: 2,500 repetitions per global feature, BH-FDR across 7 features.
- Region ablation: predefined regions and 32 random same-foreground-count removals per region.
- **Interpretation limits:** region ablation modifies geometry and brightness mass; it is NOT a pure stroke-shape intervention. B correlation is NOT proof of causal improvement. No new training.

## Feature association (pooled 0/D/8/B, conditioned on writer and actual class)

| Feature | error - correct (within-group z) | permutation p | BH q |
|---|---:|---:|---:|
| cx | 0.072 | 0.3303 | 0.4831 |
| cy | -0.114 | 0.1200 | 0.4831 |
| width | 0.072 | 0.3451 | 0.4831 |
| height | 0.012 | 0.8649 | 0.8649 |
| ink_ratio | -0.074 | 0.3275 | 0.4831 |
| gray_mean | -0.073 | 0.3339 | 0.4831 |
| stroke_proxy | 0.019 | 0.7941 | 0.8649 |

## Region ablation (exploratory coordinate relevance only)

| Region | Images | Mean excess abs margin effect | image bootstrap 95% interval |
|---|---:|---:|---|
| center_horizontal | 89 | 4.082 | [2.438, 5.925] |
| center_vertical | 88 | 0.140 | [-0.765, 1.045] |
| left_middle | 88 | 2.154 | [1.095, 3.291] |
| right_middle | 87 | 1.173 | [0.356, 2.018] |

See evidence_summary.json, feature_comparison.csv, margin_ig_diagnostics.csv, region_ablation_controls.csv and actual real-image grids.

Official original Test and professor Test were not accessed. Baseline EXP-CMB-001 files were not changed.
