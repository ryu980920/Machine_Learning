# EXP-CMB-002 continuation — morphology and controlled pixel swaps

- Original held-out OOF and 5 frozen models re-read; no MLP training.
- Same 640 focus images. Global morphology feature permutations: 3,000 within-writer-and-class shuffles, BH across 11.
- Five preselected writer/class strata: 2,000 shuffles, BH across all 5×18 tests.
- Pixel swap samples: 62 selected, 61 with strict valid proposals, full 0–255 histogram and bbox preserved.

## Global morphology association

| Feature | error-minus-correct z | p | q |
|---|---:|---:|---:|
| holes_128 | -0.103 | 0.1753 | 0.2754 |
| connected_components_128 | 0.107 | 0.0583 | 0.1583 |
| left_mass_fraction | -0.092 | 0.2093 | 0.2877 |
| right_mass_fraction | -0.011 | 0.8830 | 0.9713 |
| middle_mass_fraction | 0.080 | 0.2839 | 0.3470 |
| upper_mass_fraction | 0.146 | 0.0530 | 0.1583 |
| central_row_mass_fraction | 0.150 | 0.0410 | 0.1583 |
| left_column_mass_fraction | -0.001 | 0.9930 | 0.9930 |
| bilateral_asymmetry | 0.119 | 0.1146 | 0.2102 |
| upper_lower_asymmetry | 0.225 | 0.0030 | 0.0330 |
| centerline_foreground_128 | 0.134 | 0.0720 | 0.1583 |

## Image-level topology intervention contrasts

| Group | N images | Selected - control absolute margin | Bootstrap 95% CI | Mean connection change (selected / control) |
|---|---:|---:|---|---|
| all | 61 | 0.076 | [-0.090, 0.246] | -11.24 / -8.71 |
| 0D | 29 | 0.105 | [-0.125, 0.336] | -11.67 / -9.26 |
| 8B | 32 | 0.051 | [-0.184, 0.288] | -10.85 / -8.21 |

## Targeted writer-class tests

BH q<0.05 among 5 selected strata × 18 features: 6.
This is **exploratory** even if significant; selected strata arise from baseline errors.
See targeted_writer_class_tests.csv for all adjusted p-values.

**Limits:** the pixel swaps are matched on 4 exchanges and displacement and preserve histogram, bounding box, thresholded foreground pixel count, with <=0.3px CoM movement. Four-neighbour connectivity is only a morphological proxy. No final causal claim of semantic shape recognition.
