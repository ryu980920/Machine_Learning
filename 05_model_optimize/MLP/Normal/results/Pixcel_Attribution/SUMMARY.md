# Normal MLP baseline: Stage 1+2 XAI report

## Reproduction check
- Official: 76,800 train / 19,200 Validation; accuracy 97.31770833% (published 97.31770833%).
- Official CM cell L1 difference from historical: **0**.
- Handmade: 2,560 development-only images; accuracy 62.85156250% (published 62.85156250%).
- Handmade prediction disagreements versus published CSV: **0 / 2560**.
- Selected images with actual pixels and IG: 98, Official 42, Handmade 56.
- Max Integrated Gradients completeness residual: 0.1257.

## Read the outputs
- `selected_cases.csv`: exact image provenance and objective/probabilities.
- `review_grids/*.png`: original 28x28 images + signed and absolute pixel attributions.
- `selected_pixels_and_IG.npz`: matching images and attribution maps, same order as selected_cases.csv.
- `handmade_all_predictions.csv`: fresh predictions for 2,560 images.
- `run_metadata.json`: checksums, accuracy, reproduction deviations, software versions.

## Interpretation limits
Attribution shows contributions to logits (or predicted-versus-true margin), **not proof** that
the model relies on shape versus brightness/position. Causal confirmation via controlled
image transforms is deliberately deferred to Stage 3 after the user's approval.
Official X_test/y_test were never read. No hand-written images were used in fit().
If historical predictions differ, interpret all attributions as **the retrained baseline**,
not as the historically saved model's exact weights.
