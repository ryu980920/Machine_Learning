# SVC Normal — optimization log

## Hypothesis and method

Raw pixels are sensitive to stroke position, slant and handwriting style. CoM centering and deskew reduce geometric variation. HOG captures gradient orientation; combining normalized pixel and HOG features may balance local shape and detailed stroke evidence. RBF SVC C/gamma tuned only using pilot subsets before validating top feature candidates on full Training.

## Key evidence

- Baseline full-data: official 97.6667%, Hand-made 61.5625%.
- Pilot CoM: 55.8203% to 72.8125% Hand-made on 8,000-row train (not directly comparable to baseline full).
- Pilot CoM+Deskew: Hand-made 76.7969%.
- Pilot HOG: Hand-made 89.3750%; Hybrid 89.2188%; C=1 gamma=scale favored among nine pilot candidates.
- Full-data HOG: official 98.2656%, Hand-made 91.7188%, 87.89s fit.
- Full-data Hybrid: official 98.8698%, Hand-made 91.6016%, 224.85s fit.

## Selection

Hybrid for superior official Validation performance, with HOG-only retained as a fast, competitive alternative. Do not interpret pilot deltas as full-data ablation causal estimates. Official Test not used.
