# SVC Combined — optimization log

## Design

Keep all official Training (76,800) in each of four writer-heldout folds, hold out one entire Hand-made writer (640), train with the remaining three writers' original (1,920) and training-only augmented examples. Report the 2,560 pooled OOF development predictions.

## Decisions

1. CMB-001: Hybrid20:80 beats Normal Hybrid development 91.6016% → 93.8672%; 64 corrected, 6 new errors.
2. CMB-002: HOG only gives 93.5547%; slightly below Hybrid 93.8672%, though faster.
3. CMB-003: Hybrid augmentation 0/25/50/100% gives 93.8672 / 94.2969 / 94.6875 / 94.9219%, respectively.
4. CMB-004: 130 errors remain at 100%, 122 persist across 25/50/100% candidates. D→0 (24), B→8 (13), B→0 (8) dominate; motivates targeted augmentation.
5. CMB-005: 120 D/B targeted training-only augmented examples improves 94.9219% to 95.0781% (+4/2,560), Macro F1 0.950784. D/B recalls improve; 0 does not decline and 8 improves.

## Limitations

The same four held-out writers were reused for selecting features, augmentation rates and target classes. Hence 95.0781% is a *development OOF* estimate, not independent unseen-writer Test performance. Fold-wise Official Validation at CMB-005 ranges 98.7917–98.8490%, versus Normal Hybrid 98.8698%. No Official Test run was conducted.
