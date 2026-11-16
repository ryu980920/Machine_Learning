#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP015 stage-2 targeted fine sweep, prompted by the completed 45-model stage 1.

Reuses every data, intervention, family-probability and evaluation implementation
of exp015_scale_stroke_grid. Does NOT change any preregistered shortcut gate.
30% x 28% is deliberately omitted because stage 1 already contains the 3-seed result.
"""
import argparse
import exp015_scale_stroke_grid as core

# 25%, 30%, 35% are between the failed 20% and eligible 30%-40% coarse candidates.
# 21%, 35%, 42% stroke mixing refines between 14% and 28% and tests stronger control.
# Also measure 28% on the previously untested 25% and 35% ratios.
FINE={
    25:(.21,.28,.35,.42),
    30:(.21,.35,.42),
    35:(.21,.28,.35,.42),
}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=core.SEEDS,required=True)
    p.add_argument("--ratio-percent",type=int,choices=tuple(FINE),required=True)
    p.add_argument("--models-root",type=core.Path,required=True)
    p.add_argument("--out",type=core.Path,required=True)
    args=p.parse_args()
    core.FRACTIONS=tuple(x/100 for x in FINE)
    core.STROKES=FINE[args.ratio_percent]
    core.run(args.seed,args.ratio_percent/100,args.out,args.models_root)
if __name__=="__main__":main()
