#!/usr/bin/env python3
"""EXP-CMB-006: fixed-MLP Size augmentation 40/50% x Stroke 0/14/28%.

Reuse the already validated EXP-CMB-005 implementation for all preprocessing,
label splits, raw-dataset SHA checks, OOF 16-class/240-pair diagnostics,
5-scale probes, and model weights. Changing only the 6-arm design ensures that
CMB005 historical code stays completely unchanged.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import exp_cmb005_balanced_scale_stroke as baseline

ARMS={
    "scale40_stroke00":(.4,0.),
    "scale40_stroke14":(.4,.14),
    "scale40_stroke28":(.4,.28),
    "scale50_stroke00":(.5,0.),
    "scale50_stroke14":(.5,.14),
    "scale50_stroke28":(.5,.28),
}
SEEDS=(42,43,44)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm",choices=tuple(ARMS),required=True)
    ap.add_argument("--seed",type=int,choices=SEEDS,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()

    # EXP-CMB-005 run/prepareraw consult this module-level mapping dynamically.
    baseline.ARMS.clear()
    baseline.ARMS.update(ARMS)
    baseline.run(args.arm,args.seed,args.out)

    mf=args.out/"manifest.json"
    info=json.loads(mf.read_text(encoding="utf-8"))
    assert info["experiment"]=="EXP-CMB-005"
    assert info["arm"]==args.arm and info["seed"]==args.seed
    assert len(info["model_sha256"])==4
    assert info["official_n"]==76800
    assert info["official_val_n"]==19200
    assert info["training_handmade_each_fold"]==1920
    assert info["oof_n"]==2560
    assert not info["original_test_used"] and not info["professor_test_used"]
    assert not info["new_handmade_480_used"]
    assert info["no_mlp_hyperparameter_search"]
    requested=ARMS[args.arm][0]
    assert abs(info["augmentation"]["actual_ratio"]-requested)<.002

    info.update({
        "experiment":"EXP-CMB-006",
        "reference_experiment":"EXP-CMB-005",
        "reference_script":"exp_cmb005_balanced_scale_stroke.py",
        "pre_registered_arm_set":ARMS,
        "fixed_model_hyperparameters":True,
        "independent_new_models":4,
        "total_planned_fits":72,
        "pre_registered_size_rates":[.4,.5],
        "pre_registered_stroke_strengths":[0.,.14,.28],
        "controls":"Compare paired seeds/writers/images with previously frozen CMB005",
        "original_test_used":False,
        "professor_test_used":False,
        "model_selection":"none; metrics explanatory only"
    })
    mf.write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("EXP_CMB_006_COMPLETED",args.arm,args.seed,info["oof_metrics"],
          flush=True)

if __name__=="__main__":
    main()
