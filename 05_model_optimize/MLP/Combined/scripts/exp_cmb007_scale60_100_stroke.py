#!/usr/bin/env python3
"""EXP-CMB-007: fixed-MLP Size augmentation 60/70/80/90/100% x Stroke 0/14/28%.

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

from exp016_high_fraction_grid import nested_subset as high_fraction_nested_subset

ARMS={f"scale{pct:02d}_stroke{stroke:02d}":(pct/100,stroke/100)
      for pct in (60,70,80,90,100) for stroke in (0,14,28)}
SEEDS=(42,43,44)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm",choices=tuple(ARMS),required=True)
    ap.add_argument("--seed",type=int,choices=SEEDS,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()

    # EXP-CMB-005 run/prepareraw consult this module-level mapping dynamically.
    # Use Normal EXP016 high fraction selections, not Normal EXP015 <=50%.
    # This is a controlled extension preserving all historical 50% source IDs.
    baseline.nested_subset=high_fraction_nested_subset
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
        "experiment":"EXP-CMB-007",
        "reference_experiment":"EXP-CMB-005",
        "reference_script":"exp_cmb005_balanced_scale_stroke.py",
        "pre_registered_arm_set":ARMS,
        "fixed_model_hyperparameters":True,
        "independent_new_models":4,
        "total_planned_fits":180,
        "pre_registered_size_rates":[.6,.7,.8,.9,1.],
        "training_source_selection":"EXP016 high_fraction_grid nested_subset",
        "historical_first_50_percent_ids_retained":True,
        "pre_registered_stroke_strengths":[0.,.14,.28],
        "controls":"Compare paired seeds/writers/images with previously frozen CMB005",
        "original_test_used":False,
        "professor_test_used":False,
        "model_selection":"none; metrics explanatory only"
    })
    mf.write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("EXP_CMB_007_COMPLETED",args.arm,args.seed,info["oof_metrics"],
          flush=True)

if __name__=="__main__":
    main()
