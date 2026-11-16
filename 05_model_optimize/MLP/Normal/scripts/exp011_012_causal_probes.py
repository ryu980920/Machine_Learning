#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP유011 & EXP유012: original EXP006 model fixed and matched inputs.

Use archived exact 9 EXP006 models from successful GitHub Actions run 43785390739
(C.O.M., Size-only, Size->CoM, each seed42-44). NO retraining. The original
model's diagonal predictions must exactly match archived manifests and per-group
correct counts before counterfactual probes are accepted.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import affine_transform

from exp006_size_com_comparison import (
    ROOT, WRITERS, LABELS, PAIRS, load_official, load_writers,
    image_geometry, process, save_csv, sha256
)
from Automatic_Centering_Transformer import center_batch,center_shifts
from stroke_normalization import topology

SEEDS=(42,43,44)
TRAINED=("com","size","size_com")
EVALS=("raw","com","size","size_com")
SCALES=(.8,.9,1.0,1.1,1.2)
COMPETITOR={"B":"8","8":"B","D":"0","0":"D","F":"5","5":"F"}

def evaluate_groups(truth,pred,writers=None):
    groups={"all":np.ones(len(truth),bool),
            "digits":truth<10,"letters":truth>=10}
    for i,cl in enumerate(LABELS):
        groups["class_"+cl]=truth==i
    if writers is not None:
        for w in WRITERS:
            groups[f"writer_{w}"]=writers==w
            groups[f"writer_{w}_digits"]=(writers==w)&(truth<10)
            groups[f"writer_{w}_letters"]=(writers==w)&(truth>=10)
    out={}
    for name,mask in groups.items():
        n=int(mask.sum())
        if n:
            out[name]={"n":n,"correct":int(np.sum(pred[mask]==truth[mask])),
                       "accuracy":float(np.mean(pred[mask]==truth[mask]))}
    return out

def fixed_scale(batch,scale):
    """Uniform bbox-centered affine, identical order/cval as original EXP006."""
    a=np.asarray(batch,dtype=np.float32).reshape(-1,28,28)
    if scale==1.0:return a.copy().reshape(-1,784)
    g=image_geometry(a)
    out=np.empty_like(a)
    inv=1.0/scale
    for i,im in enumerate(a):
        cy,cx=g["bbox_cy"][i],g["bbox_cx"][i]
        out[i]=np.clip(affine_transform(im,np.eye(2)*inv,
                    offset=(cy*(1-inv),cx*(1-inv)),output_shape=(28,28),
                    order=1,mode="constant",cval=0.0,prefilter=False),0,255)
    return out.reshape(-1,784)

def paired(a,b,mask):
    old=a[mask];new=b[mask]
    return {"n":int(mask.sum()),
      "old_correct":int(old.sum()),"new_correct":int(new.sum()),
      "fixed":int((~old&new).sum()),
      "new_wrong":int((old&~new).sum()),
      "net_correct":int(new.sum()-old.sum()),
      "delta_accuracy_pp":float(100*(new.mean()-old.mean()))}

def run(seed,models_root,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    official=load_official(ROOT/"02_data"/"datasets"/"MNIST_EMNIST_official_split_112000.npz")
    hm=load_writers(ROOT/"02_data"/"Hand_made")
    raw_handmade=np.concatenate([hm[w]["X"] for w in WRITERS])
    y_hm=np.concatenate([hm[w]["y"] for w in WRITERS]).astype(int)
    writer_ids=np.repeat(WRITERS,640)
    y_val=np.asarray(official["y_val"],dtype=int)
    assert len(raw_handmade)==2560 and official["X_val"].shape==(19200,784)
    target=int(np.median(image_geometry(official["X_train"])["long_side"]))
    assert target==20
    raw_sets={"OfficialVal":official["X_val"],"Handmade":raw_handmade}
    preprocessed={}
    for dataset,raw in raw_sets.items():
        for method in EVALS:
            preprocessed[dataset,method]=(
              np.asarray(raw,dtype=np.float32).reshape(-1,784) if method=="raw"
              else process(raw,method,target)
            )
    models={}
    historical={}
    for method in TRAINED:
        folder=Path(models_root)/f"normal-exp006-{method}-{seed}"
        manifest=json.loads((folder/"run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["method"]==method and manifest["seed"]==seed
        assert manifest["official_sha256"]==official["sha256"]
        assert all(manifest["writer_sha256"][str(w)]==hm[w]["sha256"] for w in WRITERS)
        assert not manifest["official_test_used"] and not manifest["professor_test_used"]
        assert not manifest["handmade_used_in_fit"]
        model_path=folder/"model.joblib"
        assert sha256(model_path)==manifest["model_sha256"],"Archived checkpoint hash mismatch"
        models[method]=joblib.load(model_path)
        historical[method]=manifest
        print("LOADED AND HASH VERIFIED",method,seed,flush=True)

    cross_metrics=[]
    cross_predictions=[]
    prediction_lookup={}
    for method,model in models.items():
        for dataset in ("OfficialVal","Handmade"):
            truth=y_val if dataset=="OfficialVal" else y_hm
            writers=None if dataset=="OfficialVal" else writer_ids
            for representation in EVALS:
                X=preprocessed[dataset,representation]
                pred=np.asarray(model.predict(X),dtype=int)
                assert len(pred)==len(truth)
                if representation==method:
                    g=evaluate_groups(truth,pred,writers)
                    for group in ("all","digits","letters"):
                        historical_correct=historical[method][
                            "official_validation" if dataset=="OfficialVal" else "handmade"
                        ][group]["correct"]
                        actual=g[group]["correct"]
                        if actual!=historical_correct:
                            raise AssertionError(f"HISTORICAL MISMATCH {method} {seed} "
                              f"{dataset} {group}: recorded={historical_correct} current={actual}")
                    print("EXACT ORIGINAL MODEL/RESULT MATCH",method,seed,dataset,
                          "correct",g["all"]["correct"],flush=True)
                group_data=evaluate_groups(truth,pred,writers)
                for group,val in group_data.items():
                    cross_metrics.append({"seed":seed,"trained":method,
                        "dataset":dataset,"input":representation,"group":group,**val})
                if dataset=="Handmade":
                    prediction_lookup[method,representation]=pred
                    for i,(t,p) in enumerate(zip(truth,pred)):
                        cross_predictions.append({"seed":seed,"trained":method,
                            "input":representation,"writer":int(writer_ids[i]),
                            "index":int(i%640),"true":LABELS[t],
                            "pred":LABELS[p],"correct":int(t==p)})
                print("EXP011_CROSS",seed,method,dataset,representation,
                      "all",f"{group_data['all']['accuracy']*100:.3f}%",
                      "digits",f"{group_data['digits']['accuracy']*100:.3f}%",
                      "letters",f"{group_data['letters']['accuracy']*100:.3f}%",flush=True)
    save_csv(out/"cross_metrics.csv",cross_metrics)
    save_csv(out/"cross_predictions.csv",cross_predictions)
    paired_results=[]
    diag_old=(prediction_lookup["size","size"]==y_hm)
    diag_new=(prediction_lookup["size_com","size_com"]==y_hm)
    for group,mask in (("all",np.ones(len(y_hm),bool)),
                       ("digits",y_hm<10),("letters",y_hm>=10),
                       *[(f"class_{cl}",y_hm==j) for j,cl in enumerate(LABELS)]):
        result=paired(diag_old,diag_new,mask)
        paired_results.append({"seed":seed,"contrast":"size diagonal -> size_com diagonal",
                               "group":group,**result})
        for method in TRAINED:
            a=(prediction_lookup[method,"size"]==y_hm)
            b=(prediction_lookup[method,"size_com"]==y_hm)
            paired_results.append({"seed":seed,"contrast":f"fixed model={method}: size input -> size_com input",
                                   "group":group,**paired(a,b,mask)})
        for rep in ("size","size_com"):
            a=(prediction_lookup["size",rep]==y_hm)
            b=(prediction_lookup["size_com",rep]==y_hm)
            paired_results.append({"seed":seed,"contrast":f"fixed input={rep}: size train -> size_com train",
                                   "group":group,**paired(a,b,mask)})
    save_csv(out/"paired_effects.csv",paired_results)
    pair_errors=[]
    for method in TRAINED:
        for inp in EVALS:
            p=prediction_lookup[method,inp]
            for a,b in PAIRS:
                pair_errors.append({"seed":seed,"trained":method,"input":inp,
                   "true":a,"pred":b,
                   "n_errors":int(np.sum((y_hm==LABELS.index(a)) &
                                             (p==LABELS.index(b))))})
    save_csv(out/"pair_errors.csv",pair_errors)

    # EXP011 controls: geometry before/after CoM on the same Size-only image.
    sized=preprocessed["Handmade","size"]
    corrected=preprocessed["Handmade","size_com"]
    sized_geo=image_geometry(sized)
    centered_geo=image_geometry(corrected)
    shifts=center_shifts(sized,"mass")
    geometry=[]
    for i in range(len(y_hm)):
        geometry.append({"seed":seed,"writer":int(writer_ids[i]),
            "index":int(i%640),"class":LABELS[int(y_hm[i])],
            "size_com_shift_x":int(shifts[i,0]),"size_com_shift_y":int(shifts[i,1]),
            "size_com_shift_distance":float(np.hypot(*shifts[i])),
            "com_distance_before":float(sized_geo["com_offset"][i]),
            "com_distance_after":float(centered_geo["com_offset"][i]),
            "long_side_before":int(sized_geo["long_side"][i]),
            "long_side_after":int(centered_geo["long_side"][i]),
            "mass_before":float(sized_geo["mass"][i]),
            "mass_after":float(centered_geo["mass"][i]),
            "edge_mass_before":float(sized_geo["edge_mass"][i]),
            "edge_mass_after":float(centered_geo["edge_mass"][i]),
            "size_model_size_correct":int(diag_old[i]),
            "size_com_model_size_com_correct":int(diag_new[i]),
            "same_size_model_after_com_correct":int(
                 prediction_lookup["size","size_com"][i]==y_hm[i]),
            "same_size_com_model_before_com_correct":int(
                 prediction_lookup["size_com","size"][i]==y_hm[i]),
            })
    save_csv(out/"size_com_geometry_and_outcomes.csv",geometry)

    # EXP012: perturb SAME image's uniform BBox size while re-centering each
    # perturbed image, keeping source class and model fixed.
    raw=np.asarray(raw_handmade,dtype=np.float32).reshape(-1,784)
    unscaled_mass=image_geometry(raw)["mass"]
    unscaled_topo=[topology(img) for img in raw.reshape(-1,28,28)]
    scale_rows=[]
    scale_geometry=[]
    for scale in SCALES:
        src=fixed_scale(raw,scale)
        dxdy=center_shifts(src,"mass")
        x=center_batch(src,"mass")
        geo=image_geometry(x)
        for i in range(len(y_hm)):
            if y_hm[i] in [LABELS.index(k) for k in COMPETITOR]:
                changed=int(topology(x[i].reshape(28,28))!=unscaled_topo[i])
            else:
                changed=-1
            scale_geometry.append({"seed":seed,"writer":int(writer_ids[i]),
              "index":int(i%640),"true":LABELS[int(y_hm[i])],
              "scale":scale,
              "bbox_width_after":float(geo["width"][i]),
              "bbox_height_after":float(geo["height"][i]),
              "bbox_longside_after":float(geo["long_side"][i]),
              "mass_ratio_after_over_original":float(geo["mass"][i]/
                                    max(unscaled_mass[i],1e-10)),
              "com_offset_after":float(geo["com_offset"][i]),
              "shift_x":int(dxdy[i,0]),"shift_y":int(dxdy[i,1]),
              "topology_changed_for_focus":changed})
        for trained,model in models.items():
            proba=np.asarray(model.predict_proba(x),dtype=float)
            labels=np.asarray(model.named_steps["model"].classes_,dtype=int)
            assert len(labels)==16 and np.array_equal(labels,np.arange(16))
            pred=proba.argmax(axis=1)
            for i in range(len(y_hm)):
                true=int(y_hm[i]); cl=LABELS[true]
                confuser=COMPETITOR.get(cl)
                true_score=proba[i,true]
                runner_up=max(proba[i,j] for j in range(16) if j!=true)
                rival_idx=LABELS.index(confuser) if confuser else -1
                scale_rows.append({
                    "seed":seed,"trained":trained,"writer":int(writer_ids[i]),
                    "index":int(i%640),"true":cl,
                    "scale":scale,"pred":LABELS[int(pred[i])],
                    "correct":int(pred[i]==true),
                    "p_true":float(true_score),
                    "p_best_other":float(runner_up),
                    "true_vs_best_margin":float(true_score-runner_up),
                    "confuser":confuser if confuser else "",
                    "p_confuser":float(proba[i,rival_idx]) if confuser else "",
                    "true_vs_pair_margin":float(true_score-proba[i,rival_idx]) if confuser else ""
                })
            group=evaluate_groups(y_hm,pred,writer_ids)
            print("EXP012_SCALE",seed,trained,scale,"all",
                  round(100*group["all"]["accuracy"],3),"digits",
                  round(100*group["digits"]["accuracy"],3),"letters",
                  round(100*group["letters"]["accuracy"],3),flush=True)
    save_csv(out/"scale_class_response.csv",scale_rows)
    save_csv(out/"scale_geometry.csv",scale_geometry)
    manifest={"experiment":"Normal-EXP011-EXP012","seed":seed,
        "archived_original_run":43785390739,
        "original_models_checked":["com","size","size_com"],
        "checkpoints_sha256":{m:historical[m]["model_sha256"] for m in TRAINED},
        "exact_original_prediction_counts_verified":True,
        "training_performed":False,
        "size_preprocessing":"EXP006 Combined 32 threshold 20px longside bilinear",
        "com_preprocessing":"EXP006 mass center threshold32 to 13.5px",
        "scales":list(SCALES),
        "scale_intervention":"affine around raw bbox center, then CoM centered",
        "handmade_n":2560,
        "official_validation_n":19200,
        "train_source_sha256":official["sha256"],
        "official_original_test_used":False,
        "professor_test_used":False,"handmade_training_used":False,
        "scale_topology_footprint_measured":True}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("EXP011_EXP012_COMPLETE",seed,flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--seed",type=int,choices=SEEDS,required=True)
    a.add_argument("--models-root",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    args=a.parse_args()
    run(args.seed,args.models_root,args.out)
