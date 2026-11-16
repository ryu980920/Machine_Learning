#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-grid fixed-model OfficialVal scale stress test of all 9 eligible EXP016 models.

This is a diagnostic only. Official Val is already a repeatedly used development
set; this does NOT create an independent holdout or alter the original Gate.
No training or test data access; all 10 checkpoints fixed.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import joblib
import numpy as np
from exp006_size_com_comparison import (
    ROOT, WRITERS, LABELS, PAIRS, load_official, load_writers,
    image_geometry, process, save_csv, sha256)
from exp011_012_causal_probes import fixed_scale
from exp014_balanced_augmentation import original_model

SCALES=(.8,.9,1.,1.1,1.2)
CANDIDATES={
    "size70_stroke14":("new",70),
    "size80_stroke14":("new",80),
    "size70_stroke00":("new",70),
    "size90_stroke42":("new",90),
    "size90_stroke14":("new",90),
    "size90_stroke00":("new",90),
    "size30_stroke28":("old",30),
    "size40_stroke28":("old",40),
    "size50_stroke28":("old",50),
}

def run(seed,new,old,controls,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    official=load_official(ROOT/"02_data"/"datasets"/"MNIST_EMNIST_official_split_112000.npz")
    handmade=load_writers(ROOT/"02_data"/"Hand_made")
    y=np.asarray(official["y_val"],dtype=int)
    X=official["X_val"]
    assert X.shape==(19200,784) and set(np.unique(y))==set(range(16))
    target=int(np.median(image_geometry(official["X_train"])["long_side"]))
    assert target==20
    baseline,h=original_model(controls,seed,"com",official,handmade)
    models={"baseline_com":baseline}
    hashes={"baseline_com":h["model_sha256"]}
    for name,(source,ratio) in CANDIDATES.items():
        folder=(Path(new)/f"exp016-s{seed}-r{ratio}" if source=="new"
                else Path(old)/f"exp015-s{seed}-r{ratio}")
        manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        assert manifest["seed"]==seed and manifest["official_source_sha256"]==official["sha256"]
        assert manifest["original_predictions_verified"]
        assert not manifest["official_test_used"] and not manifest["professor_test_used"]
        path=folder/f"{name}.joblib"
        assert path.is_file(),path
        models[name]=joblib.load(path)
        hashes[name]=sha256(path)
    baseX=process(X,"com",target)
    basepred=baseline.predict(baseX)
    for group,mask in (("all",np.ones(len(y),bool)),("digits",y<10),("letters",y>=10)):
        assert int(np.sum(basepred[mask]==y[mask]))==h["official_validation"][group]["correct"]
    pletter={name:np.empty((len(SCALES),len(y)),np.float32) for name in models}
    preds={name:np.empty((len(SCALES),len(y)),np.uint8) for name in models}
    for k,scale in enumerate(SCALES):
        XX=baseX if scale==1 else process(fixed_scale(X,scale),"com",target)
        for name,m in models.items():
            proba=np.asarray(m.predict_proba(XX),dtype=np.float32)
            assert np.array_equal(np.asarray(m.named_steps["model"].classes_,dtype=int),np.arange(16))
            pletter[name][k]=proba[:,10:].sum(axis=1)
            preds[name][k]=proba.argmax(axis=1)
        print("EXP016_OFFICIALVAL_SCALED",seed,scale,flush=True)
    result=[];pair=[]
    masks={"all":np.ones(len(y),bool),"digits":y<10,"letters":y>=10}
    for i,label in enumerate(LABELS):
        masks["class_"+label]=y==i
    for name in models:
        p=pletter[name];pred=preds[name]
        zero=SCALES.index(1.)
        for k,scale in enumerate(SCALES):
            for group,mask in masks.items():
                now=p[k,mask];original=p[zero,mask]
                result.append({"seed":seed,"model":name,"scale":scale,"group":group,
                   "n":int(mask.sum()),
                   "family_flip_rate":float(np.mean((now>=.5)!=(original>=.5))),
                   "mean_abs_delta_p_letter":float(np.mean(np.abs(now-original))),
                   "mean_signed_delta_p_letter":float(np.mean(now-original)),
                   "accuracy":float(np.mean(pred[k,mask]==y[mask]))})
            for a,b in PAIRS:
                mask=y==LABELS.index(a)
                pair.append({"seed":seed,"model":name,"scale":scale,
                     "actual":a,"predicted":b,
                     "count":int(np.sum(pred[k,mask]==LABELS.index(b)))})
        np.savez_compressed(out/f"officialval_response_{name}.npz",p_letter=p,
             cls_pred=pred,y_true=y.astype(np.uint8),scales=np.array(SCALES,dtype=np.float32))
    save_csv(out/"officialval_scale_sensitivity.csv",result)
    save_csv(out/"officialval_pair_errors.csv",pair)
    manifest={"experiment":"EXP016_OFFICIALVAL_SCALE_AUDIT",
       "seed":seed,"candidates":list(models),"source_models_sha256":hashes,
       "n_official_val":len(y),"original_exp006_com_predictions_verified":True,
       "scale_factors":list(SCALES),"size_target":target,
       "preprocessing":"scale raw OfficialVal image, THEN CoM mass centering",
       "official_data_sha256":official["sha256"],
       "fine_tuning_or_training_performed":False,
       "official_test_used":False,"professor_test_used":False}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("EXP016_OFFICIALVAL_AUDIT_COMPLETE",seed,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",required=True,type=int,choices=(42,43,44))
    p.add_argument("--new-models-root",required=True,type=Path)
    p.add_argument("--old-models-root",required=True,type=Path)
    p.add_argument("--controls-root",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    run(a.seed,a.new_models_root,a.old_models_root,a.controls_root,a.out)
