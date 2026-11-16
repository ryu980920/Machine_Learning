#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP014: class-family-independent size augmentation and stroke controls.
Three new MLPC classifiers per seed, each fits 76,800 official Train source images.
Archived EXP006 CoM/SizeCoM models serve as frozen controls; never use test sets.
"""
from __future__ import annotations
import argparse,csv,gzip,json,time,warnings
from pathlib import Path
import joblib, cv2
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from exp006_size_com_comparison import (
    ROOT,WRITERS,LABELS,PixelScaler255,load_official,load_writers,
    image_geometry,apply_size,process,save_csv,sha256)
from exp011_012_causal_probes import fixed_scale
from stroke_normalization import topology,KERNEL

VARIANTS=("balanced_scale","balanced_scale_stroke","sizecom_balanced_scale")
SCALES=(.8,.9,1.,1.1,1.2)
PAIRS=(("B","8"),("8","B"),("D","0"),("0","D"),("F","5"),("5","F"))

def balanced_indices(y,seed=20261116):
    rng=np.random.default_rng(seed)
    selections=[]; targets=np.zeros(len(y),np.int32)
    for cls in range(16):
        orig=np.flatnonzero(y==cls)
        # Each official source class can have an uneven count in this split.
        # Sample near one-half, rounded DOWN to a multiple of five, so EACH
        # target size is assigned exactly equally within EACH true class.
        n_aug=(len(orig)//10)*5
        if n_aug<5: raise AssertionError("insufficient per-class training data")
        sel=np.sort(rng.choice(orig,size=n_aug,replace=False))
        choices=np.tile(np.array([16,18,20,22,24],dtype=np.int32),
                        len(sel)//5)
        rng.shuffle(choices)
        targets[sel]=choices
        selections.extend(sel.tolist())
    ids=np.array(sorted(selections),dtype=np.int32)
    assert .48 <= len(ids)/len(y) <= .50
    assert set(np.unique(targets[ids]))=={16,18,20,22,24}
    return ids,targets

def stroke_jitter(images,ids):
    """Conservative, topology-checked gray morphological jitter (train only)."""
    a=np.asarray(images,dtype=np.float32).reshape(-1,28,28).copy()
    kept=0;changed=0
    for j,im in enumerate(a):
        # Deterministic parity independent of label and class.
        proposed=cv2.dilate(im,KERNEL) if j%2 else cv2.erode(im,KERNEL)
        mixed=np.clip(.72*im+.28*proposed,0,255)
        if topology(mixed)==topology(im):
            a[j]=mixed
            changed+=1
        else:
            kept+=1
    return a.reshape(-1,784), {"stroke_changed":changed,"stroke_rejected_topology":kept}

def make_training(raw,y,target,variant):
    mode="size_com" if variant=="sizecom_balanced_scale" else "com"
    base=process(raw,mode,target)
    ids,targets=balanced_indices(y)
    augmented=np.empty((len(ids),784),dtype=np.float32)
    for desired in (16,18,20,22,24):
        mask=targets[ids]==desired
        s=ids[mask]
        assert len(s)>0 and len(s)==sum((np.sum(y==cl)//10) for cl in range(16))
        aug,_=apply_size(raw[s],desired)
        augmented[mask]=aug
    stroke_stats={"stroke_changed":0,"stroke_rejected_topology":0}
    if variant=="balanced_scale_stroke":
        augmented,stroke_stats=stroke_jitter(augmented,ids)
    # All evaluation uses the exact same deterministic mode, without augmentation.
    replacement=process(augmented,mode,target)
    base[ids]=replacement
    geom=image_geometry(base)
    sizes=[{"family":family,"n":int(np.sum(mask)),
             "longside_median":float(np.median(geom["long_side"][mask])),
             "longside_mean":float(np.mean(geom["long_side"][mask])),
             "mass_median":float(np.median(geom["mass"][mask])),
             "width_median":float(np.median(geom["width"][mask])),
             "height_median":float(np.median(geom["height"][mask]))}
            for family,mask in (("digits",y<10),("letters",y>=10))]
    return base,mode,stroke_stats,sizes

def group_metrics(y,pred,writers=None):
    masks={"all":np.ones(len(y),bool),"digits":y<10,"letters":y>=10}
    for j,l in enumerate(LABELS): masks["class_"+l]=y==j
    if writers is not None:
        for w in WRITERS:
            masks[f"writer_{w}"]=writers==w
            masks[f"writer_{w}_digits"]=(writers==w)&(y<10)
            masks[f"writer_{w}_letters"]=(writers==w)&(y>=10)
    return [{"group":name,"n":int(mask.sum()),"correct":int((pred[mask]==y[mask]).sum()),
             "accuracy":float(np.mean(pred[mask]==y[mask])),
             "macro_f1":float(f1_score(y[mask],pred[mask],
                 labels=np.unique(y[mask]),average="macro",zero_division=0))}
            for name,mask in masks.items() if mask.any()]

def original_model(path,seed,method,official,hm):
    d=Path(path)/f"normal-exp006-{method}-{seed}"
    manifest=json.loads((d/"run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["method"]==method and manifest["seed"]==seed
    assert manifest["official_sha256"]==official["sha256"]
    assert all(manifest["writer_sha256"][str(w)]==hm[w]["sha256"] for w in WRITERS)
    assert sha256(d/"model.joblib")==manifest["model_sha256"]
    assert not manifest["official_test_used"] and not manifest["professor_test_used"]
    return joblib.load(d/"model.joblib"),manifest

def run(seed,model_root,out):
    out=Path(out)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    t0=time.monotonic()
    official=load_official(ROOT/"02_data"/"datasets"/"MNIST_EMNIST_official_split_112000.npz")
    hm=load_writers(ROOT/"02_data"/"Hand_made")
    Xraw=official["X_train"];ytr=np.asarray(official["y_train"],dtype=int)
    Xval=official["X_val"];yval=np.asarray(official["y_val"],dtype=int)
    Xhm=np.concatenate([hm[w]["X"] for w in WRITERS])
    yhm=np.concatenate([hm[w]["y"] for w in WRITERS]).astype(int)
    writers=np.repeat(WRITERS,640)
    assert Xraw.shape==(76800,784) and Xval.shape==(19200,784) and Xhm.shape==(2560,784)
    target=int(np.median(image_geometry(Xraw)["long_side"]))
    assert target==20
    models={};modes={};manifests={}
    for method in ("com","size_com"):
        m,h=original_model(model_root,seed,method,official,hm)
        name="baseline_"+method
        models[name]=m;modes[name]=method
        # Verify the archived official validation and handmade predictions.
        for dataset,X,y,record in (("OfficialVal",Xval,yval,h["official_validation"]),
                                    ("Handmade",Xhm,yhm,h["handmade"])):
            p=m.predict(process(X,method,target))
            for group,mask in (("all",np.ones(len(y),bool)),("digits",y<10),("letters",y>=10)):
                if int((p[mask]==y[mask]).sum())!=record[group]["correct"]:
                    raise AssertionError("Original EXP006 reproducibility failure: "+name+dataset+group)
        manifests[name]=h["model_sha256"]
    fitted_geometry=[]; fit_records=[]
    for variant in VARIANTS:
        xtr,mode,stroke_stat,geos=make_training(Xraw,ytr,target,variant)
        assert len(xtr)==76800
        estimator=Pipeline([("scale_0_1",PixelScaler255()),
                             ("model",MLPClassifier(random_state=seed))])
        tfit=time.monotonic()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            estimator.fit(xtr,ytr)
        sec=time.monotonic()-tfit
        models[variant]=estimator;modes[variant]=mode
        for v in geos: fitted_geometry.append({"variant":variant,**v})
        fit_records.append({"variant":variant,"seed":seed,"training_rows":len(xtr),
                            "fit_seconds":sec,"iterations":int(estimator.named_steps["model"].n_iter_),
                            "fit_warnings":" | ".join(str(x.message) for x in seen),
                            **stroke_stat})
        joblib.dump(estimator,out/f"{variant}.joblib",compress=3)
        print("EXP014_TRAINED",seed,variant,"seconds",round(sec,1),flush=True)
        del xtr
    save_csv(out/"training_geometry.csv",fitted_geometry)
    save_csv(out/"fit_records.csv",fit_records)
    metrics=[];pairs=[]
    for name,model in models.items():
        mode=modes[name]
        for dataset,X,y,w in (("OfficialVal",Xval,yval,None),("Handmade",Xhm,yhm,writers)):
            pred=np.asarray(model.predict(process(X,mode,target)),dtype=int)
            for gm in group_metrics(y,pred,w):
                metrics.append({"seed":seed,"model":name,"preprocessing":mode,
                                "dataset":dataset,**gm})
            if dataset=="Handmade":
                for a,b in PAIRS:
                    m=y==LABELS.index(a)
                    pairs.append({"seed":seed,"model":name,"actual":a,"predicted":b,
                                  "count":int(np.count_nonzero(pred[m]==LABELS.index(b)))})
    save_csv(out/"accuracy_metrics.csv",metrics)
    save_csv(out/"pair_errors.csv",pairs)
    scale_rows=[]; image_rows=[]
    for scale in SCALES:
        scaled=fixed_scale(Xhm,scale)
        features={mode:process(scaled,mode,target) for mode in ("com","size_com")}
        for name,model in models.items():
            mode=modes[name]
            proba=np.asarray(model.predict_proba(features[mode]),dtype=np.float64)
            labels=np.asarray(model.named_steps["model"].classes_,dtype=int)
            assert np.array_equal(labels,np.arange(16))
            pred=proba.argmax(axis=1)
            p_letter=proba[:,10:].sum(axis=1)
            for gm in group_metrics(yhm,pred,writers):
                scale_rows.append({"seed":seed,"model":name,"scale":scale,**gm,
                    "mean_p_letter":float(p_letter.mean())})
            for i in range(len(yhm)):
                image_rows.append({"seed":seed,"model":name,"scale":scale,
                    "writer":int(writers[i]),"index":int(i%640),
                    "true":LABELS[int(yhm[i])],"pred":LABELS[int(pred[i])],
                    "p_letter":float(p_letter[i])})
        print("EXP014_SCALED",seed,scale,flush=True)
    save_csv(out/"scale_metrics.csv",scale_rows)
    save_csv(out/"scale_image_family_probabilities.csv",image_rows)
    man={"experiment":"EXP014","seed":seed,"variants":list(VARIANTS),
         "scale_interventions":list(SCALES),"augmentation_fraction":.5,
         "augmented_targets_longside":[16,18,20,22,24],
         "balanced_targets_per_class":True,"training_rows_per_variant":76800,
         "model_hyperparameters":"sklearn MLPClassifier defaults, seed controlled",
         "archived_control_model_sha256":manifests,
         "original_checkpoint_hash_and_predictions_verified":True,
         "official_sha256":official["sha256"],
         "writer_sha256":{str(w):hm[w]["sha256"] for w in WRITERS},
         "original_test_used":False,"professor_test_used":False,
         "handmade_used_in_fit":False,"validation_used_in_fit":False,
         "training_time_seconds":time.monotonic()-t0}
    (out/"manifest.json").write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding="utf-8")
    print("EXP014_COMPLETE",seed,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True,choices=(42,43,44))
    p.add_argument("--models-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.seed,a.models_root,a.out)
