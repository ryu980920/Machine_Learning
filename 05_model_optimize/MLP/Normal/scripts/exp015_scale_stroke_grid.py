#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP015: predeclared size-fraction x stroke-strength MLP training optimization.

All MLPs see EXACTLY 76,800 unique original Official Train source rows.
Handmade (2,560) & OfficialVal (19,200) are NEVER used in fitting or fitting augmentation.
No original Test or professor Test data access. Archived EXP006 control hash and predictions
must match before results are accepted. No mutations to final pipeline or main.
"""
from __future__ import annotations
import argparse, csv, json, time, warnings
from pathlib import Path
import cv2
import joblib
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.neural_network import MLPClassifier

from exp006_size_com_comparison import (
    ROOT, WRITERS, LABELS, PixelScaler255, load_official, load_writers,
    image_geometry, apply_size, process, save_csv, sha256)
from exp011_012_causal_probes import fixed_scale
from exp014_balanced_augmentation import balanced_indices, original_model, group_metrics, PAIRS
from stroke_normalization import topology, KERNEL

SCALES=(.8,.9,1.0,1.1,1.2)
TARGETS=(16,18,20,22,24)
STROKES=(0.,.14,.28)  # same gray-level morphological blend as EXP014, varied strength
FRACTIONS=(.1,.2,.3,.4,.5)
SEEDS=(42,43,44)
AUG_SEED=20261116

def nested_subset(y,ratio):
    """Subsets of EXACT historic EXP014 50% IDs for controlled paired ratios.

    For each true class and each target size the chosen original source IDs
    are nested across fractions, sampled with reproducible independent RNG.
    At ratio .5 the IDs/size assignment are precisely EXP014 balanced_indices.
    """
    assert ratio in FRACTIONS
    ids50, target50=balanced_indices(y,seed=AUG_SEED)
    if ratio==.5: return ids50,target50
    chosen=[]
    for klass in range(16):
        for desired in TARGETS:
            source=ids50[(y[ids50]==klass)&(target50[ids50]==desired)]
            n=int(np.rint((ratio/.5)*len(source)))
            if not 0<n<=len(source): raise AssertionError((klass,desired,ratio))
            rng=np.random.default_rng(AUG_SEED+97*klass+desired*13)
            order=rng.permutation(len(source))
            # All fractions share the same order; smaller subsets nested in bigger.
            chosen.extend(source[order[:n]].tolist())
    ids=np.asarray(sorted(chosen),dtype=np.int32)
    assert len(np.unique(ids))==len(ids)
    assert abs(len(ids)/len(y)-ratio)<.003
    return ids,target50

def scale_base(raw,ids,targets):
    """Only original source bytes determine the augmentation; class independent size assignment."""
    resized=np.empty((len(ids),784),dtype=np.float32)
    for desired in TARGETS:
        at=np.flatnonzero(targets[ids]==desired)
        assert len(at)>0
        out,_=apply_size(raw[ids[at]],desired)
        resized[at]=out
    return resized

def stroke_jitter(images,strength):
    """Use identical morphological transform as EXP014, optional strength=0."""
    x=np.asarray(images,dtype=np.float32).reshape(-1,28,28).copy()
    if strength==0: return x.reshape(-1,784),{"modified":0,"rejected":0}
    changed=0;rejected=0
    for i,img in enumerate(x):
        target=cv2.erode(img,KERNEL) if (i%2)==0 else cv2.dilate(img,KERNEL)
        changed_img=np.clip((1-strength)*img+strength*target,0,255)
        if topology(changed_img)==topology(img):
            x[i]=changed_img
            changed+=1
        else:
            rejected+=1
    return x.reshape(-1,784),{"modified":changed,"rejected":rejected}

def preview_data(x):
    geo=image_geometry(x)
    return {k:np.asarray(geo[k]) for k in ("long_side","width","height","mass")}

def feature_probe(xtrain,ytrain,xval,yval):
    """Auxiliary diagnostic only: whether available size-only label signal remains."""
    train=preview_data(xtrain)["long_side"].reshape(-1,1)
    val=preview_data(xval)["long_side"].reshape(-1,1)
    yy=(ytrain>=10).astype(int)
    yv=(yval>=10).astype(int)
    m=make_pipeline(StandardScaler(),LogisticRegression(
        class_weight="balanced",random_state=42,max_iter=750))
    m.fit(train,yy)
    proba=m.predict_proba(val)[:,1]
    return {"size_only_val_bal_acc":float(balanced_accuracy_score(yv,proba>=.5)),
            "size_only_val_roc_auc":float(roc_auc_score(yv,proba))}

def summary_geometry(x,y):
    geo=preview_data(x)
    return [{"family":f,"count":int(m.sum()),
        **{k+"_mean":float(np.mean(v[m])) for k,v in geo.items()},
        **{k+"_median":float(np.median(v[m])) for k,v in geo.items()}}
        for f,m in (("digits",y<10),("letters",y>=10))]

def evaluate(model,seed,name,ratio,strength,xval,yval,xhm,yhm,writers,scaled, out):
    metrics=[];sensitivity=[];pair_errors=[]
    for ds,x,y,w in (("OfficialVal",xval,yval,None),("Handmade",xhm,yhm,writers)):
        prediction=np.asarray(model.predict(x),dtype=int)
        for r in group_metrics(y,prediction,w):
            metrics.append({"seed":seed,"variant":name,"ratio":ratio,"stroke_strength":strength,
                            "dataset":ds,**r})
        if ds=="Handmade":
            for a,b in PAIRS:
                m=y==LABELS.index(a)
                pair_errors.append({"seed":seed,"variant":name,"ratio":ratio,
                    "stroke_strength":strength,"actual":a,"predicted":b,
                    "count":int(np.sum(prediction[m]==LABELS.index(b)))})
    p=np.empty((len(SCALES),len(yhm)),dtype=np.float32)
    pred=np.empty((len(SCALES),len(yhm)),dtype=np.uint8)
    for k,scale in enumerate(SCALES):
        values=np.asarray(model.predict_proba(scaled[scale]),dtype=np.float32)
        labels=np.asarray(model.named_steps["model"].classes_,dtype=int)
        assert np.array_equal(labels,np.arange(16))
        p[k]=values[:,10:].sum(axis=1)
        pred[k]=values.argmax(axis=1)
    base_index=SCALES.index(1.0)
    for k,scale in enumerate(SCALES):
        for group,mask in (("digits",yhm<10),("letters",yhm>=10),
                           ("all",np.ones(len(yhm),bool)),
                           *[(f"writer_{w}_{family}",(writers==w)&mask)
                             for w in WRITERS for family,mask in
                             (("digits",yhm<10),("letters",yhm>=10))]):
            current=p[k,mask]
            base=p[base_index,mask]
            family_flip=(current>=.5)!=(base>=.5)
            accuracy=(pred[k,mask]==yhm[mask]).mean()
            sensitivity.append({"seed":seed,"variant":name,"ratio":ratio,
               "stroke_strength":strength,"scale":scale,"group":group,
               "n":int(mask.sum()),"family_flip_rate":float(np.mean(family_flip)),
               "mean_abs_delta_p_letter":float(np.mean(np.abs(current-base))),
               "mean_signed_delta_p_letter":float(np.mean(current-base)),
               "mean_p_letter":float(current.mean()),"accuracy":float(accuracy)})
    np.savez_compressed(out/f"response_{name}.npz",p_letter=p,cls_pred=pred,
        y_true=yhm.astype(np.uint8),writer=writers.astype(np.uint8),
        scales=np.asarray(SCALES,dtype=np.float32))
    return metrics,sensitivity,pair_errors

def run(seed,ratio,out,models_root):
    if seed not in SEEDS or ratio not in FRACTIONS: raise ValueError((seed,ratio))
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    t0=time.monotonic()
    o=load_official(ROOT/"02_data"/"datasets"/"MNIST_EMNIST_official_split_112000.npz")
    hm=load_writers(ROOT/"02_data"/"Hand_made")
    Xraw=o["X_train"];ytrain=np.asarray(o["y_train"],dtype=int)
    Xval0=o["X_val"];yval=np.asarray(o["y_val"],dtype=int)
    Xhm0=np.concatenate([hm[w]["X"] for w in WRITERS])
    yhm=np.concatenate([hm[w]["y"] for w in WRITERS]).astype(int)
    writers=np.repeat(np.array(WRITERS),640)
    assert Xraw.shape==(76800,784) and Xval0.shape==(19200,784) and Xhm0.shape==(2560,784)
    target=int(np.median(image_geometry(Xraw)["long_side"]))
    assert target==20
    control,stored=original_model(models_root,seed,"com",o,hm)
    xval=process(Xval0,"com",target)
    xhm=process(Xhm0,"com",target)
    for X,y,which in ((xval,yval,"official_validation"),(xhm,yhm,"handmade")):
        cpred=np.asarray(control.predict(X),dtype=int)
        for group,mask in (("all",np.ones(len(y),bool)),("digits",y<10),("letters",y>=10)):
            assert int(np.sum(cpred[mask]==y[mask]))==stored[which][group]["correct"],(seed,which,group)
    scaled={s:process(fixed_scale(Xhm0,s),"com",target) for s in SCALES}
    base=process(Xraw,"com",target)
    ids,target_map=nested_subset(ytrain,ratio)
    resized=scale_base(Xraw,ids,target_map)
    assert len(base)==76800 and len(resized)==len(ids)
    expected_per_class_target={}
    for klass in range(16):
        counts=[int(np.sum((ytrain[ids]==klass)&(target_map[ids]==size))) for size in TARGETS]
        assert len(set(counts))==1,(klass,counts)
        expected_per_class_target[LABELS[klass]]=counts[0]
    print("EXP015_SETUP",seed,ratio,"n_aug",len(ids),"target counts",expected_per_class_target,flush=True)
    metrics=[];sensitivity=[];pairs=[];trains=[];geometry=[]
    if ratio==.1:   # archived CoM reference (duplicate if other ratios, skip)
        m,s,p=evaluate(control,seed,"baseline_com",0.,0.,xval,yval,xhm,yhm,writers,scaled,out)
        metrics+=m;sensitivity+=s;pairs+=p
    for stroke in STROKES:
        name=f"size{int(round(ratio*100)):02d}_stroke{int(round(stroke*100)):02d}"
        begin=time.monotonic()
        changed,stats=stroke_jitter(resized,stroke)
        xtrain=base.copy()
        xtrain[ids]=process(changed,"com",target)
        geom=summary_geometry(xtrain,ytrain)
        for g in geom:geometry.append({"seed":seed,"variant":name,"ratio":ratio,
            "stroke_strength":stroke,**g})
        probe=feature_probe(xtrain,ytrain,xval,yval)
        m=Pipeline([("scale_0_1",PixelScaler255()),
                    ("model",MLPClassifier(random_state=seed))])
        with warnings.catch_warnings(record=True) as notes:
            warnings.simplefilter("always")
            m.fit(xtrain,ytrain)
        fitted=time.monotonic()-begin
        joblib.dump(m,out/f"{name}.joblib",compress=3)
        tr={"seed":seed,"variant":name,"ratio":ratio,
             "actual_fraction":float(len(ids)/len(ytrain)),
             "stroke_strength":stroke,"n_train":int(len(ytrain)),
             "n_augment":int(len(ids)),"n_stroke_changed":stats["modified"],
             "n_stroke_rejected":stats["rejected"],
             "fit_time_seconds":fitted,
             "fit_iterations":int(m.named_steps["model"].n_iter_),
             "fit_warnings":" | ".join(str(x.message) for x in notes),
             **probe}
        trains.append(tr)
        a,b,c=evaluate(m,seed,name,ratio,stroke,xval,yval,xhm,yhm,writers,scaled,out)
        metrics+=a;sensitivity+=b;pairs+=c
        print("EXP015_MODEL",seed,name,"seconds",round(fitted,2),"eval",
              [(q["dataset"],q["group"],round(q["accuracy"],5)) for q in a
               if q["group"] in ("all","digits","letters")],flush=True)
        del changed,xtrain,m
    save_csv(out/"training_records.csv",trains)
    save_csv(out/"training_geometry.csv",geometry)
    save_csv(out/"accuracy_metrics.csv",metrics)
    save_csv(out/"scale_sensitivity.csv",sensitivity)
    save_csv(out/"pair_errors.csv",pairs)
    manifest={"experiment":"EXP015","seed":seed,"ratio_requested":ratio,
      "ratio_actual":len(ids)/len(ytrain),"stroke_strengths":list(STROKES),
      "scales":list(SCALES),"targets":list(TARGETS),
      "balanced_per_class_per_target":True,"nested_sampling_from_exp014_half":True,
      "official_train_n":76800,"official_val_n":19200,"handmade_n":2560,
      "official_source_sha256":o["sha256"],
      "handmade_source_sha256":{str(w):hm[w]["sha256"] for w in WRITERS},
      "archived_exp006_com_model_sha256":stored["model_sha256"],
      "original_predictions_verified":True,
      "actual_training_rows_per_variant":76800,"official_test_used":False,
      "professor_test_used":False,"handmade_used_in_fit":False,
      "official_val_used_in_fit":False,
      "total_run_seconds":time.monotonic()-t0}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("EXP015_COMPLETE",seed,ratio,round(time.monotonic()-t0,2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True,choices=SEEDS)
    p.add_argument("--ratio-percent",type=int,required=True,choices=(10,20,30,40,50))
    p.add_argument("--models-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.seed,a.ratio_percent/100,a.out,a.models_root)
