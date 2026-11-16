#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP유010 — Official Training vs Hand-made geometry, Original/Size/Size->CoM.

Reuses EXACT EXP유006 apply_size and center_batch; all source data are developer
train/validation/handmade only. No MLP fitting or test access.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance

from exp006_size_com_comparison import (
    ROOT, WRITERS, LABELS, load_official, load_writers,
    image_geometry, process, save_csv
)
from Automatic_Centering_Transformer import center_shifts

FEATURES=("width","height","long_side","cx","cy","com_offset",
          "bbox_cx","bbox_cy","ink_ratio","mass","edge_mass")
METHODS=("raw","size","size_com")

def get_processed(raw,method,target):
    if method=="raw":
        return np.asarray(raw,dtype=np.float32).reshape(-1,784)
    return process(raw,method,target)

def extract(raw,method,target,batch_size=1024):
    n=len(raw)
    full={k:np.empty(n,dtype=np.float64) for k in FEATURES}
    full["scale"]=np.ones(n,dtype=np.float64)
    full["shift_x"]=np.zeros(n,dtype=np.float64)
    full["shift_y"]=np.zeros(n,dtype=np.float64)
    for start in range(0,n,batch_size):
        end=min(n,start+batch_size)
        v=np.asarray(raw[start:end],dtype=np.float32).reshape(-1,784)
        if method=="size_com":
            presize=process(v,"size",target)
            shift=center_shifts(presize,"mass")
            full["shift_x"][start:end]=shift[:,0]
            full["shift_y"][start:end]=shift[:,1]
            processed=process(v,"size_com",target)
        elif method=="size":
            processed=process(v,"size",target)
        else:
            processed=v
        before=image_geometry(v)
        geo=image_geometry(processed)
        if method!="raw":
            full["scale"][start:end]=float(target)/before["long_side"]
        for key in FEATURES:
            full[key][start:end]=geo[key]
        if start%16384==0:
            print(f"GEOMETRY {method} {start}:{end}/{n}",flush=True)
    return full

def summarize(dataset,method,labels,writers,g):
    groups={"all":np.ones(len(labels),dtype=bool),
            "digits":labels<10,"letters":labels>=10}
    for i,cl in enumerate(LABELS):
        groups["class_"+cl]=labels==i
    if writers is not None:
        for w in WRITERS:
            groups["writer_"+str(w)]=writers==w
            groups["writer_"+str(w)+"_digits"]=(writers==w)&(labels<10)
            groups["writer_"+str(w)+"_letters"]=(writers==w)&(labels>=10)
    result=[]
    for group,mask in groups.items():
        if not mask.any():continue
        for k in (*FEATURES,"scale","shift_x","shift_y"):
            x=g[k][mask]
            result.append({"dataset":dataset,"preprocessing":method,"group":group,
                "feature":k,"n":len(x),"mean":float(x.mean()),
                "median":float(np.median(x)),
                "std":float(x.std()),
                "p10":float(np.quantile(x,.1)),
                "p90":float(np.quantile(x,.9))})
    return result

def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    official=load_official(ROOT/"02_data"/"datasets"/"MNIST_EMNIST_official_split_112000.npz")
    handmade=load_writers(ROOT/"02_data"/"Hand_made")
    assert official["X_train"].shape==(76800,784)
    assert official["X_val"].shape==(19200,784)
    target=int(np.median(image_geometry(official["X_train"])["long_side"]))
    assert target==20,"EXP006 target mismatch"
    x_hm=np.concatenate([handmade[w]["X"] for w in WRITERS],axis=0)
    y_hm=np.concatenate([handmade[w]["y"] for w in WRITERS],axis=0)
    writer_ids=np.repeat(WRITERS,640)
    sources={
      "OfficialTrain":(official["X_train"],official["y_train"],None),
      "OfficialVal":(official["X_val"],official["y_val"],None),
      "Handmade":(x_hm,y_hm,writer_ids)
    }
    all_geo={}
    summary=[]
    per_handmade=[]
    for dataset,(X,y,wids) in sources.items():
        for method in METHODS:
            g=extract(X,method,target)
            all_geo[dataset,method]=g
            summary+=summarize(dataset,method,y,wids,g)
            if dataset=="Handmade":
                for i in range(len(X)):
                    per_handmade.append({
                        "preprocessing":method,
                        "writer":int(wids[i]),
                        "index":int(i%640),
                        "label":LABELS[int(y[i])],
                        **{k:float(g[k][i]) for k in (*FEATURES,"scale","shift_x","shift_y")}
                    })
    save_csv(out/"geometry_summary.csv",summary)
    save_csv(out/"handmade_geometry_per_image.csv",per_handmade)
    distances=[]
    for method in METHODS:
        for class_id,cl in enumerate(LABELS):
            for segment,mask_train,mask_hm,mask_val in [
                ("all",np.ones(76800,dtype=bool),
                 np.ones(2560,dtype=bool),np.ones(19200,dtype=bool)),
                ("class_"+cl,official["y_train"]==class_id,
                 y_hm==class_id,official["y_val"]==class_id)
            ]:
                # Evaluate all once; individual classes 16x, all only once
                if segment=="all" and class_id>0:continue
                train=all_geo["OfficialTrain",method]
                val=all_geo["OfficialVal",method]
                hm=all_geo["Handmade",method]
                for feature in ("width","height","long_side","cx","cy",
                                "com_offset","ink_ratio","mass"):
                    distances.append({"preprocessing":method,"segment":segment,
                        "feature":feature,
                        "n_train":int(mask_train.sum()),
                        "n_val":int(mask_val.sum()),
                        "n_handmade":int(mask_hm.sum()),
                        "train_vs_handmade_w1":float(wasserstein_distance(
                            train[feature][mask_train],hm[feature][mask_hm])),
                        "train_vs_val_w1":float(wasserstein_distance(
                            train[feature][mask_train],val[feature][mask_val]))})
    save_csv(out/"geometry_distance.csv",distances)
    manifest={"experiment":"Normal-EXP010","preprocessing_methods":list(METHODS),
      "code":"original exp006 image_geometry, process + center_shifts",
      "target_bbox_longside":target,"foreground_threshold":32,
      "official_train_n":76800,"official_validation_n":19200,
      "handmade_n":2560,"writers":list(WRITERS),"labels":LABELS,
      "official_sha256":official["sha256"],
      "writer_sha256":{str(w):handmade[w]["sha256"] for w in WRITERS},
      "official_original_test_used":False,"professor_test_used":False,
      "handmade_training_used":False,"model_training_used":False}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("EXP010_GEOMETRY_COMPLETE",flush=True)
    for cl in ("B","D","8","0"):
        for method in METHODS:
            g_train=all_geo["OfficialTrain",method]
            g_hm=all_geo["Handmade",method]
            clidx=LABELS.index(cl)
            print("EXAMPLE",cl,method,
                  "train_long_p50",np.median(g_train["long_side"][official["y_train"]==clidx]),
                  "handmade_long_p50",np.median(g_hm["long_side"][y_hm==clidx]),
                  flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    run(a.out)
