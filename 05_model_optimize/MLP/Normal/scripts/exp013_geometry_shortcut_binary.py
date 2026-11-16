#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP013: Can geometry alone predict digit-vs-letter labels?
No original/professor Test. Fits ONLY logistic/tree binary probes, no MLP retraining.
Original EXP006/010 pixel processing and the existing official split are reused.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, confusion_matrix
from exp006_size_com_comparison import ROOT, WRITERS, LABELS, load_official, load_writers, image_geometry, process, save_csv

METHODS=("raw","com","size","size_com")
FEATURES=("long_side","width","height","mass")
GROUPS={
 "size_only":("long_side",),
 "width_height":("width","height"),
 "ink_mass_only":("mass",),
 "all_four":FEATURES,
 "without_mass":("long_side","width","height"),
 "without_size":("width","height","mass"),
}
def extract(x,method,target,batch=1024):
    result=np.empty((len(x),len(FEATURES)),dtype=np.float64)
    for i in range(0,len(x),batch):
        v=np.asarray(x[i:i+batch],dtype=np.float32).reshape(-1,784)
        if method!="raw":
            v=process(v,method,target)
        geo=image_geometry(v)
        for k,name in enumerate(FEATURES):
            result[i:i+len(v),k]=geo[name]
    if not np.isfinite(result).all(): raise AssertionError("non-finite geometry")
    return result
def classifier(kind):
    if kind=="logreg":
        return make_pipeline(StandardScaler(),
            LogisticRegression(class_weight="balanced",max_iter=1500,random_state=42))
    if kind=="tree":
        return DecisionTreeClassifier(max_depth=3,min_samples_leaf=40,
                                       class_weight="balanced",random_state=42)
    raise ValueError(kind)
def metrics(y,prob,threshold=0.5):
    pred=(prob>=threshold).astype(int)
    mat=confusion_matrix(y,pred,labels=[0,1])
    return {"n":int(len(y)),"accuracy":float(accuracy_score(y,pred)),
      "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
      "roc_auc":float(roc_auc_score(y,prob)),
      "digits_recall":float(mat[0,0]/mat[0].sum()),
      "letters_recall":float(mat[1,1]/mat[1].sum()),
      "digit_as_letter":int(mat[0,1]),"letter_as_digit":int(mat[1,0])}
def run(out):
    out=Path(out)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    official=load_official(ROOT/"02_data"/"datasets"/"MNIST_EMNIST_official_split_112000.npz")
    hm=load_writers(ROOT/"02_data"/"Hand_made")
    assert official["X_train"].shape==(76800,784) and official["X_val"].shape==(19200,784)
    X_h=np.concatenate([hm[w]["X"] for w in WRITERS])
    y_h=np.concatenate([hm[w]["y"] for w in WRITERS]).astype(int)
    w_h=np.repeat(WRITERS,640)
    assert len(y_h)==2560
    target=int(np.median(image_geometry(official["X_train"])["long_side"]))
    assert target==20
    y_tr=(np.asarray(official["y_train"])>=10).astype(int)
    y_va=(np.asarray(official["y_val"])>=10).astype(int)
    y_h=(y_h>=10).astype(int)
    rows=[];dists=[];examples=[]
    start=time.monotonic()
    for method in METHODS:
        train=extract(official["X_train"],method,target)
        valid=extract(official["X_val"],method,target)
        hand=extract(X_h,method,target)
        for k,f in enumerate(FEATURES):
            for family in (0,1):
                v=train[y_tr==family,k]
                dists.append({"preprocessing":method,"feature":f,
                  "family":"digits" if family==0 else "letters",
                  "n":len(v),"mean":float(np.mean(v)),"median":float(np.median(v)),
                  "p10":float(np.quantile(v,0.1)),"p90":float(np.quantile(v,0.9))})
        for name,features in GROUPS.items():
            idx=[FEATURES.index(f) for f in features]
            for kind in ("logreg","tree"):
                probe=classifier(kind)
                probe.fit(train[:,idx],y_tr)
                for dataset,X,y,w in (("OfficialVal",valid,y_va,None),
                                      ("Handmade",hand,y_h,w_h)):
                    prob=np.asarray(probe.predict_proba(X[:,idx])[:,1])
                    base={"preprocessing":method,"features":name,
                          "classifier":kind,"dataset":dataset}
                    rows.append({**base,"group":"all",**metrics(y,prob)})
                    if w is not None:
                        for writer in WRITERS:
                            m=w==writer
                            rows.append({**base,"group":f"writer_{writer}",
                                         **metrics(y[m],prob[m])})
                if name=="all_four" and kind=="logreg":
                    coefficients=probe.named_steps["logisticregression"].coef_.ravel()
                    for f,coef in zip(features,coefficients):
                        examples.append({"method":method,"feature":f,
                          "standardized_logistic_coef":float(coef)})
        print("EXP013",method,"processed",len(train),len(valid),len(hand),
              "seconds",round(time.monotonic()-start,1),flush=True)
    save_csv(out/"binary_probe_metrics.csv",rows)
    save_csv(out/"class_family_feature_distributions.csv",dists)
    save_csv(out/"standardized_coefficients.csv",examples)
    manifest={"experiment":"EXP013","target_longside_px":target,"foreground_threshold":32,
      "feature_names":list(FEATURES),"feature_sets":{k:list(v) for k,v in GROUPS.items()},
      "binary_targets":{"digits_0_9":0,"letters_A_F":1},"classifier_kinds":["logreg","tree"],
      "model_training":"binary probes only; original MLP unmodified",
      "n_official_train":76800,"n_official_val":19200,"n_handmade":2560,
      "official_sha256":official["sha256"],
      "writers_sha256":{str(w):hm[w]["sha256"] for w in WRITERS},
      "handmade_used_in_fit":False,"validation_used_in_fit":False,
      "original_test_used":False,"professor_test_used":False}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("EXP013_COMPLETE",len(rows),len(dists),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args().out)
