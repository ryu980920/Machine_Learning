#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP017 stage 1: 80-14 fixed preprocessing. Tune MLP only using Official Train/Val.

Team handwriting, Original Test and professor Test are NEVER read. Generate exact
EXP016 80-14 deterministic augmented images, independently of MLP seed.
"""
from __future__ import annotations
import argparse, warnings, time, json, csv
from pathlib import Path
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from exp006_size_com_comparison import ROOT, LABELS, PixelScaler255, load_official, process, image_geometry
from exp011_012_causal_probes import fixed_scale
from exp016_high_fraction_grid import nested_subset, scale_base, stroke_jitter
from exp014_balanced_augmentation import balanced_indices

SEEDS=(42,43,44)
SCALES=(.8,1.,1.2)
HIDDEN=((100,),(128,),(192,),(256,),(128,64),(256,128))
MORE=((128,),(192,),(128,64),(256,128))
CONF=(
    [(h,a,.001) for h in HIDDEN for a in (.0001,.001,.01)]
    +[(h,.001,lr) for h in MORE for lr in (.0005,.002)]
)
assert len(CONF)==26
def config_name(h,a,lr):
    return "h"+"x".join(map(str,h))+"_a"+f"{a:.4g}".replace(".","p").replace("-","m")+"_lr"+f"{lr:.4g}".replace(".","p").replace("-","m")
def load_data():
    official=load_official(ROOT/"02_data"/"datasets"/"MNIST_EMNIST_official_split_112000.npz")
    Xraw=official["X_train"]
    Y=np.asarray(official["y_train"],dtype=np.int32)
    Xv=official["X_val"]
    Yv=np.asarray(official["y_val"],dtype=np.int32)
    assert Xraw.shape==(76800,784) and Xv.shape==(19200,784)
    assert set(np.unique(Y))==set(range(16)) and set(np.unique(Yv))==set(range(16))
    target=int(np.median(image_geometry(Xraw)["long_side"]))
    assert target==20
    train=process(Xraw,"com",target)
    ids,targets=nested_subset(Y,.8)
    ids_half,_=balanced_indices(Y,seed=20261116)
    resized=scale_base(Xraw,ids,targets)
    changed,stats=stroke_jitter(resized,.14,ids,ids_half)
    train[ids]=process(changed,"com",target)
    assert len(ids)==61440 and train.shape==(76800,784),len(ids)
    val={s:process(Xv if s==1. else fixed_scale(Xv,s),"com",target) for s in SCALES}
    assert all(v.shape==(19200,784) for v in val.values())
    return train,Y,val,Yv,official["sha256"],len(ids),stats

def measure(model, val, truth, seed, name, config):
    pred={}; proba={}
    for scale in SCALES:
        probabilities=np.asarray(model.predict_proba(val[scale]),dtype=np.float32)
        assert probabilities.shape==(len(truth),16)
        pred[scale]=probabilities.argmax(axis=1).astype(np.uint8)
        proba[scale]=probabilities[:,10:].sum(axis=1)
    base_family=proba[1.]>=.5
    rows=[];cls=[]
    for scale in SCALES:
        for label,mask in [("all",np.ones(len(truth),dtype=bool)),("digits",truth<10),("letters",truth>=10)]+[
            ("class_"+sym,truth==j) for j,sym in enumerate(LABELS)]:
            x=proba[scale][mask];base=proba[1.][mask]
            rows.append({"seed":seed,"variant":name,"hidden":str(config[0]),"alpha":config[1],
                "lr":config[2],"scale":scale,"group":label,"n":int(mask.sum()),
                "accuracy":float(np.mean(pred[scale][mask]==truth[mask])),
                "family_flip":float(np.mean((x>=.5)!=base_family[mask])),
                "abs_delta_p_letter":float(np.mean(np.abs(x-base))),
                "signed_delta_p_letter":float(np.mean(x-base))})
    return rows,{"p_letter":np.stack([proba[s] for s in SCALES]).astype(np.float32),
                 "pred":np.stack([pred[s] for s in SCALES]).astype(np.uint8),
                 "labels":truth.astype(np.uint8),"scales":np.array(SCALES,dtype=np.float32)}
def save(path,rows):
    assert rows
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(seed,part,out):
    assert seed in SEEDS and part in (0,1,2)
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    started=time.monotonic()
    X,Y,val,yv,digest,naug,stroke_stats=load_data()
    conf=[(i,*c) for i,c in enumerate(CONF) if i%3==part]
    measurements=[];records=[]
    for index,h,a,lr in conf:
        name=config_name(h,a,lr)
        begin=time.monotonic()
        clf=Pipeline([("scale_0_1",PixelScaler255()),
            ("model",MLPClassifier(hidden_layer_sizes=h,alpha=a,learning_rate_init=lr,
                activation="relu",solver="adam",max_iter=200,early_stopping=False,
                random_state=seed))])
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            clf.fit(X,Y)
        path=out/(name+".joblib")
        joblib.dump(clf,path,compress=3)
        rows,detail=measure(clf,val,yv,seed,name,(h,a,lr))
        measurements+=rows
        np.savez_compressed(out/(name+"_officialval_response.npz"),**detail)
        base=next(r for r in rows if r["group"]=="all" and r["scale"]==1.)
        records.append({"seed":seed,"index":index,"variant":name,
          "hidden":str(h),"alpha":a,"learning_rate_init":lr,
          "activation":"relu","solver":"adam","max_iter":200,"early_stopping":False,
          "n_iter":clf.named_steps["model"].n_iter_,
          "official_val_accuracy":base["accuracy"],
          "fit_seconds":time.monotonic()-begin,
          "warning":" | ".join(str(w.message) for w in ws),
          "train_n":len(Y),"augmented_n":naug,
          "stroke_modified":stroke_stats["modified"],"stroke_rejected":stroke_stats["rejected"],
          "official_source_sha256":digest})
        print("EXP017_COARSE",seed,part,index,name,"acc",round(base["accuracy"],5),
              "iters",clf.named_steps["model"].n_iter_,flush=True)
        del clf
    save(out/"metrics.csv",measurements)
    save(out/"training_records.csv",records)
    manifest={"experiment":"EXP017","stage":"coarse","seed":seed,"part":part,
       "config_count":len(conf),"configs":[r["variant"] for r in records],
       "training_rows":76800,"official_val_rows":19200,"augmented_rows":naug,
       "original_official_sha256":digest,"size_aug_ratio":.8,
       "stroke_blend":.14,"preprocessing":"size_stroke_com_train__com_val",
       "team_handwriting_loaded":False,"handmade_used":False,
       "original_test_used":False,"professor_test_used":False,
       "runtime_seconds":time.monotonic()-started}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("EXP017_COARSE_COMPLETE",seed,part,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True,choices=SEEDS)
    p.add_argument("--part",type=int,required=True,choices=(0,1,2))
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.seed,a.part,a.out)
