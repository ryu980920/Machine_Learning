#!/usr/bin/env python3
"""EXP025: batch-size-only MLP trial. No Handmade or Test access at training stage."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse, hashlib, json, time, warnings
from pathlib import Path
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from exp017_hparam_coarse import load_data,measure,save,PixelScaler255

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",required=True,type=int,choices=(42,43,44))
    p.add_argument("--batch",required=True,type=int,choices=(32,64,128,256))
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    out=a.out
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    start=time.monotonic()
    X,y,val,yv,source_sha,naug,stroke=load_data()
    assert len(y)==76800 and naug==61440 and source_sha=="0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879"
    assert stroke["modified"]==57642 and stroke["rejected"]==3798,stroke
    clf=Pipeline([("scale_0_1",PixelScaler255()),("model",MLPClassifier(
        hidden_layer_sizes=(256,128),activation="relu",solver="adam",
        alpha=.03,learning_rate_init=.001,batch_size=a.batch,
        max_iter=200,early_stopping=False,random_state=a.seed,verbose=True))])
    started_fit=time.monotonic()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        clf.fit(X,y)
    fit_seconds=time.monotonic()-started_fit
    ck=out/"model.joblib"
    joblib.dump(clf,ck,compress=3)
    official_rows,response=measure(clf,val,yv,a.seed,f"batch_{a.batch}",((256,128),.03,.001))
    save(out/"official_metrics.csv",official_rows)
    np.savez_compressed(out/"official_response.npz",**response)
    save(out/"training_curve.csv",[{"epoch":i+1,"training_loss":v} for i,v in enumerate(clf.named_steps["model"].loss_curve_)])
    rec={"experiment":"EXP025","seed":a.seed,"batch_size":a.batch,"n_iter":int(clf.named_steps["model"].n_iter_),
      "fit_seconds":fit_seconds,"elapsed_seconds":time.monotonic()-start,
      "train_n":len(y),"officialval_n":len(yv),"augmented_n":naug,
      "train_preprocessing":"EXP017 Size80% Stroke14% CoM","source_sha256":source_sha,"stroke":stroke,
      "params":{"hidden_layer_sizes":[256,128],"alpha":.03,"lr":.001,"solver":"adam","activation":"relu",
       "batch_size":a.batch,"max_iter":200,"early_stopping":False,"random_state":a.seed},
      "model_sha256":sha(ck),"warnings":[str(w.message) for w in captured],
      "handmade_used_in_training_or_selection":False,"original_test_used":False,"professor_test_used":False}
    (out/"manifest.json").write_text(json.dumps(rec,indent=2,ensure_ascii=False))
    print("EXP025_BATCH_COMPLETE",a.batch,a.seed,"OfficialVal",next(r["accuracy"] for r in official_rows if r["scale"]==1. and r["group"]=="all"),"n_iter",rec["n_iter"],flush=True)

if __name__=="__main__":main()
