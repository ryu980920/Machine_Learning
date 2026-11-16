#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normal HypB EXP009: training-set augmentation using Original and exact HypA C9.

Comparison preregistered in HypA EXP004 future experiment section:
  A C9-only, B raw-only, C stratified 50:50 replacement,
  D raw+C9, E raw twice.
For equal stochastic-gradient-update budget, A/B/C use 200 epochs on 76,800;
D/E use 100 epochs on 153,600. All have 76,800 nominal minibatches
(batch_size='auto'=200). Train/val is the original source-safe split.
Each model evaluated on both raw and C9 transformed Official Val and Handmade.

No Original Test, no professor Test, no handmade fit or training.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier

from exp_a004_followup import Stage2Preprocessing
from exp_a004_official_mlp import load_inputs, checksum, CLASSES

CONDITIONS=("c9_only","raw_only","mix_50_50","raw_plus_c9","raw_twice")
EPOCHS={"c9_only":200,"raw_only":200,"mix_50_50":200,
        "raw_plus_c9":100,"raw_twice":100}
LABELS="0123456789ABCDEF"
PAIRS=(("B","8"),("8","B"),("D","0"),("0","D"),("5","F"),("F","5"))

def save_csv(path,rows):
    if not rows: raise ValueError("No rows for "+str(path))
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def group_rows(condition,seed,dataset,representation,truth,pred,writers=None):
    truth=np.asarray(truth,dtype=int)
    pred=np.asarray(pred,dtype=int)
    group={"all":np.ones(len(truth),dtype=bool),
           "digits":truth<10,"letters":truth>=10}
    for i,cl in enumerate(LABELS):
        group["class_"+cl]=(truth==i)
    if writers is not None:
        for w in [41,42,43,44]:
            group[f"writer_{w}"]=(writers==w)
            group[f"writer_{w}_digits"]=(writers==w)&(truth<10)
            group[f"writer_{w}_letters"]=(writers==w)&(truth>=10)
    out=[]
    for name,mask in group.items():
        if not mask.any():continue
        if name.startswith("class_"):
            ids=[LABELS.index(name[-1])]
        elif name=="digits" or name.endswith("_digits"):
            ids=list(range(10))
        elif name=="letters" or name.endswith("_letters"):
            ids=list(range(10,16))
        else:ids=list(range(16))
        t,p=truth[mask],pred[mask]
        out.append({"condition":condition,"seed":seed,"dataset":dataset,
                    "input":representation,"group":name,"n":len(t),
                    "correct":int(np.sum(t==p)),
                    "accuracy":float(np.mean(t==p)),
                    "macro_f1":float(f1_score(t,p,average="macro",
                                        labels=ids,zero_division=0))})
    return out

def train(condition,out,repo_root):
    if condition not in CONDITIONS:raise ValueError(condition)
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    root=Path(repo_root).resolve()
    x_raw,y_train,x_val,y_val,x_hm,y_hm,writers,positions=load_inputs(root)
    assert x_raw.shape==(76800,784) and x_val.shape==(19200,784)
    assert x_hm.shape==(2560,784)
    c9=Stage2Preprocessing("C9_BBoxLongside20").fit(x_raw)
    c9_info=c9.description()
    print("EXP009 ORIGINAL C9 FIT: "+json.dumps(c9_info),flush=True)
    if c9_info["shape"]["median"]!=20.0:
        raise AssertionError("Original HypA C9 target differs from 20px")
    x_c9=c9.transform(x_raw)
    v_c9=c9.transform(x_val)
    h_c9=c9.transform(x_hm)

    # Keep labels/sample index unchanged. All normalized C9 pixels are copied
    # from the ORIGINAL HypA Stage2 implementation, with final CoM.
    raw=np.asarray(x_raw,dtype=np.float32)/255.0
    adjusted=x_c9/255.0
    if condition=="c9_only":
        train_x=adjusted
        train_y=y_train
    elif condition=="raw_only":
        train_x=raw
        train_y=y_train
    elif condition=="mix_50_50":
        splitter=StratifiedShuffleSplit(n_splits=1,test_size=.5,random_state=20261116)
        _,c9_indices=next(splitter.split(raw,y_train))
        mask=np.zeros(len(raw),dtype=bool)
        mask[c9_indices]=True
        train_x=raw.copy()
        train_x[mask]=adjusted[mask]
        train_y=y_train
        assert mask.sum()==38400
        del mask,c9_indices
    elif condition=="raw_plus_c9":
        train_x=np.concatenate([raw,adjusted],axis=0)
        train_y=np.concatenate([y_train,y_train],axis=0)
    elif condition=="raw_twice":
        train_x=np.concatenate([raw,raw],axis=0)
        train_y=np.concatenate([y_train,y_train],axis=0)
    assert len(train_x)==(153600 if condition in ("raw_plus_c9","raw_twice") else 76800)
    assert len(train_y)==len(train_x)
    assert np.isfinite(train_x).all() and 0<=train_x.min()<=train_x.max()<=1.0

    evaluation={
        ("OfficialVal","raw"):(x_val/255.0,y_val,None,None),
        ("OfficialVal","c9"):(v_c9/255.0,y_val,None,None),
        ("Handmade","raw"):(x_hm/255.0,y_hm,writers,positions),
        ("Handmade","c9"):(h_c9/255.0,y_hm,writers,positions),
    }
    metrics=[]
    predictions=[]
    pair_errors=[]
    fits=[]
    source_path=root/"02_data/datasets/MNIST_EMNIST_official_split_112000.npz"
    for seed in [42,43,44]:
        model=MLPClassifier(hidden_layer_sizes=(100,),activation="relu",
            solver="adam",alpha=0.0001,learning_rate_init=0.001,
            max_iter=EPOCHS[condition],tol=0.0,
            n_iter_no_change=201,batch_size="auto",shuffle=True,
            random_state=seed)
        tic=time.monotonic()
        print(f"START EXP009 condition={condition} seed={seed} "
              f"samples={len(train_y)} max_iter={model.max_iter}",flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",ConvergenceWarning)
            model.fit(train_x,train_y)
        duration=time.monotonic()-tic
        for (dataset,representation),(X,truth,ws,ids) in evaluation.items():
            pred=model.predict(X).astype(int)
            metrics.extend(group_rows(condition,seed,dataset,representation,
                                      truth,pred,ws))
            if dataset=="Handmade":
                for i in range(len(truth)):
                    predictions.append({"condition":condition,"seed":seed,
                        "input":representation,"writer":int(ws[i]),
                        "index_in_writer":int(ids[i]),
                        "actual":LABELS[int(truth[i])],
                        "predicted":LABELS[int(pred[i])]})
                for t,p in PAIRS:
                    pair_errors.append({"condition":condition,"seed":seed,
                      "input":representation,"actual":t,"predicted":p,
                      "count":int(np.sum((truth==LABELS.index(t))&
                                              (pred==LABELS.index(p))))})
            overall=np.mean(truth==pred)
            print(f"RESULT {condition} seed={seed} {dataset}/{representation}: "
                  f"{overall*100:.4f}%",flush=True)
        model_file=out/f"mlp_{condition}_seed{seed}.joblib"
        joblib.dump(model,model_file,compress=2)
        fits.append({"condition":condition,"seed":seed,
                     "train_examples":len(train_y),"epochs_limit":EPOCHS[condition],
                     "minibatch_updates_nominal":EPOCHS[condition]*
                       ((len(train_y)+199)//200),
                     "actual_epochs":int(model.n_iter_),
                     "train_loss":float(model.loss_),
                     "train_seconds":round(duration,1),
                     "model_sha256":checksum(model_file)})
        save_csv(out/"group_metrics.csv",metrics)
        save_csv(out/"handmade_predictions.csv",predictions)
        save_csv(out/"pair_errors.csv",pair_errors)
        save_csv(out/"fit_metrics.csv",fits)
        print(f"FIT END {condition} seed={seed} "
              f"epoch={model.n_iter_} time={duration:.1f}s",flush=True)
        del model
        gc.collect()
    manifest={"experiment":"Normal-HypB-EXP009",
        "condition":condition,"seeds":[42,43,44],
        "study":"Original HypA C9; raw+C9 Data Augmentation",
        "training_n":len(train_y),"original_unique_n":76800,
        "epochs_limit":EPOCHS[condition],"batch_size_auto":200,
        "nominal_gradient_updates":EPOCHS[condition]*((len(train_y)+199)//200),
        "fixed_aug_selection_seed":20261116,
        "original_hypa_c9":c9_info,
        "source_sha256":checksum(source_path),
        "python_version":platform.python_version(),
        "numpy_version":np.__version__,"scipy_version":scipy.__version__,
        "sklearn_version":sklearn.__version__,
        "original_test_used":False,"professor_test_used":False,
        "handmade_used_for_fitting":False,
        "evaluation_inputs":["raw","c9"],"evaluation_datasets":["OfficialVal","Handmade"],
        "fit_metrics":fits}
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),
                                         encoding="utf-8")
    print("COMPLETE EXP009 "+condition+" all 3 models trained",flush=True)

def read_csv(p):
    with Path(p).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def aggregate(root,out):
    root=Path(root)
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    files={}
    grouped=[]
    preds={}
    manifests={}
    for condition in CONDITIONS:
        folder=root/("exp009-"+condition)
        manifest=json.loads((folder/"run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["condition"]==condition
        assert not manifest["original_test_used"] and not manifest["professor_test_used"]
        assert not manifest["handmade_used_for_fitting"]
        assert manifest["nominal_gradient_updates"]==76800
        manifests[condition]=manifest
        rows=read_csv(folder/"group_metrics.csv")
        for s in [42,43,44]:
            for dataset in ["OfficialVal","Handmade"]:
                for input_kind in ["raw","c9"]:
                    for group in ["all","digits","letters"]:
                        if len([r for r in rows if r["seed"]==str(s) and
                            r["dataset"]==dataset and r["input"]==input_kind and
                            r["group"]==group])!=1:
                            raise AssertionError(f"Missing metric: {condition} {s} {dataset} {input_kind} {group}")
        grouped.extend(rows)
        pred=read_csv(folder/"handmade_predictions.csv")
        for seed in [42,43,44]:
            for inp in ["raw","c9"]:
                subset=[r for r in pred if r["seed"]==str(seed) and r["input"]==inp]
                assert len(subset)==2560
                key_set={(r["writer"],r["index_in_writer"]) for r in subset}
                assert len(key_set)==2560
                preds[condition,seed,inp]={ (r["writer"],r["index_in_writer"]):r
                                           for r in subset }
        files[condition]=folder

    # Preserve raw metrics, produce group summary and paired per-image changes.
    save_csv(out/"all_group_metrics.csv",grouped)
    summaries=[]
    for cond in CONDITIONS:
        for dataset in ["OfficialVal","Handmade"]:
            for inp in ["raw","c9"]:
                for group in ["all","digits","letters"]:
                    xs=[r for r in grouped if r["condition"]==cond and
                        r["dataset"]==dataset and r["input"]==inp and r["group"]==group]
                    if len(xs)!=3:raise AssertionError("Missing 3-seed row")
                    summaries.append({"condition":cond,"dataset":dataset,
                      "input":inp,"group":group,"n_per_seed":xs[0]["n"],
                      "seed_count":3,
                      "accuracy_mean":float(np.mean([float(r["accuracy"]) for r in xs])),
                      "accuracy_min":min(float(r["accuracy"]) for r in xs),
                      "accuracy_max":max(float(r["accuracy"]) for r in xs),
                      "macro_f1_mean":float(np.mean([float(r["macro_f1"]) for r in xs]))})
    save_csv(out/"three_seed_summary.csv",summaries)
    paired=[]
    for condition in CONDITIONS:
        if condition=="c9_only":continue
        for seed in [42,43,44]:
            for inp in ["raw","c9"]:
                source=preds["c9_only",seed,inp]
                candidate=preds[condition,seed,inp]
                for group in ["all","digits","letters"]:
                    keys=[k for k in source if group=="all" or
                          ((source[k]["actual"] in "0123456789")==(group=="digits"))]
                    fixed=broken=0
                    for key in keys:
                        a,b=source[key],candidate[key]
                        if a["actual"]!=b["actual"]:
                            raise AssertionError("Truth label mismatch")
                        ac=a["actual"]==a["predicted"]
                        bc=b["actual"]==b["predicted"]
                        fixed+=int(not ac and bc)
                        broken+=int(ac and not bc)
                    paired.append({"condition":condition,"seed":seed,
                      "input":inp,"group":group,"n":len(keys),
                      "corrected_from_c9":fixed,"new_wrong_vs_c9":broken,
                      "net_correct":fixed-broken,
                      "delta_acc_pp":100*(fixed-broken)/len(keys)})
    save_csv(out/"paired_handmade_vs_c9.csv",paired)
    def pct(c,d,k,g):
        rows=[r for r in summaries if r["condition"]==c and r["dataset"]==d
              and r["input"]==k and r["group"]==g]
        assert len(rows)==1
        return f"{100*rows[0]['accuracy_mean']:.4f}%"
    result=["# Normal 가설 B EXP유009 — 원본 + C9 Data Augmentation 실측",
        "",
        "**상태: 공식 Train 재학습 및 3-Seed 평가 완료.**",
        "",
        "- HypA EXP004 original C9 implementation and same train split.",
        "- All 5 methods fit only on official Train, no original/professor Test, no handmade training.",
        "- Training minibatch update budget **matched at 76,800 nominal steps** (76800×200 epochs vs 153600×100 epochs).",
        "- Train original vs C9 selection 50:50 uses fixed StratifiedShuffleSplit seed=20261116.",
        "- Both raw and C9 inference separately evaluated, no post-hoc selection on handmade.",
        "",
        "## Evaluation on C9-normalized inputs",
        "",
        "| Training | Official Val | Handmade all | Digits | Letters |",
        "|---|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        result.append(f"| {c} | {pct(c,'OfficialVal','c9','all')} | "
          f"{pct(c,'Handmade','c9','all')} | {pct(c,'Handmade','c9','digits')} | "
          f"{pct(c,'Handmade','c9','letters')} |")
    result+=["","## Evaluation on original raw inputs","","| Training | Official Val | Handmade all | Digits | Letters |",
              "|---|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        result.append(f"| {c} | {pct(c,'OfficialVal','raw','all')} | "
          f"{pct(c,'Handmade','raw','all')} | {pct(c,'Handmade','raw','digits')} | "
          f"{pct(c,'Handmade','raw','letters')} |")
    result+=["","## Interpretation",
      "- Differences are based on actual 3-seed model training in the same workflow.",
      "- C9-only versus raw+C9 and raw twice isolates added transform diversity from mere sample repeats under matched nominal step budget.",
      "- Each 153600 set contains **the same 76800 underlying official training images** twice, with an alternate view; it is not independent new data.",
      "- The four writers in handmade have been used in development repeatedly; do not treat their scores as independent test performance.",
      "- Official Validation threshold ≥97.30% and handmade digits ≥85%, letters ≥80% are simultaneous criteria.",
      "- No final Pipeline change and no main merge are implied."
    ]
    (out/"SUMMARY.md").write_text("\n".join(result)+"\n",encoding="utf-8")
    print("EXP009 SUMMARY COMPLETED",flush=True)
    print("\n".join(result),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",default=".")
    p.add_argument("--condition",choices=CONDITIONS)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--aggregate",action="store_true")
    p.add_argument("--input-root",type=Path)
    args=p.parse_args()
    if args.aggregate:
        if args.input_root is None:p.error("--aggregate requires input-root")
        aggregate(args.input_root,args.output)
    else:
        if args.condition is None:p.error("Need --condition")
        train(args.condition,args.output,args.repo_root)

if __name__=="__main__":
    main()
