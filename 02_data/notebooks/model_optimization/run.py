#!/usr/bin/env python3
"""Train fresh baseline/candidate estimators using EXP-유-003's exact split.

No pretrained .joblib loading; official Test and Handmade are not read.
"""
from __future__ import annotations
import argparse, csv, importlib, json, platform, time, warnings
from pathlib import Path
import numpy as np
import sklearn
from common import fixed_split, pipeline, scores, DEFAULT_DATA, LABELS

KEYS = ("svc", "hgb", "mlp", "extra_trees", "knn")
BASELINE_ACCURACY = {
 "svc":.9766666666666667, "hgb":.9746354166666666,
 "mlp":.9731770833333333, "extra_trees":.9696875,
 "knn":.9629166666666666,
}

def run_model(key, variant, arrays, outdir):
    xtr,ytr,xva,yva,sha=arrays
    module=importlib.import_module(key)
    estimator,scaled=(module.build_baseline() if variant=="baseline" else module.build_candidate())
    model=pipeline(estimator,scaled)
    t=time.perf_counter()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        model.fit(xtr,ytr)
    fit=time.perf_counter()-t
    t=time.perf_counter()
    pred=model.predict(xva)
    predict=time.perf_counter()-t
    result={
       "key":key,"variant":variant,
       "dataset_sha256":sha,"train_n":len(ytr),"validation_n":len(yva),
       "split":"StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42), first fold",
       "scaling":"pixel/255" if scaled else "raw 0-255",
       "hyperparameters":{str(k):repr(v) for k,v in estimator.get_params().items()},
       "fit_seconds":fit,"predict_seconds":predict,
       "warnings":[f"{w.category.__name__}: {w.message}" for w in ws],
       "python":platform.python_version(),"sklearn":sklearn.__version__,"numpy":np.__version__,
       "historical_baseline_accuracy":BASELINE_ACCURACY[key],
       **scores(yva,pred)
    }
    target=outdir/variant/key
    target.mkdir(parents=True,exist_ok=True)
    (target/"metrics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    with (target/"confusion_matrix.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["true/pred",*LABELS])
        for c,row in zip(LABELS,result["confusion_matrix"]):w.writerow([c,*row])
    with (target/"per_class.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f);w.writerow(["class","precision","recall","f1","support"])
        for c,d in result["per_class"].items():w.writerow([c,d["precision"],d["recall"],d["f1"],d["support"]])
    print(f"{key} {variant}: accuracy={result['accuracy']:.6f}, macro_f1={result['macro_f1']:.6f}, historical={BASELINE_ACCURACY[key]:.6f}",flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True,choices=[*KEYS,"all"])
    p.add_argument("--variant",choices=["baseline","candidate"],default="baseline")
    p.add_argument("--data",type=Path,default=DEFAULT_DATA)
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"outputs")
    args=p.parse_args()
    arrays=fixed_split(args.data)
    for key in (KEYS if args.model=="all" else [args.model]):
        run_model(key,args.variant,arrays,args.output)

if __name__=="__main__":
    main()
