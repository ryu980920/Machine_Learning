#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP017 fine stage; dynamically selected architectures from frozen coarse study.

No Handmade file accesses. Official Train 76800-only fitting; OfficialVal evaluation.
"""
from __future__ import annotations
import argparse,json,warnings,time
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from exp006_size_com_comparison import PixelScaler255
from exp017_hparam_coarse import load_data,measure,save,SEEDS

KINDS=("alpha_down","alpha_up","lr_down","lr_up","early_stopping")
def configs(coarse_json):
    doc=json.loads(Path(coarse_json).read_text(encoding="utf-8"))
    assert doc["stage"]=="coarse" and doc["experiment"]=="EXP017"
    selected=doc["selected_4_architectures"]
    assert len(selected)==4 and len({tuple(x["hidden"]) for x in selected})==4
    allc=[]
    for slot,v in enumerate(selected):
        h=tuple(v["hidden"]);a=float(v["alpha"]);lr=float(v["lr"])
        for kind in KINDS:
            aa=a;ll=lr;early=False
            if kind=="alpha_down":aa=max(1e-6,a*.3)
            if kind=="alpha_up":aa=min(.1,a*3)
            if kind=="lr_down":ll=lr*.7
            if kind=="lr_up":ll=lr*1.4
            if kind=="early_stopping":early=True
            name=f"slot{slot+1}_{kind}"
            allc.append({"variant":name,"base_variant":v["variant"],
              "slot":slot,"kind":kind,"hidden":h,"alpha":aa,
              "learning_rate_init":ll,"early_stopping":early,
              "max_iter":260 if early else 200,
              "n_iter_no_change":15 if early else 10,
              "validation_fraction":.1 if early else .1})
    assert len(allc)==20 and len({x["variant"] for x in allc})==20
    return allc
def run(seed,slot,coarse_json,out):
    assert seed in SEEDS and slot in range(4)
    selected=[c for c in configs(coarse_json) if c["slot"]==slot]
    assert len(selected)==5
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    started=time.monotonic()
    X,Y,val,yv,digest,naug,stats=load_data()
    records=[];metrics=[]
    for config in selected:
        name=config["variant"]
        args={k:config[k] for k in ("hidden","alpha","learning_rate_init","early_stopping","max_iter","n_iter_no_change","validation_fraction") if k!="hidden"}
        args["hidden_layer_sizes"]=config["hidden"]
        args.update({"activation":"relu","solver":"adam","random_state":seed})
        model=Pipeline([("scale_0_1",PixelScaler255()),("model",MLPClassifier(**args))])
        begin=time.monotonic()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X,Y)
        joblib.dump(model,out/(name+".joblib"),compress=3)
        rows,response=measure(model,val,yv,seed,name,(config["hidden"],config["alpha"],config["learning_rate_init"]))
        metrics+=rows
        import numpy as np
        np.savez_compressed(out/(name+"_officialval_response.npz"),**response)
        accuracy=next(x["accuracy"] for x in rows if x["scale"]==1. and x["group"]=="all")
        records.append({**{k:(list(v) if isinstance(v,tuple) else v) for k,v in config.items()},
                        "seed":seed,"n_train_source":len(Y),"n_augment":naug,
                        "original_official_sha256":digest,
                        "stroke_modified":stats["modified"],"stroke_rejected":stats["rejected"],
                        "n_iter":model.named_steps["model"].n_iter_,
                        "official_val_accuracy":accuracy,
                        "fit_seconds":time.monotonic()-begin,
                        "warnings":" | ".join(str(i.message) for i in w)})
        print("EXP017_FINE",seed,slot,name,"acc",round(accuracy,6),
              "iter",model.named_steps["model"].n_iter_,flush=True)
        del model
    save(out/"training_records.csv",records)
    save(out/"metrics.csv",metrics)
    manifest={"experiment":"EXP017","stage":"fine","seed":seed,"slot":slot,
       "base_configs":[c["base_variant"] for c in selected],
       "configs":[x["variant"] for x in selected],
       "training_rows_source":76800,"official_val_rows":19200,"augmented_rows":naug,
       "official_source_sha256":digest,"preprocessing":"80pct_size_14pct_stroke_com",
       "team_handwriting_loaded":False,"handmade_used":False,
       "official_test_used":False,"professor_test_used":False,
       "duration_seconds":time.monotonic()-started}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("EXP017_FINE_COMPLETE",seed,slot,flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,required=True,choices=SEEDS)
    ap.add_argument("--slot",type=int,required=True,choices=(0,1,2,3))
    ap.add_argument("--coarse-json",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    run(args.seed,args.slot,args.coarse_json,args.out)
