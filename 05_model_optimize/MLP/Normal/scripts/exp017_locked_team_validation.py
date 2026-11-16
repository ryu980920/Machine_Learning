#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-freeze team handwriting validation ONLY. Do not run during EXP017 tuning.

Input .npz: X [N,784] (0..255), y [N] with labels 0..15,
optional writer_id [N]. Validate unseen handwritten sheets without refitting.
No Original Test or professor Test. Includes paired 0.8-1.2x interventions.
"""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np,joblib
from exp006_size_com_comparison import LABELS,process
from exp011_012_causal_probes import fixed_scale
SCALES=(.8,.9,1.,1.1,1.2)
def csv_write(path,rs):
    assert rs
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def run(data,model_folder,out):
    out=Path(out);data=Path(data);model_folder=Path(model_folder)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    with np.load(data,allow_pickle=False) as npz:
        assert "X" in npz and "y" in npz
        X=np.array(npz["X"],dtype=np.float32)
        y=np.array(npz["y"],dtype=int)
        writers=(np.array(npz["writer_id"]).astype(str)
                 if "writer_id" in npz else np.array(["unknown"]*len(y)))
    assert X.ndim==2 and X.shape[1]==784
    assert np.isfinite(X).all() and np.min(X)>=0 and np.max(X)<=255
    assert y.shape==(len(X),) and writers.shape==(len(X),)
    assert np.isin(y,np.arange(16)).all()
    assert len(np.unique(y))==16,"every class 0..15 required"
    prepared={s:process(X if s==1. else fixed_scale(X,s),"com",20)
              for s in SCALES}
    ckpts=sorted(model_folder.glob("seed*.joblib"))
    assert len(ckpts)==3,ckpts
    records=[];pair=[];loaded=[]
    for path in ckpts:
        import re
        z=re.fullmatch(r"seed(42|43|44)\.joblib",path.name)
        assert z is not None,path.name
        seed=int(z.group(1))
        mdl=joblib.load(path)
        pf={};yp={}
        for s in SCALES:
            prob=np.asarray(mdl.predict_proba(prepared[s]),dtype=np.float32)
            assert prob.shape==(len(y),16)
            pf[s]=np.sum(prob[:,10:],axis=1)
            yp[s]=np.argmax(prob,axis=1)
        baseline=(pf[1.]>=.5)
        masks={"all":np.ones(len(y),bool),"digits":y<10,"letters":y>=10}
        for i,c in enumerate(LABELS):masks["class_"+c]=y==i
        for w in np.unique(writers):
            masks["writer_"+str(w)]=writers==w
        for scale in SCALES:
            for group,mask in masks.items():
                if not mask.any():continue
                records.append({"seed":seed,"scale":scale,"group":group,"n":int(mask.sum()),
                    "accuracy":float(np.mean(yp[scale][mask]==y[mask])),
                    "family_flip":float(np.mean((pf[scale][mask]>=.5)!=baseline[mask])),
                    "mean_abs_delta_p_letter":float(np.mean(np.abs(pf[scale][mask]-pf[1.][mask])))})
        for actual,pred in (("B","8"),("8","B"),("D","0"),("0","D"),("F","5"),("5","F")):
            m=y==LABELS.index(actual)
            pair.append({"seed":seed,"actual":actual,"predicted":pred,
                "count":int(np.sum(yp[1.][m]==LABELS.index(pred)))})
        loaded.append({"seed":seed,"filename":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    csv_write(out/"team_validation_metrics_all_seed.csv",records)
    csv_write(out/"team_validation_confusion_pairs.csv",pair)
    summ=[]
    for group in ("all","digits","letters"):
        for scale in SCALES:
            values=[r["accuracy"] for r in records if r["group"]==group and r["scale"]==scale]
            flips=[r["family_flip"] for r in records if r["group"]==group and r["scale"]==scale]
            assert len(values)==3
            summ.append({"group":group,"scale":scale,
                "accuracy_seed_mean":float(np.mean(values)),
                "family_flip_seed_mean":float(np.mean(flips)),
                "accuracy_seed_sd":float(np.std(values,ddof=1))})
    csv_write(out/"team_validation_summary.csv",summ)
    manifest={"experiment":"EXP017","stage":"locked_team_validation",
      "dataset_sha256":hashlib.sha256(data.read_bytes()).hexdigest(),
      "n_images":len(y),"n_unique_writers":len(np.unique(writers)),
      "per_class_counts":{cl:int(np.sum(y==i)) for i,cl in enumerate(LABELS)},
      "frozen_checkpoints":loaded,"inference_only":True,
      "trained_or_tuned_on_current_data":False,
      "original_test_used":False,"professor_test_used":False,
      "disclaimer":"If images overlap EXP014-016 Handmade, this is reused development data, not independent blind holdout."}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("EXP017_LOCKED_TEAM_VALIDATION_DONE",len(y),len(np.unique(writers)),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz",required=True,type=Path,help="New team handwriting X[N,784], y[N], optional writer_id[N]")
    ap.add_argument("--frozen-models",required=True,type=Path,help="seed42.joblib, seed43.joblib, seed44.joblib")
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    run(a.npz,a.frozen_models,a.out)
