#!/usr/bin/env python3
"""EXP-CMB-003: 10 controlled input perturbations vs 5 frozen EXP-CMB-001 MLPs.

Scientific scope: diagnose whether the observed writer/class geometry & ink
differences cause meaningful corrections under controlled, label-blind changes.
No training. No Original Test, professor Test or new input-derived fitting.
The same transformation is applied to all images for each condition.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from scipy.ndimage import maximum_filter, minimum_filter, affine_transform
from scipy.stats import binomtest
import scipy
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

ROOT=Path(__file__).resolve().parents[4]
COMBINED=ROOT/"05_model_optimize"/"MLP"/"Combined"
sys.path.insert(0,str(COMBINED/"scripts"))
import exp_cmb002_evidence as core

WRITERS=core.WRITERS
LABELS=core.LABELS
FOCUS=(("0","D"),("D","0"),("8","B"),("B","8"))
PRIMARY=(("41","D","0"),("43","8","B"),("43","0","D"),("41","B","8"))

@dataclass(frozen=True)
class Variant:
    name:str
    family:str
    value:object

VARIANTS=[
    Variant("original","original",None),
    Variant("shift_L1","shift",(-1,0)),
    Variant("shift_R1","shift",(1,0)),
    Variant("shift_U1","shift",(0,-1)),
    Variant("shift_D1","shift",(0,1)),
    Variant("scale_095","scale",.95),
    Variant("scale_105","scale",1.05),
    Variant("stroke_thin_25","stroke",("thin",.25)),
    Variant("stroke_thick_25","stroke",("thick",.25)),
    Variant("brightness_090","brightness",.90),
    Variant("brightness_110","brightness",1.10),
]
assert len(VARIANTS)==11

def transform(batch,condition):
    """Exact Normal Input_Sensitivity transform for matching variants."""
    a=np.asarray(batch,dtype=np.float32).reshape(-1,28,28)
    if condition.family=="original":
        z=a.copy()
    elif condition.family=="shift":
        dx,dy=condition.value
        z=np.zeros_like(a)
        ys=slice(max(0,-dy),min(28,28-dy))
        yd=slice(max(0,dy),min(28,28+dy))
        xs=slice(max(0,-dx),min(28,28-dx))
        xd=slice(max(0,dx),min(28,28+dx))
        z[:,yd,xd]=a[:,ys,xs]
    elif condition.family=="brightness":
        z=np.clip(a*float(condition.value),0,255)
    elif condition.family=="stroke":
        direction,fraction=condition.value
        if direction=="thick":
            modified=maximum_filter(a,size=(1,3,3),mode="constant",cval=0)
        else:
            modified=minimum_filter(a,size=(1,3,3),mode="constant",cval=0)
        z=(1-fraction)*a+fraction*modified
    elif condition.family=="scale":
        inv=1.0/float(condition.value)
        offset=13.5*(1.0-inv)
        z=affine_transform(a,matrix=np.diag([1,inv,inv]),
                           offset=(0.0,offset,offset),
                           output_shape=a.shape,order=1,mode="constant",
                           cval=0.0,prefilter=False)
    else:
        raise ValueError(condition.family)
    return np.rint(np.clip(z,0,255)).astype(np.float32).reshape(-1,784)

def self_test():
    x=np.zeros((2,784),dtype=np.float32)
    x[0,13*28+14]=255
    x[1,0]=255
    assert np.array_equal(transform(x,VARIANTS[0]),x)
    assert transform(x,VARIANTS[1])[1].sum()==0
    for v in VARIANTS:
        q=transform(x,v)
        assert q.shape==x.shape and q.dtype==np.float32
        assert q.min()>=0 and q.max()<=255
    print("TRANSFORM SELF-TEST 11 CONDITIONS PASSED",flush=True)

def load_official(manifest):
    path=ROOT/manifest["input_data_paths"]["official"]
    assert core.sha256(path)==manifest["dataset_sha256"]["official_v3"]
    split=COMBINED/"results"/"EXP_CMB_001"/"official_split_indices.npz"
    with np.load(split,allow_pickle=False) as d:
        train=d["train_idx"].astype(int)
        val=d["val_idx"].astype(int)
        train_group=d["train_group_ids"].astype(int)
        val_group=d["val_group_ids"].astype(int)
    assert (len(train),len(val))==(76800,19200)
    assert len(np.unique(np.r_[train,val]))==96000
    assert not np.intersect1d(train_group,val_group).size
    # Only read 96,000 official TRAIN arrays. NEVER access any *test* key.
    with np.load(path,allow_pickle=False) as d:
        assert "X_train" in d and "y_train" in d
        X=d["X_train"][val].reshape(-1,784).astype(np.float32)
        y=d["y_train"][val].astype(int)
    assert X.shape==(19200,784) and np.bincount(y,minlength=16).tolist()==manifest["official_split"]["validation_class_counts_0_to_15"]
    return X,y

def metrics(y,p):
    scores=precision_recall_fscore_support(y,p,labels=np.arange(16),zero_division=0)
    prec,rec,f1,_=scores
    return {"n":len(y),"accuracy":float(np.mean(y==p)),
            "macro_precision":float(prec.mean()),"macro_recall":float(rec.mean()),
            "macro_f1":float(f1.mean()),"correct":int(np.sum(y==p))}

def paired_counts(y,before,after):
    was=before==y
    now=after==y
    c=int(np.sum(~was&now))
    i=int(np.sum(was&~now))
    f=int(np.sum(before!=after))
    wrongchanged=int(np.sum(~was&~now&(before!=after)))
    p=float(binomtest(c,c+i,p=.5).pvalue) if c+i else 1.0
    return {"corrections":c,"induced_errors":i,"net_correct":c-i,
            "prediction_flip_count":f,"prediction_flip_rate":f/len(y),
            "wrong_to_other_wrong":wrongchanged,"mcnemar_exact_p":p}

def bh(rows,key="mcnemar_exact_p"):
    idx=np.argsort([r[key] for r in rows])
    q=1.0
    for rank in range(len(rows)-1,-1,-1):
        i=idx[rank]
        q=min(q,rows[i][key]*len(rows)/(rank+1))
        rows[i]["BH_q_10_conditions"]=float(q)

def plot_cm(y,p,out,name):
    cm=confusion_matrix(y,p,labels=np.arange(16))
    for mode in ("count","normalized"):
        val=cm if mode=="count" else 100*cm/np.maximum(cm.sum(1,keepdims=True),1)
        fig,ax=plt.subplots(figsize=(9,8))
        maxcolor=160 if mode=="count" else 100
        im=ax.imshow(val,cmap="viridis",vmin=0,vmax=maxcolor)
        fig.colorbar(im,ax=ax,fraction=.045,pad=.03)
        ax.set(xticks=np.arange(16),yticks=np.arange(16),xticklabels=list(LABELS),
               yticklabels=list(LABELS),xlabel="Predicted",ylabel="Actual",
               title=f"Combined OOF | {name} | {mode}")
        for i in range(16):
            for j in range(16):
                if cm[i,j]:
                    ax.text(j,i,("%.1f"%val[i,j] if mode=="normalized" else str(cm[i,j])),
                            ha="center",va="center",fontsize=5.5,
                            color="black" if val[i,j]>.53*maxcolor else "white")
        fig.tight_layout()
        fig.savefig(out/f"{name}_{mode}.png",dpi=160)
        plt.close(fig)

def plot_tradeoff(rows,out):
    v=[r for r in rows if r["variant"]!="original" and r["model"]=="Combined"]
    fig,ax=plt.subplots(figsize=(9,5))
    x=np.arange(len(v))
    ax.bar(x-.2,[r["corrections"] for r in v],width=.4,label="Wrong→correct")
    ax.bar(x+.2,[r["induced_errors"] for r in v],width=.4,label="Correct→wrong")
    ax.set(xticks=x,xticklabels=[r["variant"] for r in v],
           ylabel="OOF image count",title="EXP-CMB-003 | Corrections vs induced errors")
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"corrections_vs_induced.png",dpi=180)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    figdir=out/"figures"
    figdir.mkdir(parents=True,exist_ok=True)
    self_test()
    started=time.perf_counter()
    manifest,prior,table,raw,labels=core.load_inputs()
    models=core.load_models(manifest)
    normal=models["normal_mlp.joblib"]
    X=np.concatenate([raw[w].reshape(640,784).astype(np.float32) for w in WRITERS])
    y=np.concatenate([labels[w] for w in WRITERS]).astype(int)
    writers=np.repeat(np.asarray(WRITERS),640)
    indices=np.tile(np.arange(640),4)
    assert len(X)==2560 and len(prior)==2560
    assert np.array_equal(writers,[r["writer"] for r in prior])
    assert np.array_equal(indices,[r["index_in_writer"] for r in prior])
    assert np.array_equal(y,[LABELS.index(r["true_label"]) for r in prior])
    maskw={w:(writers==w) for w in WRITERS}

    official_x,official_y=load_official(manifest)
    models_all={"Normal":normal,"Combined":None}
    base_pred={}
    base_pred["Normal"]=normal.predict(X).astype(int)
    com_pred=np.zeros(2560,dtype=int)
    for w in WRITERS:
        com_pred[maskw[w]]=models["combined_heldout_"+str(w)+"_mlp.joblib"].predict(X[maskw[w]])
    base_pred["Combined"]=com_pred
    for model,col in (("Normal","normal_pred"),("Combined","combined_pred")):
        expected=np.asarray([LABELS.index(r[col]) for r in prior])
        assert np.array_equal(base_pred[model],expected),(model,"baseline mismatch")
    assert (np.sum(base_pred["Normal"]==y),np.sum(base_pred["Combined"]==y))==(1602,2080)
    print("REPRODUCED ALL ORIGINAL OOF PREDICTIONS FOR NORMAL AND COMBINED",flush=True)

    base_official={"Normal":normal.predict(official_x).astype(int)}
    for w in WRITERS:
        base_official["Combined_"+str(w)]=models["combined_heldout_"+str(w)+"_mlp.joblib"].predict(official_x).astype(int)
    assert np.mean(base_official["Normal"]==official_y)==manifest["evaluations"]["normal_official_dev_validation"]["accuracy"]
    original_mass=np.maximum(X.sum(axis=1),1)
    original_official_mass=np.maximum(official_x.sum(axis=1),1)

    rows_oof=[];rows_writer=[];rows_group=[];rows_class=[];rows_pair=[];rows_official=[]
    predrows=[];diagnostics=[];full_oof={}
    for cond in VARIANTS:
        transformed=transform(X,cond)
        x_off=transform(official_x,cond)
        mass_loss=np.maximum(0,1-transformed.sum(axis=1)/original_mass)
        mass_change=(transformed.sum(axis=1)-X.sum(axis=1))/original_mass
        clip_or_change=float(np.mean(mass_loss>1e-8))
        s_time=time.perf_counter()
        new_pred={"Normal":normal.predict(transformed).astype(int)}
        cp=np.zeros_like(y)
        for w in WRITERS:
            cp[maskw[w]]=models["combined_heldout_"+str(w)+"_mlp.joblib"].predict(transformed[maskw[w]]).astype(int)
        new_pred["Combined"]=cp
        handmade_predict_sec=time.perf_counter()-s_time
        full_oof[cond.name]=cp.copy()
        # Ensure first condition exactly reconstructs original and no label-based changes.
        if cond.name=="original":
            assert all(np.array_equal(new_pred[m],base_pred[m]) for m in ("Normal","Combined"))

        diagnostics.append({"variant":cond.name,"family":cond.family,
           "handmade_any_mass_loss_fraction":clip_or_change,
           "handmade_average_fraction_mass_lost":float(mass_loss.mean()),
           "handmade_average_fraction_mass_change":float(mass_change.mean()),
           "handmade_fraction_nonidentical":float(np.mean(np.any(transformed!=X,axis=1))),
           "official_average_fraction_mass_lost":float(np.maximum(0,1-x_off.sum(1)/original_official_mass).mean()),
           "handmade_model_prediction_seconds":handmade_predict_sec})
        for m in ("Normal","Combined"):
            p=new_pred[m]
            before=base_pred[m]
            base=metrics(y,before)
            now=metrics(y,p)
            paired=paired_counts(y,before,p)
            rows_oof.append({"variant":cond.name,"family":cond.family,"model":m,
               **{k+"_baseline":v for k,v in base.items() if k in ("accuracy","macro_f1")},
               **now,"delta_acc_pp":100*(now["accuracy"]-base["accuracy"]),
               "delta_macro_f1":now["macro_f1"]-base["macro_f1"],
               **paired})
            for w in WRITERS:
                z=maskw[w]
                mm=metrics(y[z],p[z])
                pp=paired_counts(y[z],before[z],p[z])
                rows_writer.append({"variant":cond.name,"model":m,"writer":w,
                  **mm,"delta_acc_pp":100*(mm["accuracy"]-np.mean(before[z]==y[z])),
                  **pp})
            for name,group_mask in (("digit",y<10),("letter",y>=10)):
                mm=metrics(y[group_mask],p[group_mask])
                pp=paired_counts(y[group_mask],before[group_mask],p[group_mask])
                rows_group.append({"variant":cond.name,"model":m,"group":name,
                  **mm,"delta_acc_pp":100*(mm["accuracy"]-np.mean(before[group_mask]==y[group_mask])),
                  **pp})
            prec,rec,f1,support=precision_recall_fscore_support(y,p,labels=np.arange(16),zero_division=0)
            for j,c in enumerate(LABELS):
                z=y==j
                rows_class.append({"variant":cond.name,"model":m,"class":c,"n":int(support[j]),
                  "precision":float(prec[j]),"recall":float(rec[j]),"f1":float(f1[j]),
                  "before_correct":int(np.sum(before[z]==y[z])),
                  "after_correct":int(np.sum(p[z]==y[z])),
                  "corrections":int(np.sum((before[z]!=y[z])&(p[z]==y[z]))),
                  "induced_errors":int(np.sum((before[z]==y[z])&(p[z]!=y[z])))})
            for actual,rival in FOCUS:
                mask=y==LABELS.index(actual)
                wrong=int(np.sum(p[mask]==LABELS.index(rival)))
                old=int(np.sum(before[mask]==LABELS.index(rival)))
                rows_pair.append({"variant":cond.name,"model":m,"writer":"all",
                  "actual":actual,"predicted":rival,"original_errors":old,
                  "new_errors":wrong,"delta_wrong":wrong-old})
                for w in WRITERS:
                    z=mask&maskw[w]
                    rows_pair.append({"variant":cond.name,"model":m,"writer":w,
                     "actual":actual,"predicted":rival,
                     "original_errors":int(np.sum(before[z]==LABELS.index(rival))),
                     "new_errors":int(np.sum(p[z]==LABELS.index(rival))),
                     "delta_wrong":int(np.sum(p[z]==LABELS.index(rival))-
                                      np.sum(before[z]==LABELS.index(rival)))})
            # Save image-level paired transitions as compressed CSV
            for i in range(len(y)):
                predrows.append((cond.name,m,int(writers[i]),int(indices[i]),LABELS[y[i]],
                   LABELS[before[i]],LABELS[p[i]],int(before[i]==y[i]),int(p[i]==y[i])))

        for key,model in (("Normal",normal),*[( "Combined_"+str(w),
            models["combined_heldout_"+str(w)+"_mlp.joblib"]) for w in WRITERS]):
            t0=time.perf_counter()
            q=model.predict(x_off).astype(int)
            tm=time.perf_counter()-t0
            before=base_official[key]
            mm=metrics(official_y,q)
            pp=paired_counts(official_y,before,q)
            rows_official.append({"variant":cond.name,"model":key,
                **mm,"delta_acc_pp":100*(mm["accuracy"]-np.mean(before==official_y)),
                "prediction_seconds":tm,**pp})
        plot_cm(y,new_pred["Combined"],figdir,cond.name)
        print(cond.name,"| Combined",rows_oof[-1]["accuracy"],
              "| net",rows_oof[-1]["net_correct"],flush=True)

    # FDR for 10 perturbations within the same model; avoid treating baseline as a hypothesis.
    for m in ("Normal","Combined"):
        rr=[r for r in rows_oof if r["model"]==m and r["variant"]!="original"]
        bh(rr)
    # Forward margins of true class vs predeclared competitor on same held-out images.
    margins=[]
    for cond in VARIANTS:
        x=transform(X,cond)
        lnorm=core.logits(normal,x)
        lcom=np.empty((2560,16),dtype=float)
        for w in WRITERS:
            z=maskw[w]
            lcom[z]=core.logits(models["combined_heldout_"+str(w)+"_mlp.joblib"],x[z])
        for m,log in (("Normal",lnorm),("Combined",lcom)):
            for wstr,actual,rival in PRIMARY:
                w=int(wstr)
                z=maskw[w]&(y==LABELS.index(actual))
                aa=log[z,LABELS.index(actual)]-log[z,LABELS.index(rival)]
                oldx=core.logits(normal if m=="Normal" else models["combined_heldout_"+str(w)+"_mlp.joblib"],X[z])
                old=oldx[:,LABELS.index(actual)]-oldx[:,LABELS.index(rival)]
                nwrong=int(np.sum(full_oof[cond.name][z]==LABELS.index(rival))) if m=="Combined" else ""
                margins.append({"variant":cond.name,"model":m,"writer":w,"actual":actual,
                    "competitor":rival,"n":int(z.sum()),
                    "mean_margin":float(aa.mean()),"mean_delta_margin":float((aa-old).mean()),
                    "fraction_margin_below_zero":float(np.mean(aa<0)),
                    "target_pair_error_count":nwrong})

    # Explicit followup gate: no new trained pipeline can be claimed by this diagnostic.
    crows=[r for r in rows_oof if r["model"]=="Combined" and r["variant"]!="original"]
    official_base={w:np.mean(base_official["Combined_"+str(w)]==official_y) for w in WRITERS}
    candidates=[]
    for r in crows:
        v=r["variant"]
        wr=[a for a in rows_writer if a["variant"]==v and a["model"]=="Combined"]
        os=[a for a in rows_official if a["variant"]==v and a["model"].startswith("Combined_")]
        good=(r["BH_q_10_conditions"]<.05 and r["net_correct"]>=13
              and sum(a["net_correct"]>=0 for a in wr)>=3
              and min(a["delta_acc_pp"] for a in os)>=-.3)
        if good:
            candidates.append({"variant":v,"oof_delta_pp":r["delta_acc_pp"],
                               "BH_q":r["BH_q_10_conditions"],
                               "min_official_delta_pp":min(a["delta_acc_pp"] for a in os),
                               "writers_not_worse":sum(a["net_correct"]>=0 for a in wr)})
    core.write_csv(out/"oof_metrics.csv",rows_oof)
    core.write_csv(out/"writer_metrics.csv",rows_writer)
    core.write_csv(out/"digit_letter_metrics.csv",rows_group)
    core.write_csv(out/"per_class_metrics.csv",rows_class)
    core.write_csv(out/"pair_confusion_changes.csv",rows_pair)
    core.write_csv(out/"official_validation_metrics.csv",rows_official)
    core.write_csv(out/"focus_margin_diagnostics.csv",margins)
    core.write_csv(out/"transform_diagnostics.csv",diagnostics)
    with gzip.open(out/"paired_predictions.csv.gz","wt",newline="",encoding="utf-8") as f:
        wr=csv.writer(f)
        wr.writerow(["variant","model","writer","index_in_writer","actual","original_prediction",
                     "transformed_prediction","original_correct","transformed_correct"])
        wr.writerows(predrows)
    plot_tradeoff(rows_oof,figdir)
    report={"experiment_id":"EXP-CMB-003","status":"completed_input_diagnostic",
       "baseline_reproduced":{"normal_correct":1602,"combined_correct":2080,"n":2560},
       "hypothesis":"B spatial and ink differences impact held-out writer errors",
       "conditions":[{"name":v.name,"family":v.family,"value":v.value} for v in VARIANTS],
       "predefined_candidate_gate":{
         "accuracy_gain_min_pp":.5,"paired_BH_q_lt":.05,
         "writer_nonworsening_min":3,"per_fold_official_delta_acc_min_pp":-.3},
       "screening_candidates_for_retraining":candidates,
       "decision":"SCREEN_RETRAIN_CANDIDATES" if candidates else "NO_GLOBAL_RETRAIN_CANDIDATE_FROM_TESTED_VARIANTS",
       "note":"Transform of validation alone is a fixed-model diagnostic; no inference accuracy gain proves retrained model improvement.",
       "normal_comparison":True,"normal_and_combined_frozen_model_count":5,
       "new_training_runs":0,"official_original_test_used":False,"professor_final_test_used":False,
       "versions":{"sklearn":sklearn.__version__,"numpy":np.__version__,"scipy":scipy.__version__},
       "models_zip_members_sha256":manifest["models_zip_members_sha256"],
       "elapsed_seconds":time.perf_counter()-started}
    (out/"run_manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    md=["# EXP-CMB-003 — Frozen Combined MLP input sensitivity: automatic metrics","",
        "**Status:** experimentally executed; final research interpretation in EXP유003 log.",
        "Frozen Normal 1 + Combined 4, original 2,560 Hand-made predictions revalidated.",
        "Official Original Test/Professor Test not loaded, no training.",
        "",
        "| Condition | OOF Accuracy | Δpp | Corrected | Newly wrong | Flip rate | McNemar exact p | BH q (10) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows_oof:
        if r["model"]=="Combined":
            md.append("| {} | {:.3%} | {:+.3f} | {} | {} | {:.2%} | {:.4g} | {} |".format(
                r["variant"],r["accuracy"],r["delta_acc_pp"],r["corrections"],
                r["induced_errors"],r["prediction_flip_rate"],r["mcnemar_exact_p"],
                ("{:.4g}".format(r["BH_q_10_conditions"]) if "BH_q_10_conditions" in r else "—")))
    md+=["",f"**Gate result:** {report['decision']}","",
         "Screening candidates are not finished models; further 4-Fold Training/Validation evaluation would be a separate study.",
         "No criterion met does not imply no geometry effect: it means these global single-variable inference-time interventions are not a ready-to-deploy solution.",
         "Consult writer_metrics.csv, pair_confusion_changes.csv, focus_margin_diagnostics.csv and official_validation_metrics.csv.",
         ""]
    (out/"SUMMARY.md").write_text("\n".join(md),encoding="utf-8")
    print((out/"SUMMARY.md").read_text(encoding="utf-8"),flush=True)
    print("EXP-CMB-003 COMPLETE: 11 CONDITIONS / 5 FROZEN MODELS / NO TEST",flush=True)


if __name__=="__main__":
    main()
