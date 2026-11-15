#!/usr/bin/env python3
"""EXP-CMB-004: preregistered explanatory BBox-Size vs CoM-Center study.

This is a new 4-Fold (Leave-One-Writer-Out) training comparison. No model/seed/
training-size tuning, and no accuracy threshold determines whether the study
answers its research question. The historical EXP-CMB-001 is a frozen baseline.
See logs/가설B/EXP유001_Geometry_Normalization.md for pre-run criteria.

All images are 28x28 grayscale 0..255. Only original official X_train keys
are read. NEVER access the original Test or professor Test datasets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import affine_transform
import sklearn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[4]
CMB = ROOT / "05_model_optimize" / "MLP" / "Combined"
sys.path.insert(0,str(CMB/"scripts"))
sys.path.insert(0,str(ROOT/"05_model_optimize"/"MLP"/"Normal"/"scripts"))
import exp_cmb002_evidence as evidence
from combined_mlp_writer_cv import load_official, load_writers, new_mlp
from Automatic_Centering_Transformer import center_batch, center_shifts, raw_centers
from pixel_transform import PixelScaler255

LABELS="0123456789ABCDEF"
WRITERS=(41,42,43,44)
METHODS=("bbox_center","bbox_size","com_center","size_then_com")
THRESHOLD=32
BASE=CMB/"results"/"EXP_CMB_001"
PAIRS=(("0","D"),("D","0"),("8","B"),("B","8"))

def csv_save(path,rows,headers=None):
    rows=list(rows)
    assert rows or headers, path
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=headers or list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def image_geometry(batch):
    a=np.asarray(batch,dtype=np.float32).reshape(-1,28,28)
    fg=a>=THRESHOLD
    assert np.all(fg.any(axis=(1,2))),"empty binary foreground"
    xx=fg.any(1)
    yy=fg.any(2)
    xmin=xx.argmax(1)
    xmax=27-xx[:,::-1].argmax(1)
    ymin=yy.argmax(1)
    ymax=27-yy[:,::-1].argmax(1)
    width=xmax-xmin+1
    height=ymax-ymin+1
    bbox_x=(xmax+xmin)/2.0
    bbox_y=(ymax+ymin)/2.0
    m=np.where(fg,a,0.0).astype(np.float64,copy=False)
    mass=m.sum((1,2))
    cx=(m*np.arange(28,dtype=float)[None,None,:]).sum((1,2))/mass
    cy=(m*np.arange(28,dtype=float)[None,:,None]).sum((1,2))/mass
    return dict(width=width,height=height,long_side=np.maximum(width,height),
                bbox_cx=bbox_x,bbox_cy=bbox_y,cx=cx,cy=cy,
                com_offset=np.hypot(cx-13.5,cy-13.5),
                bbox_offset=np.hypot(bbox_x-13.5,bbox_y-13.5),
                ink_ratio=fg.mean((1,2)),
                mass=a.sum((1,2),dtype=np.float64),
                edge_mass=np.sum(a[:,[0,1,26,27],:],axis=(1,2))+
                          np.sum(a[:,:, [0,1,26,27]],axis=(1,2)))

def apply_size(batch,target):
    a=np.asarray(batch,dtype=np.float32).reshape(-1,28,28)
    geo=image_geometry(a)
    factors=float(target)/geo["long_side"].astype(float)
    result=np.empty_like(a)
    for i in range(len(a)):
        inv=1.0/factors[i]
        center=(geo["bbox_cy"][i],geo["bbox_cx"][i])
        output=affine_transform(a[i],np.eye(2)*inv,
                 offset=(center[0]*(1-inv),center[1]*(1-inv)),
                 output_shape=(28,28),order=1,
                 mode="constant",cval=0.0,prefilter=False)
        result[i]=np.clip(output,0,255)
    return result.reshape(-1,784),factors

def apply_method(x,method,target):
    start=time.perf_counter()
    if method=="bbox_center":
        v=center_batch(x,"bbox")
        scales=np.ones(len(x),dtype=float)
    elif method=="bbox_size":
        v,scales=apply_size(x,target)
    elif method=="com_center":
        v=center_batch(x,"mass")
        scales=np.ones(len(x),dtype=float)
    elif method=="size_then_com":
        v,scales=apply_size(x,target)
        v=center_batch(v,"mass")
    elif method=="original":
        v=np.asarray(x,dtype=np.float32).reshape(-1,784)
        scales=np.ones(len(x),dtype=float)
    else:
        raise ValueError(method)
    assert np.isfinite(v).all() and v.min()>=0 and v.max()<=255
    return v.astype(np.float32,copy=False),scales,time.perf_counter()-start

def metrics(y,p,labels=np.arange(16)):
    pr,re,f1,s=precision_recall_fscore_support(y,p,labels=labels,zero_division=0)
    return dict(n=len(y),correct=int(np.sum(y==p)),accuracy=float(np.mean(y==p)),
                macro_precision=float(pr.mean()),macro_recall=float(re.mean()),
                macro_f1=float(f1.mean()))

def transitions(y,p0,p1):
    a=p0==y
    b=p1==y
    return dict(corrected=int(np.sum(~a&b)),
                induced=int(np.sum(a&~b)),
                both_wrong_same=int(np.sum(~a&~b&(p0==p1))),
                both_wrong_changed=int(np.sum(~a&~b&(p0!=p1))),
                both_correct=int(np.sum(a&b)),
                prediction_flip=int(np.sum(p0!=p1)),
                delta_correct=int(b.sum()-a.sum()))

def cmplot(out,label,y,p):
    cm=confusion_matrix(y,p,labels=np.arange(16))
    for kind in ("count","normalized"):
        dat=cm if kind=="count" else 100.0*cm/np.maximum(cm.sum(1,keepdims=True),1)
        fig,ax=plt.subplots(figsize=(9,7.8))
        limit=160 if kind=="count" else 100
        im=ax.imshow(dat,cmap="viridis",vmin=0,vmax=limit,interpolation="nearest")
        fig.colorbar(im,ax=ax,pad=.025,fraction=.04)
        ax.set(xticks=np.arange(16),yticks=np.arange(16),
           xticklabels=list(LABELS),yticklabels=list(LABELS),
           xlabel="Predicted",ylabel="Actual",title=label+" "+kind)
        for i in range(16):
            for j in range(16):
                if cm[i,j]:
                    ax.text(j,i,("%.1f"%dat[i,j] if kind=="normalized" else str(cm[i,j])),
                            ha="center",va="center",fontsize=5,
                            color="black" if dat[i,j]>.55*limit else "white")
        fig.tight_layout()
        fig.savefig(out/f"{label}_{kind}.png",dpi=150)
        plt.close(fig)
    return cm

def geo_summary(name,before,after,scales):
    a,b=image_geometry(before),image_geometry(after)
    startmass=np.maximum(a["mass"],1)
    return dict(sample=name,n=len(before),
      mean_long_before=float(a["long_side"].mean()),mean_long_after=float(b["long_side"].mean()),
      std_long_before=float(a["long_side"].std()),std_long_after=float(b["long_side"].std()),
      mean_com_offset_before=float(a["com_offset"].mean()),
      mean_com_offset_after=float(b["com_offset"].mean()),
      mean_bbox_offset_before=float(a["bbox_offset"].mean()),
      mean_bbox_offset_after=float(b["bbox_offset"].mean()),
      mean_abs_size_ratio_change=float(np.mean(np.abs(scales-1.0))),
      fraction_size_ratio_gt_1_10=float(np.mean(np.abs(scales-1)>=.10)),
      fraction_changed=float(np.mean(np.any(np.abs(np.asarray(before).reshape(-1,784)-after)>.1,axis=1))),
      mean_abs_ink_mass_fraction_change=float(np.mean(np.abs((b["mass"]-a["mass"])/startmass))),
      mean_positive_edge_mass_fraction=float(np.mean(b["edge_mass"]/np.maximum(b["mass"],1))),
      fraction_abs_com_change_over_half_px=float(np.mean(
        np.hypot(b["cx"]-a["cx"],b["cy"]-a["cy"])>.5)))

def find_unexpected_pixel_changes(before,after,name):
    if name=="bbox_size" or name=="size_then_com":
        return
    a=np.asarray(before).reshape(-1,28,28)
    b=np.asarray(after).reshape(-1,28,28)
    if name in ("bbox_center","com_center"):
        # no interpolation in integer center-only methods
        assert np.all(np.isin(np.unique(b),np.unique(a)))
    assert np.isfinite(b).all()

def per_image_explanation(method,held,old_y,old_p,new_p,original,changed,ratio):
    g0=image_geometry(original)
    g1=image_geometry(changed)
    # For all 640 in this writer, score the 4 confusion directions using the
    # competing class fixed by the *true* label, for posthoc research ONLY.
    rows=[]
    for i in range(len(old_y)):
        t=LABELS[int(old_y[i])]
        if t not in ("0","D","8","B"):
            continue
        rival={"0":"D","D":"0","8":"B","B":"8"}[t]
        rows.append({"method":method,"writer":held,"index_in_writer":i,
          "actual":t,"competitor":rival,
          "baseline_pred":LABELS[int(old_p[i])],"new_pred":LABELS[int(new_p[i])],
          "baseline_correct":int(old_p[i]==old_y[i]),
          "new_correct":int(new_p[i]==old_y[i]),
          "transition":"corrected" if old_p[i]!=old_y[i] and new_p[i]==old_y[i] else
                "new_error" if old_p[i]==old_y[i] and new_p[i]!=old_y[i] else
                "persistent_error" if new_p[i]!=old_y[i] else "both_correct",
          "scale_factor":float(ratio[i]),
          "bbox_long_before":int(g0["long_side"][i]),
          "bbox_long_after":int(g1["long_side"][i]),
          "bbox_change":int(g1["long_side"][i]-g0["long_side"][i]),
          "com_x_before":float(g0["cx"][i]),"com_y_before":float(g0["cy"][i]),
          "com_x_after":float(g1["cx"][i]),"com_y_after":float(g1["cy"][i]),
          "com_shift_px":float(np.hypot(g1["cx"][i]-g0["cx"][i],
                                      g1["cy"][i]-g0["cy"][i])),
          "mass_change_ratio":float((g1["mass"][i]-g0["mass"][i])/max(g0["mass"][i],1))})
    return rows

def plot_examples(root,original,transformed,method,held):
    if held!=41 and held!=43:return
    ids=[32,104,11,317]
    fig,axes=plt.subplots(len(ids),2,figsize=(6,11))
    for j,i in enumerate(ids):
        axes[j,0].imshow(original[i].reshape(28,28),cmap="gray",vmin=0,vmax=255)
        axes[j,1].imshow(transformed[i].reshape(28,28),cmap="gray",vmin=0,vmax=255)
        for k in (0,1):axes[j,k].axis("off")
        axes[j,0].set_title(f"Writer {held} #{i} raw")
        axes[j,1].set_title(f"{method}")
    fig.tight_layout()
    fig.savefig(root/f"examples_{method}_writer_{held}.png",dpi=150)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--method",choices=METHODS,required=True)
    args=p.parse_args()
    out=args.out.resolve()
    if (out/"metrics.csv").exists():
        raise RuntimeError("Existing result, do not overwrite: "+str(out))
    out.mkdir(parents=True,exist_ok=True)
    (out/"figures").mkdir(exist_ok=True)
    start=time.perf_counter()
    old_manifest,orig_pred,oldtab,raw_oof,old_labels=evidence.load_inputs()
    official=load_official(ROOT/old_manifest["input_data_paths"]["official"])
    writerdata=load_writers(ROOT/old_manifest["input_data_paths"]["handmade_directory"])
    assert np.array_equal(official["train_idx"],
       np.load(BASE/"official_split_indices.npz")["train_idx"])
    assert np.array_equal(official["val_idx"],
       np.load(BASE/"official_split_indices.npz")["val_idx"])
    goal=int(np.median(image_geometry(official["X_train"])["long_side"]))
    assert 10<=goal<=26
    print("SOURCE CHECKS PASSED / official-only BBox long-side median target",goal,flush=True)
    t0=time.perf_counter()
    off_train,rs_train,_=apply_method(official["X_train"],args.method,goal)
    off_val,rs_val,_=apply_method(official["X_val"],args.method,goal)
    authors={}
    author_ratios={}
    for w in WRITERS:
        authors[w],author_ratios[w],_=apply_method(writerdata[w]["X"],args.method,goal)
        find_unexpected_pixel_changes(writerdata[w]["X"],authors[w],args.method)
    preprocessing_seconds=time.perf_counter()-t0
    diag=[{"method":args.method,**geo_summary("official_train",official["X_train"],off_train,rs_train)},
          {"method":args.method,**geo_summary("official_val",official["X_val"],off_val,rs_val)}]
    for w in WRITERS:
        diag.append({"method":args.method,**geo_summary("writer_"+str(w),writerdata[w]["X"],authors[w],author_ratios[w])})
        plot_examples(out/"figures",writerdata[w]["X"],authors[w],args.method,w)
    csv_save(out/"geometry_diagnostics.csv",diag)

    # Historical OOF is the reference and MUST never be re-fitted or changed.
    base_correct={w:np.array([LABELS.index(oldtab[w,i]["combined_pred"]) for i in range(640)]) for w in WRITERS}
    yref={w:writerdata[w]["y"] for w in WRITERS}
    for w in WRITERS:
        assert np.array_equal(yref[w],
            np.array([LABELS.index(oldtab[w,i]["true_label"]) for i in range(640)]))
    preds={}
    official_rows=[];writer_rows=[];class_rows=[];paired=[];explanation=[]
    fit_rows=[];loss_rows=[];geoms=[];costs=[]
    model_data=[]
    for held in WRITERS:
        startfold=time.perf_counter()
        tr_authors=tuple(w for w in WRITERS if w!=held)
        X=np.concatenate([off_train]+[authors[w] for w in tr_authors])
        y=np.concatenate([official["y_train"]]+[yref[w] for w in tr_authors])
        assert X.shape==(78720,784) and y.shape==(78720,)
        m=new_mlp()
        tfit=time.perf_counter()
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            m.fit(X,y)
        fitsec=time.perf_counter()-tfit
        net=m.named_steps["model"]
        assert m.named_steps["scale_0_1"].__class__ is PixelScaler255
        assert net.get_params()["random_state"]==42
        assert net.n_features_in_==784
        tpred=time.perf_counter()
        pred=m.predict(authors[held]).astype(int)
        pred_seconds=time.perf_counter()-tpred
        preds[held]=pred
        tdev=time.perf_counter()
        official_pred=m.predict(off_val).astype(int)
        dev_seconds=time.perf_counter()-tdev
        v=metrics(yref[held],pred)
        ov=metrics(official["y_val"],official_pred)
        orig=base_correct[held]
        tr=transitions(yref[held],orig,pred)
        writer_rows.append({"method":args.method,"writer":held,**v,**tr,
           "baseline_accuracy":float(np.mean(orig==yref[held])),
           "delta_accuracy_pp":float(100*(v["accuracy"]-np.mean(orig==yref[held])))})
        official_rows.append({"method":args.method,"writer":held,**ov,
           "fit_seconds":fitsec,"predict_seconds":dev_seconds})
        fit_rows.append({"method":args.method,"heldout":held,"train_n":len(X),
            "train_authors":",".join(map(str,tr_authors)),
            "fit_seconds":fitsec,"n_iter":int(net.n_iter_),"train_loss":float(net.loss_),
            "prediction_seconds_640":pred_seconds,
            "official_prediction_seconds_19200":dev_seconds,
            "warnings":" | ".join(str(x.message) for x in warns),
            "fold_wall_seconds":time.perf_counter()-startfold})
        loss_rows.extend({"method":args.method,"writer":held,
            "iteration":i+1,"training_loss":float(loss)} for i,loss in enumerate(net.loss_curve_))
        for j,cl in enumerate(LABELS):
            z=yref[held]==j
            origok=int(np.sum(orig[z]==j))
            newok=int(np.sum(pred[z]==j))
            class_rows.append({"method":args.method,"writer":held,"class":cl,"n":int(z.sum()),
                "old_correct":origok,"new_correct":newok,
                "corrected":int(np.sum((orig[z]!=j)&(pred[z]==j))),
                "induced":int(np.sum((orig[z]==j)&(pred[z]!=j)))})
        # Per image metric includes raw bbox/CoM change and label-aware
        # diagnostic rivals only; NEVER used at transform fitting/inference.
        explanation.extend(per_image_explanation(args.method,held,yref[held],
            orig,pred,writerdata[held]["X"],authors[held],author_ratios[held]))
        for i in range(640):
            paired.append({"method":args.method,"writer":held,"index_in_writer":i,
                "actual":LABELS[int(yref[held][i])],
                "baseline_pred":LABELS[int(orig[i])],
                "prediction":LABELS[int(pred[i])],
                "baseline_correct":int(orig[i]==yref[held][i]),
                "correct":int(pred[i]==yref[held][i])})
        # record margins for focus 0/D/8/B, not class-conditioned decisions
        g=evidence.logits(m,authors[held])
        for j,cl in enumerate(yref[held]):
            actual=LABELS[int(cl)]
            if actual in ("0","D","8","B"):
                rival={"0":"D","D":"0","8":"B","B":"8"}[actual]
                index=next(i for i in range(len(explanation)-160,len(explanation))
                           if explanation[i]["writer"]==held and
                              explanation[i]["index_in_writer"]==j)
                explanation[index]["actual_vs_rival_logit"]=float(g[j,int(cl)]-g[j,LABELS.index(rival)])
                explanation[index]["predicted_rival"]=int(pred[j]==LABELS.index(rival))
        mem=io.BytesIO()
        dump(m,mem,compress=3)
        payload=mem.getvalue()
        model_data.append((f"{args.method}_heldout_{held}_mlp.joblib",payload))
        cmplot(out/"figures",args.method+"_writer_"+str(held),yref[held],pred)
        print(args.method,held,"OOF",v["accuracy"],"Val",ov["accuracy"],
              "fit",round(fitsec,1),flush=True)
        del m,X,y
    whole_y=np.concatenate([yref[w] for w in WRITERS])
    whole_p=np.concatenate([preds[w] for w in WRITERS])
    whole_base=np.concatenate([base_correct[w] for w in WRITERS])
    v=metrics(whole_y,whole_p)
    tr=transitions(whole_y,whole_base,whole_p)
    assert len(whole_y)==2560 and np.sum(whole_base==whole_y)==2080
    overall=[{"method":args.method,**v,**tr,"baseline_accuracy":.8125,
       "delta_accuracy_pp":100*(v["accuracy"]-.8125),
       "preprocessing_seconds":preprocessing_seconds,
       "wall_seconds":time.perf_counter()-start}]
    groups=[]
    for label,z in (("digits",whole_y<10),("letters",whole_y>=10)):
        y=whole_y[z];pa=whole_p[z];ba=whole_base[z]
        labels=np.arange(10) if label=="digits" else np.arange(10,16)
        groups.append({"method":args.method,"group":label,**metrics(y,pa,labels),
                      **transitions(y,ba,pa),"baseline_accuracy":float(np.mean(y==ba))})
    conf=[]
    for actual,rival in PAIRS:
        cl=LABELS.index(actual);cv=LABELS.index(rival)
        for w in WRITERS:
            z=yref[w]==cl
            conf.append({"method":args.method,"writer":w,"actual":actual,"rival":rival,
              "baseline_errors":int(np.sum(base_correct[w][z]==cv)),
              "new_errors":int(np.sum(preds[w][z]==cv))})
    cmplot(out/"figures",args.method+"_pooled",whole_y,whole_p)
    csv_save(out/"metrics.csv",overall)
    csv_save(out/"writer_metrics.csv",writer_rows)
    csv_save(out/"official_val_metrics.csv",official_rows)
    csv_save(out/"class_metrics.csv",class_rows)
    csv_save(out/"group_metrics.csv",groups)
    csv_save(out/"pair_errors.csv",conf)
    csv_save(out/"fit_metrics.csv",fit_rows)
    csv_save(out/"loss_curves.csv",loss_rows)
    csv_save(out/"paired_predictions.csv",paired)
    csv_save(out/"focus_mechanism.csv",explanation)
    fig,ax=plt.subplots(figsize=(9,5))
    for w in WRITERS:
        c=[r for r in loss_rows if r["writer"]==w]
        ax.plot([r["iteration"] for r in c],[r["training_loss"] for r in c],label="heldout "+str(w))
    ax.set(xlabel="MLP iteration",ylabel="Training Loss",title=args.method+" | identical MLP settings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"figures"/"training_loss.png",dpi=170)
    plt.close(fig)
    with zipfile.ZipFile(out/"trained_models.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,rawdata in model_data:z.writestr(name,rawdata)
    summary={"experiment":"EXP-CMB-004","status":"method completed, combined mechanism comparison pending",
      "method":args.method,"target_longside_pixels":goal,"foreground_threshold":THRESHOLD,
      "source_sha256":old_manifest["dataset_sha256"],
      "baseline_exp_cmb_001_oof_correct":2080,
      "handmade_writer_heldout":list(WRITERS),
      "official_train_n":76800,"official_val_n":19200,
      "per_fold_handmade_train_n":1920,"per_fold_validation_n":640,
      "trained_models_sha256":{k:hashlib.sha256(v).hexdigest() for k,v in model_data},
      "scikit_learn_version":sklearn.__version__,"numpy_version":np.__version__,
      "scipy_version":scipy.__version__,
      "no_test_used":True,"author_or_true_label_in_preprocessing":False,
      "method_explanatory_success_requires_not_accuracy_threshold":True,
      "training_seconds_total":sum(x["fit_seconds"] for x in fit_rows),
      "preprocess_seconds_total":preprocessing_seconds,"total_seconds":time.perf_counter()-start}
    (out/"run_manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("COMPLETED",args.method,overall[0],flush=True)

if __name__=="__main__":
    main()
