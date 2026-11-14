#!/usr/bin/env python3
"""EXP유003 Spatial Distribution & MLP Response Analysis (diagnostic, no tuning).

Uses the EXACT frozen Normal MLP from completed EXP유002 Stage 3.
Studies Official Train/Validation + current Handmade geometry and relates
paired transformations to prediction flips, hidden response and fixed-logit
margin Integrated Gradients. Official Test is NEVER loaded.

Outputs are GitHub Actions artifacts for review; this script does not commit
or publish results. Features/ground-truth labels are used for investigation
only, NOT for any inference-time preprocessing rule.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
import platform
import sys
import time
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import wasserstein_distance
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from Pixcel_Attribution import (fixed_split, DEFAULT_DATA, official_indices,
    prior_official_cm, prior_handmade, read_handmade, LABELS)
from Input_Sensitivity import transform, VARIANTS, get_output, margin, select_focus

STAGE3_DIR = ROOT / "05_model_optimize" / "MLP" / "Normal" / "results" / "Input_Sensitivity"
MODEL_PATH = STAGE3_DIR / "normal_mlp_refit_stage3.joblib"
PRIOR_METADATA = STAGE3_DIR / "stage3_run_metadata.json"
COMPARISON = ("original", "shift_L1", "shift_R1", "shift_D1", "scale_090", "scale_095")
SCATTER_CLASSES = ("5", "9", "B", "F")
MAIN_THRESHOLD = 32
THRESHOLDS = (32, 64, 128)


def csvwrite(file, rows):
    if not rows:
        return
    with file.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def metrics(images: np.ndarray, threshold: int = 32):
    """Fixed absolute grayscale threshold; vectorized (n,28,28)."""
    a = images.reshape(-1, 28, 28).astype(np.float64, copy=False)
    m = a >= threshold
    active = m.sum(axis=(1, 2)).astype(np.int32)
    valid = active > 0
    xn = m.any(axis=1)
    yn = m.any(axis=2)
    xmin = np.where(valid, np.argmax(xn, axis=1), -1)
    xmax = np.where(valid, 27 - np.argmax(xn[:, ::-1], axis=1), -1)
    ymin = np.where(valid, np.argmax(yn, axis=1), -1)
    ymax = np.where(valid, 27 - np.argmax(yn[:, ::-1], axis=1), -1)
    w = np.where(valid, xmax - xmin + 1, np.nan).astype(float)
    h = np.where(valid, ymax - ymin + 1, np.nan).astype(float)
    wt = a * m
    mass = wt.sum(axis=(1, 2))
    mass_safe = np.maximum(mass, 1e-30)
    cx = (wt * np.arange(28)[None, None, :]).sum(axis=(1, 2)) / mass_safe
    cy = (wt * np.arange(28)[None, :, None]).sum(axis=(1, 2)) / mass_safe
    cx = np.where(valid, cx, np.nan)
    cy = np.where(valid, cy, np.nan)
    return {
      "cx":cx, "cy":cy,
      "bbox_cx":np.where(valid,(xmin+xmax)/2,np.nan),
      "bbox_cy":np.where(valid,(ymin+ymax)/2,np.nan),
      "width":w, "height":h,
      "left_margin":np.where(valid,xmin,np.nan).astype(float),
      "right_margin":np.where(valid,27-xmax,np.nan).astype(float),
      "top_margin":np.where(valid,ymin,np.nan).astype(float),
      "bottom_margin":np.where(valid,27-ymax,np.nan).astype(float),
      "aspect":w/np.maximum(h,1),
      "ink_frac":active/784,
      "mass":mass,
      "valid":valid,
    }


def shape_features(x: np.ndarray, thresholds=THRESHOLDS):
    parts = {t: metrics(x, t) for t in thresholds}
    n = len(x)
    result = {}
    for k,v in parts[MAIN_THRESHOLD].items():
        result[k] = v
    for t in thresholds:
        if t == MAIN_THRESHOLD:
            continue
        for key in ("cx","cy","bbox_cx","bbox_cy","width","height","ink_frac"):
            result[f"{key}_t{t}"] = parts[t][key]
        result[f"valid_t{t}"] = parts[t]["valid"]
    assert all(len(v)==n for v in result.values())
    return result


def geometry_matrix(feats):
    return np.column_stack([feats[x] for x in ("cx","cy","width","height")])


def reference_stats(xtrain, ytrain):
    center=np.zeros((16,4),dtype=np.float64)
    scale=np.zeros((16,4),dtype=np.float64)
    rows=[]
    a=geometry_matrix(xtrain)
    for i,c in enumerate(LABELS):
        v=a[(ytrain==i) & np.isfinite(a).all(axis=1)]
        if len(v)<100:raise RuntimeError(f"Insufficient official train reference for class {c}")
        med=np.median(v,axis=0)
        q1,q3=np.percentile(v,[25,75],axis=0)
        robust=np.maximum((q3-q1)/1.349,0.75)
        center[i],scale[i]=med,robust
        rows.append({
            "class":c,"n":len(v),
            **{f"median_{k}":float(med[j]) for j,k in enumerate(("cx","cy","width","height"))},
            **{f"scale_{k}":float(robust[j]) for j,k in enumerate(("cx","cy","width","height"))},
        })
    return center,scale,rows


def reference_distance(f,y,center,scale):
    v=geometry_matrix(f)
    with np.errstate(invalid="ignore"):
        d=np.sqrt(np.square((v-center[y])/scale[y]).sum(axis=1))
    return d


def describe(f,y,domain,writer=-1,aug=None):
    data=[]
    for cls in ["ALL"]+list(LABELS):
        mask=np.ones(len(y),dtype=bool) if cls=="ALL" else (y==LABELS.index(cls))
        valid=mask & f["valid"]
        row={"dataset":domain,"writer":writer,"class":cls,
             "n":int(mask.sum()),"valid_n":int(valid.sum())}
        for key in ("cx","cy","bbox_cx","bbox_cy","width","height","ink_frac",
                    "left_margin","right_margin","top_margin","bottom_margin"):
            v=f[key][valid]
            row[f"{key}_mean"]=float(np.mean(v)) if len(v) else None
            row[f"{key}_median"]=float(np.median(v)) if len(v) else None
            row[f"{key}_p10"]=float(np.percentile(v,10)) if len(v) else None
            row[f"{key}_p90"]=float(np.percentile(v,90)) if len(v) else None
        data.append(row)
    return data


def write_features(file, domain, x, y, writer, index, augmented=None, batch=1024):
    """One compact row per source image, with threshold-robustness columns."""
    fields=["dataset","writer","sample_index","class","augmented"]
    keys=list(shape_features(x[:1]).keys())
    write_header = not file.exists() or file.stat().st_size == 0
    with gzip.open(file,"at",newline="",encoding="utf-8") as stream:
        csvwriter=csv.DictWriter(stream,fieldnames=fields+keys)
        if write_header:
            csvwriter.writeheader()
        for beg in range(0,len(y),batch):
            end=min(beg+batch,len(y))
            f=shape_features(x[beg:end])
            for j in range(end-beg):
                row={"dataset":domain, "writer":int(writer[beg+j]),
                     "sample_index":int(index[beg+j]),
                     "class":LABELS[int(y[beg+j])],
                     "augmented":int(augmented[beg+j]) if augmented is not None else ""}
                for k in keys:
                    val=f[k][j]
                    row[k]=(float(val) if np.isfinite(val) else "")
                csvwriter.writerow(row)


def read_stage3_summary():
    file=STAGE3_DIR/"stage3_sensitivity_summary.csv"
    with file.open(encoding="utf-8-sig",newline="") as f:
        return {(r["domain"],r["variant"]):r for r in csv.DictReader(f)
                if r["group_type"]=="all"}


def top_hidden(model,raw):
    net=model.named_steps["model"]
    a=np.asarray(raw,dtype=np.float32)/np.float32(255)
    return np.maximum(a@net.coefs_[0]+net.intercepts_[0],0.0)


def hidden_change(old,new):
    norm0=np.linalg.norm(old,axis=1)
    norm1=np.linalg.norm(new,axis=1)
    similarity=(old*new).sum(axis=1)/np.maximum(norm0*norm1,1e-8)
    relative_l2=np.linalg.norm(new-old,axis=1)/np.maximum(norm0,1e-8)
    switched=np.mean((old>0)!=(new>0),axis=1)
    return similarity,relative_l2,switched


def ig_true_margin(net, raw, true_cls, comparison_cls, steps=128):
    """Signed IG of the SAME fixed true-minus-competitor logit objective."""
    assert true_cls != comparison_cls
    x=np.asarray(raw,dtype=np.float64).reshape(784)/255.0
    w0=np.asarray(net.coefs_[0],dtype=np.float64)
    b0=np.asarray(net.intercepts_[0],dtype=np.float64)
    w1=np.asarray(net.coefs_[1],dtype=np.float64)
    bias1=np.asarray(net.intercepts_[1],dtype=np.float64)
    diff=w1[:,true_cls]-w1[:,comparison_cls]
    alpha=(np.arange(steps,dtype=np.float64)+0.5)/steps
    z=(alpha[:,None]*x[None,:])@w0+b0
    grad=(((z>0)*diff[None,:])@w0.T).mean(axis=0)
    ig=x*grad
    def objective(v):
        hidden=np.maximum(v@w0+b0,0)
        return hidden@diff+bias1[true_cls]-bias1[comparison_cls]
    delta=float(objective(x)-objective(np.zeros_like(x)))
    residual=float(ig.sum()-delta)
    return ig.reshape(28,28),delta,residual


def focus_analysis(focus,model, variants,out):
    net=model.named_steps["model"]
    records=[]
    prepared=[]
    vmap={v.name:v for v in variants}
    for i, ex in enumerate(focus):
        original=ex["pixels"].reshape(1,784)
        z0,p0=get_output(model,original)
        h0=top_hidden(model,original)
        assert int(z0.argmax(axis=1)[0])==ex["original_pred"]
        y=int(ex["true"])
        alternate = int(ex["original_pred"])
        if alternate==y:
            competitors=z0[0].copy()
            competitors[y]=-np.inf
            alternate=int(competitors.argmax())
        for v in variants:
            altered=transform(original,v)
            logits,prob=get_output(model,altered)
            h=top_hidden(model,altered)
            cs,rel,sw=hidden_change(h0,h)
            pred=int(logits[0].argmax())
            geom=metrics(altered)
            prepared.append((i,ex,v,altered,prob,logits))
            records.append({
                "focus_id":i,"domain":ex["domain"],"writer":int(ex["writer"]),
                "index":int(ex["index"]), "true":LABELS[y],
                "original_pred":LABELS[ex["original_pred"]],
                "original_correct":int(y==ex["original_pred"]),
                "new_pred":LABELS[pred],"new_correct":int(y==pred),
                "variant":v.name,
                "p_true":float(prob[0,y]),
                "p_fixed_alternate":float(prob[0,alternate]),
                "fixed_true_minus_alternate_logit":float(logits[0,y]-logits[0,alternate]),
                "hidden_cosine_to_original":float(cs[0]),
                "hidden_relative_l2":float(rel[0]),
                "hidden_changed_active_fraction":float(sw[0]),
                "cx":float(geom["cx"][0]),"cy":float(geom["cy"][0]),
                "width":float(geom["width"][0]),"height":float(geom["height"][0]),
            })
    csvwrite(out/"focus_hidden_responses.csv",records)
    # Balanced qualitative audit: include original errors, corrected errors,
    # stable errors and formerly correct examples. Do not cherry-pick only successes.
    chosen=[]
    for direction in (("5","F"),("B","8"),("9","F"),("9","4"),
                      ("0","D"),("7","F")):
        matches=[(i,x) for i,x in enumerate(focus)
                 if x["domain"]=="Handmade" and LABELS[x["true"]]==direction[0]
                 and LABELS[x["original_pred"]]==direction[1]]
        if matches:
            chosen.append(matches[0][0])
            if len(matches)>1:chosen.append(matches[1][0])
    for cls in ("5","9","B","F","8","4"):
        matches=[i for i,x in enumerate(focus)
                 if x["domain"]=="Handmade" and LABELS[x["true"]]==cls
                 and x["true"]==x["original_pred"]]
        if matches:chosen.append(matches[0])
    chosen=list(dict.fromkeys(chosen))[:18]
    ig_variants=("original","shift_R1","shift_D1","scale_095")
    ig_records=[]
    image_arrays=[]
    ig_arrays=[]
    for i in chosen:
        ex=focus[i]
        y=int(ex["true"])
        alternate=int(ex["original_pred"])
        if alternate==y:
            logits,_=get_output(model,ex["pixels"].reshape(1,784))
            rivals=logits[0].copy()
            rivals[y]=-np.inf
            alternate=int(rivals.argmax())
        for name in ig_variants:
            raw=transform(ex["pixels"].reshape(1,784),vmap[name])
            ig,delta,residual=ig_true_margin(net,raw,y,alternate)
            ig_records.append({
                "focus_id":i, "domain":ex["domain"],"writer":ex["writer"],
                "index":ex["index"], "true":LABELS[y],"fixed_competitor":LABELS[alternate],
                "variant":name,"objective":"true_minus_fixed_competitor_logit",
                "ig_sum":float(ig.sum()),"objective_delta_from_black":delta,
                "completeness_residual":residual,"ig_absolute_sum":float(np.abs(ig).sum()),
            })
            image_arrays.append(raw.reshape(28,28).astype(np.uint8))
            ig_arrays.append(ig.astype(np.float32))
    csvwrite(out/"focus_ig_completeness.csv",ig_records)
    np.savez_compressed(out/"focus_ig_maps.npz",
                        pixels=np.stack(image_arrays),signed_IG=np.stack(ig_arrays))
    figdir=out/"focus_ig_grids"
    figdir.mkdir(exist_ok=True)
    for page in range(0,len(chosen),4):
        picks=chosen[page:page+4]
        fig,axs=plt.subplots(len(picks),len(ig_variants)*2,
                             figsize=(18,3.2*len(picks)),squeeze=False)
        for ridx,i in enumerate(picks):
            ex=focus[i]
            for j,name in enumerate(ig_variants):
                k=(chosen.index(i))*len(ig_variants)+j
                raw=image_arrays[k]
                ig=ig_arrays[k]
                bound=max(float(np.max(np.abs(ig))),1e-8)
                a=axs[ridx,2*j]
                b=axs[ridx,2*j+1]
                a.imshow(raw,cmap="gray",vmin=0,vmax=255)
                b.imshow(ig,cmap="RdBu_r",vmin=-bound,vmax=bound)
                a.set_title(f"{LABELS[ex['true']]} / orig->{LABELS[ex['original_pred']]}\n{name} image",fontsize=8)
                b.set_title("IG: true - fixed rival",fontsize=8)
                a.set_xticks([]);a.set_yticks([])
                b.set_xticks([]);b.set_yticks([])
        fig.tight_layout()
        fig.savefig(figdir/f"focus_{page//4+1:02d}.png",dpi=115)
        plt.close(fig)
    return records,ig_records


def run(args):
    started=time.monotonic()
    assert MAIN_THRESHOLD in THRESHOLDS
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    with PRIOR_METADATA.open(encoding="utf-8") as f:
        oldmeta=json.load(f)
    assert oldmeta["total_prediction_rows"]==456960
    assert oldmeta["baseline_validation_cm_L1_difference"]==0
    assert oldmeta["baseline_handmade_prediction_difference"]==0
    model=joblib.load(MODEL_PATH)  # Our own verified GitHub Actions saved checkpoint.
    print("Loaded VERIFIED frozen Stage 3 Normal MLP checkpoint",flush=True)

    xtr,ytr,xva,yva,off_sha=fixed_split(DEFAULT_DATA)
    vaidx=official_indices(DEFAULT_DATA,yva)
    with np.load(DEFAULT_DATA,allow_pickle=False) as z:
        aug=z["train_is_augmented"]
    official_index=np.arange(96000)
    train_idx=np.setdiff1d(official_index,vaidx,assume_unique=True)
    assert len(train_idx)==76800
    training_aug=aug[train_idx]
    xhm,yhm,whm,ih=read_handmade()
    pva=model.predict(xva)
    phm=model.predict(xhm)
    official_l1=int(np.abs(confusion_matrix(yva,pva,labels=np.arange(16))
                       -prior_official_cm()).sum())
    previous=prior_handmade()
    handmade_difference=sum(previous[(int(w),int(idx))] !=
                            (LABELS[int(a)],LABELS[int(b)])
                            for w,idx,a,b in zip(whm,ih,yhm,phm))
    assert official_l1==0 and handmade_difference==0
    assert abs(accuracy_score(yva,pva)-oldmeta["original_accuracy"]["OfficialVal"])<1e-12
    assert abs(accuracy_score(yhm,phm)-oldmeta["original_accuracy"]["Handmade"])<1e-12
    print("Reused frozen model: both prior confusion/prediction validations exactly match",flush=True)

    # All geometry measurements are computed on RAW source input, not transformed.
    print("Extracting Official Train/Val and 4-writer geometry",flush=True)
    ftr=shape_features(xtr)
    fva=shape_features(xva)
    fhm=shape_features(xhm)
    ftr_original={k:v[~training_aug.astype(bool)] for k,v in ftr.items()}
    yt_original=ytr[~training_aug.astype(bool)]
    ftr_aug={k:v[training_aug.astype(bool)] for k,v in ftr.items()}
    yt_aug=ytr[training_aug.astype(bool)]
    statrows=[]
    statrows+=describe(ftr,ytr,"OfficialTrain")
    statrows+=describe(ftr_original,yt_original,"OfficialTrain_Original")
    statrows+=describe(ftr_aug,yt_aug,"OfficialTrain_Augmented")
    statrows+=describe(fva,yva,"OfficialVal")
    statrows+=describe(fhm,yhm,"Handmade")
    for writer in (41,42,43,44):
        m=whm==writer
        statrows+=describe({k:v[m] for k,v in fhm.items()},yhm[m],"Handmade",writer)
    csvwrite(out/"geometry_distribution_summary.csv",statrows)
    center,scale,refrows=reference_stats(ftr,ytr)
    csvwrite(out/"official_class_references.csv",refrows)

    # Source images are 96,000 Training data (76,800 model fit + 19,200 Val).
    rawfeature=out/"all_image_geometry.csv.gz"
    write_features(rawfeature,"OfficialTrain",xtr,ytr,
                   np.full(len(ytr),-1),train_idx,training_aug)
    write_features(rawfeature,"OfficialVal",xva,yva,
                   np.full(len(yva),-1),vaidx)
    write_features(rawfeature,"Handmade",xhm,yhm,whm,ih)
    print(f"Saved per-source geometry for {len(ytr)+len(yva)+len(yhm)} images",flush=True)

    # Class-and-writer-matched descriptive comparisons, without asserting
    # independent effective samples for source-related augmented images.
    pairs=[]
    for class_idx,c in enumerate(LABELS):
        for name,mask,stats in [
            ("Handmade",yhm==class_idx,fhm),
            ("Handmade_41",(yhm==class_idx)&(whm==41),fhm),
            ("Handmade_42",(yhm==class_idx)&(whm==42),fhm),
            ("Handmade_43",(yhm==class_idx)&(whm==43),fhm),
            ("Handmade_44",(yhm==class_idx)&(whm==44),fhm),
            ("OfficialVal",yva==class_idx,fva)
        ]:
            trmask=ytr==class_idx
            row={"class":c,"target":name,"reference":"OfficialTrain",
                 "n_reference":int(trmask.sum()),"n_target":int(mask.sum())}
            for key in ("cx","cy","width","height","bbox_cx","bbox_cy"):
                trainvec=ftr[key][trmask]
                targetvec=stats[key][mask]
                trainvec=trainvec[np.isfinite(trainvec)]
                targetvec=targetvec[np.isfinite(targetvec)]
                row[f"{key}_median_difference"]=float(np.median(targetvec)-np.median(trainvec))
                row[f"{key}_wasserstein"]=float(wasserstein_distance(trainvec,targetvec))
            pairs.append(row)
    csvwrite(out/"per_class_geometry_differences.csv",pairs)

    variants={x.name:x for x in VARIANTS}
    focus=select_focus(xva,yva,pva,vaidx,xhm,yhm,phm,whm,ih)
    selected=[variants[name] for name in COMPARISON]
    focusrows,igrows=focus_analysis(focus,model,selected,out)
    summary3=read_stage3_summary()

    # Controlled comparisons use a fixed model and reference distribution.
    # Ground-truth class reference is only for POST-HOC analysis.
    condition_summary=[]
    group_summary=[]
    relation_summary=[]
    paired_count=0
    with gzip.open(out/"paired_geometry_model_response.csv.gz","wt",newline="",encoding="utf-8") as f:
        fields=[
          "dataset","writer","image_index","true","original_pred","changed_pred",
          "variant","correct_before","correct_after","outcome",
          "old_class_ref_distance","new_class_ref_distance","distance_change",
          "closer_to_official_train_class_geometry",
          "original_cx","new_cx","original_cy","new_cy",
          "original_width","new_width","original_height","new_height",
          "true_margin_before","true_margin_after","true_margin_change",
          "hidden_cosine","hidden_relative_l2","hidden_active_switch_fraction",
        ]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        domains=[
          ("OfficialVal",xva,yva,np.full(len(yva),-1),vaidx,pva,fva),
          ("Handmade",xhm,yhm,whm,ih,phm,fhm)
        ]
        for domain,raw,y,writers,indices,orig_pred,fo in domains:
            z0,_=get_output(model,raw)
            assert np.array_equal(z0.argmax(axis=1),orig_pred)
            m0=margin(z0,y)
            h0=top_hidden(model,raw)
            d0=reference_distance(fo,y,center,scale)
            for v in selected:
                n=len(y)
                pred=np.empty(n,dtype=np.int64)
                margins=np.empty(n,dtype=np.float64)
                hcos=np.empty(n,dtype=np.float64)
                hl2=np.empty(n,dtype=np.float64)
                switches=np.empty(n,dtype=np.float64)
                g={k:np.empty(n,dtype=np.float64)
                   for k in ("cx","cy","width","height")}
                for left in range(0,n,args.batch):
                    right=min(left+args.batch,n)
                    xx=transform(raw[left:right],v)
                    z,_=get_output(model,xx)
                    pred[left:right]=z.argmax(axis=1)
                    margins[left:right]=margin(z,y[left:right])
                    hc=top_hidden(model,xx)
                    hc0,hrel,hsw=hidden_change(h0[left:right],hc)
                    hcos[left:right]=hc0
                    hl2[left:right]=hrel
                    switches[left:right]=hsw
                    gf=metrics(xx,MAIN_THRESHOLD)
                    for k in g:g[k][left:right]=gf[k]
                d1=reference_distance(g,y,center,scale)
                correction=(orig_pred!=y)&(pred==y)
                induction=(orig_pred==y)&(pred!=y)
                stuck=(orig_pred!=y)&(pred!=y)
                outcome=np.where(correction,"corrected",
                          np.where(induction,"induced",
                          np.where(stuck,"still_wrong","still_correct")))
                closer=d1<d0-1e-5
                for i in range(n):
                    w.writerow({
                        "dataset":domain,"writer":int(writers[i]),
                        "image_index":int(indices[i]),"true":LABELS[y[i]],
                        "original_pred":LABELS[orig_pred[i]],"changed_pred":LABELS[pred[i]],
                        "variant":v.name,
                        "correct_before":int(orig_pred[i]==y[i]),
                        "correct_after":int(pred[i]==y[i]),"outcome":outcome[i],
                        "old_class_ref_distance":float(d0[i]),
                        "new_class_ref_distance":float(d1[i]),
                        "distance_change":float(d1[i]-d0[i]),
                        "closer_to_official_train_class_geometry":int(closer[i]),
                        "original_cx":float(fo["cx"][i]),"new_cx":float(g["cx"][i]),
                        "original_cy":float(fo["cy"][i]),"new_cy":float(g["cy"][i]),
                        "original_width":float(fo["width"][i]),"new_width":float(g["width"][i]),
                        "original_height":float(fo["height"][i]),"new_height":float(g["height"][i]),
                        "true_margin_before":float(m0[i]),
                        "true_margin_after":float(margins[i]),
                        "true_margin_change":float(margins[i]-m0[i]),
                        "hidden_cosine":float(hcos[i]),
                        "hidden_relative_l2":float(hl2[i]),
                        "hidden_active_switch_fraction":float(switches[i]),
                    })
                paired_count+=n
                prev=summary3[(domain,v.name)]
                original_acc=float(np.mean(orig_pred==y))
                acc=float(np.mean(pred==y))
                if abs(acc-float(prev["accuracy"]))>1e-9:
                    raise RuntimeError(f"Stage3 accuracy mismatch: {domain} {v.name} {acc} vs {prev['accuracy']}")
                if int(correction.sum())!=int(prev["corrections"]) or int(induction.sum())!=int(prev["induced_errors"]):
                    raise RuntimeError(f"Stage3 correction/induction mismatch for {domain} {v.name}")
                condition_summary.append({
                    "dataset":domain,"variant":v.name,
                    "n":n,"original_accuracy":original_acc,
                    "accuracy":acc,"corrected":int(correction.sum()),
                    "induced":int(induction.sum()),
                    "mean_hidden_relative_l2":float(hl2.mean()),
                    "mean_hidden_cosine":float(hcos.mean()),
                    "mean_hidden_active_switch_fraction":float(switches.mean()),
                    "mean_geometry_distance_change":float(np.nanmean(d1-d0)),
                    "mean_true_logit_margin_change":float(np.mean(margins-m0)),
                    "n_closer_class_geometry":int(closer.sum()),
                    "fraction_closer_class_geometry":float(closer.mean()),
                })
                for side in ("closer","not_closer"):
                    mm=closer if side=="closer" else ~closer
                    for key,subset in [
                         ("ALL",mm),
                         ("original_wrong",mm&(orig_pred!=y)),
                         ("original_correct",mm&(orig_pred==y))
                    ]:
                        m=int(subset.sum())
                        if m==0:continue
                        relation_summary.append({
                            "dataset":domain,"variant":v.name,
                            "distance_direction":side,"original_status":key,
                            "n":m,"corrected":int((correction&subset).sum()),
                            "induced":int((induction&subset).sum()),
                            "correction_fraction":float((correction&subset).sum()/m),
                            "induction_fraction":float((induction&subset).sum()/m),
                            "mean_delta_true_margin":float((margins-m0)[subset].mean()),
                            "mean_hidden_relative_l2":float(hl2[subset].mean()),
                            "mean_ref_distance_change":float(np.nanmean((d1-d0)[subset])),
                        })
                for group_name,group_mask in [
                  ("digits",y<10),("letters",y>=10),
                  *([(f"writer_{ww}",writers==ww) for ww in (41,42,43,44)]
                    if domain=="Handmade" else [])
                ]:
                    group_summary.append({
                       "dataset":domain,"variant":v.name,"group":group_name,
                       "n":int(group_mask.sum()),"corrected":int((correction&group_mask).sum()),
                       "induced":int((induction&group_mask).sum()),
                       "mean_distance_change":float(np.nanmean((d1-d0)[group_mask])),
                       "mean_hidden_relative_l2":float(hl2[group_mask].mean()),
                    })
                print(f"{domain} {v.name}: accuracy={acc:.5f}, "
                      f"correction={int(correction.sum())}, "
                      f"induced={int(induction.sum())}, "
                      f"hidden_relative_l2={hl2.mean():.4f}",flush=True)
    assert paired_count==21760*len(COMPARISON)
    csvwrite(out/"spatial_response_summary.csv",condition_summary)
    csvwrite(out/"spatial_response_groups.csv",group_summary)
    csvwrite(out/"geometry_proximity_vs_outcome.csv",relation_summary)

    # Diagnostic figures: descriptive, with no fitted regression or causal claims.
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),squeeze=False)
    for j,cls in enumerate(("5","B")):
        target=LABELS.index(cls)
        axs[0,j].scatter(ftr["cx"][ytr==target][::15],ftr["cy"][ytr==target][::15],
                   s=5,alpha=.22,label="Official Train sampled")
        axs[0,j].scatter(fhm["cx"][yhm==target],fhm["cy"][yhm==target],
                   s=16,alpha=.55,label="Handmade")
        axs[0,j].scatter([13.5],[13.5],marker="+",s=80,c="black")
        axs[0,j].set_xlim(0,27);axs[0,j].set_ylim(27,0)
        axs[0,j].set_xlabel("Mass centroid X (pixel)")
        axs[0,j].set_ylabel("Mass centroid Y (pixel)")
        axs[0,j].set_title(f"Class {cls} — geometry by domain")
        axs[0,j].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/"center_scatter.png",dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10.5,7.5),squeeze=False)
    for j,c in enumerate(("5","9","B","F")):
        ax=axs.flat[j];cid=LABELS.index(c)
        ax.hist(ftr["width"][ytr==cid],bins=np.arange(.5,29.5),density=True,alpha=.45,label="Official Train")
        ax.hist(fhm["width"][yhm==cid],bins=np.arange(.5,29.5),density=True,alpha=.45,label="Handmade")
        ax.set_title(f"{c} — Bounding Box Width");ax.set_xlabel("Width (pixels)")
        ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/"bbox_width_distributions.png",dpi=150);plt.close(fig)
    byclass=[r for r in pairs if r["target"]=="Handmade"]
    fig,axs=plt.subplots(2,1,figsize=(11.7,7.2),squeeze=False)
    for i,key in enumerate(("cx","width")):
        axs[i,0].bar(range(16),[r[f"{key}_median_difference"] for r in byclass])
        axs[i,0].set_xticks(range(16),list(LABELS))
        axs[i,0].axhline(0,color="gray",linewidth=.8)
        axs[i,0].set_ylabel(f"Handmade minus Official Train\nmedian {key} (px)")
    fig.tight_layout();fig.savefig(out/"per_class_spatial_difference.png",dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11.2,4.3),squeeze=False)
    for j,dataset in enumerate(("OfficialVal","Handmade")):
        vals=[r for r in condition_summary if r["dataset"]==dataset and r["variant"]!="original"]
        labels=[v["variant"] for v in vals]
        axs[0,j].bar(range(len(vals)),[v["mean_hidden_relative_l2"] for v in vals])
        axs[0,j].set_xticks(range(len(vals)),labels,rotation=45,ha="right",fontsize=8)
        axs[0,j].set_ylabel("Mean relative L2 change in 100 hidden activations")
        axs[0,j].set_title(dataset)
    fig.tight_layout();fig.savefig(out/"hidden_response_by_transform.png",dpi=150);plt.close(fig)

    valid_thres={}
    for domain,geom in [("OfficialTrain",ftr),("OfficialVal",fva),("Handmade",fhm)]:
        valid_thres[domain]={str(t):int(np.sum(geom["valid"] if t==32 else geom[f"valid_t{t}"]))
                             for t in THRESHOLDS}
    meta={
      "study":"EXP유003 Spatial Distribution & MLP Response",
      "stage3_model_path":str(MODEL_PATH.relative_to(ROOT)),
      "model_fitted_here":False,"official_test_read":False,"handmade_used_for_fit":False,
      "official_npz_sha256":off_sha,
      "number_of_source_images":int(len(ytr)+len(yva)+len(yhm)),
      "train":int(len(ytr)),"train_original":int((~training_aug.astype(bool)).sum()),
      "train_augmented":int(training_aug.sum()),
      "validation":int(len(yva)),"handmade":int(len(yhm)),
      "valid_feature_measurements":valid_thres,
      "baseline_official_accuracy":float(accuracy_score(yva,pva)),
      "baseline_handmade_accuracy":float(accuracy_score(yhm,phm)),
      "baseline_official_CM_difference":official_l1,
      "baseline_handmade_prediction_difference":handmade_difference,
      "geometry_thresholds":THRESHOLDS,
      "geometry_reference":"per-class official model training median, robust IQR standardization",
      "target_labels_used_only_for_posthoc_analysis":True,
      "paired_conditions":COMPARISON,
      "paired_response_rows":paired_count,
      "focus_samples":len(focus),"focus_response_rows":len(focusrows),
      "focus_ig_maps":len(igrows),
      "software":{"python":platform.python_version(),"sklearn":sklearn.__version__,
                  "numpy":np.__version__,"scipy":scipy.__version__},
      "elapsed_sec":time.monotonic()-started,
    }
    (out/"run_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),
                                          encoding="utf-8")

    source_stats={r["dataset"]:r for r in statrows if r["class"]=="ALL"
                  and (r["dataset"]!="Handmade" or r["writer"]==-1)}
    lines=[
      "# EXP유003 Spatial Distribution & MLP Response — Actual execution summary",
      "",
      f"- Frozen Stage 3 MLP reused, no fitting. Official Test NOT read.",
      f"- Official Train {len(ytr):,} (raw {meta['train_original']:,}, augmented {meta['train_augmented']:,}), "
      f"Validation {len(yva):,}, Handmade {len(yhm):,}.",
      f"- Historical predictions verified: CM-L1={official_l1}, Handmade mismatch={handmade_difference}.",
      f"- Fixed geometry threshold={MAIN_THRESHOLD}; robustness thresholds={list(THRESHOLDS)}.",
      f"- Paired input analyses: {paired_count:,} records, same MLP; focus {len(focus)} images, IG maps {len(igrows)}.",
      "",
      "## Measured geometry (all classes together, exploratory only)",
      "| Data | Center X median | Center Y median | Bounding box width median | height median |",
      "|---|---:|---:|---:|---:|",
    ]
    for src in ("OfficialTrain","OfficialVal","Handmade"):
        r=source_stats[src]
        lines.append(f"| {src} | {r['cx_median']:.3f} | {r['cy_median']:.3f} | "
                     f"{r['width_median']:.3f} | {r['height_median']:.3f} |")
    lines.extend([
      "",
      "## Paired same-image response and errors",
      "| Dataset | Change | Accuracy | Corrected | Newly wrong | Hidden relative L2 | Mean class-distance change |",
      "|---|---|---:|---:|---:|---:|---:|",
    ])
    for rec in condition_summary:
        if rec["variant"]!="original":
            lines.append(f"| {rec['dataset']} | {rec['variant']} | {rec['accuracy']:.3%} "
                         f"| {rec['corrected']} | {rec['induced']} "
                         f"| {rec['mean_hidden_relative_l2']:.3f} "
                         f"| {rec['mean_geometry_distance_change']:+.3f} |")
    lines.extend([
      "",
      "## Interpretation restrictions",
      "- All geometry and prediction associations are observational, even when actual perturbations are applied.",
      "- The perturbations simultaneously change many pixel coordinates and sometimes erase/reshape strokes.",
      "- Class-reference geometry uses TRUE labels for audit only; it is not an allowable deployment rule.",
      "- Grouped augmentation source samples are not independent; do not claim significance from raw counts.",
      "- Integrated Gradients compare the SAME true-vs-original-competitor logit objective within each example.",
      "- Zero-image IG baseline makes dark-background attribution zero by construction.",
      "- This experiment did not select a normalizing transformation or improve a trained model.",
      "",
      "See CSV/PNG/NPZ files for exact counts, classes, writers and fixed-model activations.",
    ])
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"EXP003 COMPLETE: {paired_count} paired rows; {len(focusrows)} focus responses, "
          f"{len(igrows)} IG maps",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("spatial_response_results"))
    parser.add_argument("--batch",type=int,default=1024)
    a=parser.parse_args()
    run(a)


if __name__=="__main__":
    main()
