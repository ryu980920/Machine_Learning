#!/usr/bin/env python3
"""EXP-CMB-002: deterministic, train-free, author-held-out error evidence.

This script analyses the completed EXP-CMB-001 models and 2,560 OOF predictions.
It must NEVER read original X_test/y_test, modify baseline files, train a model,
or infer a new deployable preprocessing rule. Outputs are research diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt
import sklearn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[4]
DIR = ROOT / "05_model_optimize" / "MLP" / "Combined"
sys.path.insert(0, str(DIR / "scripts"))  # exact original Pipeline scaler class
import pixel_transform  # noqa:F401

LABELS = "0123456789ABCDEF"
WRITERS = (41, 42, 43, 44)
FOCUS = ("0", "D", "8", "B")
RIVAL = {"0": "D", "D": "0", "8": "B", "B": "8"}
FEATURES = ("cx", "cy", "width", "height", "ink_ratio", "gray_mean",
            "stroke_proxy")
NPZDIR = ROOT / "02_data" / "Hand_made"
BASE = DIR / "results" / "EXP_CMB_001"


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_csv(path, rows, headers=None):
    rows = list(rows)
    if headers is None:
        assert rows
        headers = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def load_inputs():
    manifest = json.loads((BASE / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "EXP-CMB-001"
    assert sha256(ROOT / manifest["input_data_paths"]["official"]) == manifest["dataset_sha256"]["official_v3"]
    predpath = BASE / "OOF_predictions.csv"
    with open(predpath, newline="", encoding="utf-8-sig") as f:
        pred = list(csv.DictReader(f))
    assert len(pred) == 2560
    table = {}
    raw = {}
    labels = {}
    for w in WRITERS:
        path = NPZDIR / ("ml26_" + str(w) + ".npz")
        assert sha256(path) == manifest["dataset_sha256"]["handmade"][str(w)]
        with np.load(path, allow_pickle=False) as d:
            xx = d["img"].copy()
            yy = d["label_id"].astype(int)
            labeltext = d["label"].astype(str)
        assert xx.shape == (640, 28, 28) and xx.dtype == np.uint8
        assert np.array_equal(np.bincount(yy, minlength=16), np.full(16, 40))
        assert np.array_equal(labeltext, np.array(list(LABELS))[yy])
        raw[w], labels[w] = xx, yy
    for row in pred:
        w, idx = int(row["writer"]), int(row["index_in_writer"])
        key = (w, idx)
        assert w in WRITERS and 0 <= idx < 640 and key not in table
        assert row["true_label"] == LABELS[labels[w][idx]]
        row["writer"], row["index_in_writer"] = w, idx
        assert row["normal_correct"] == str(int(row["normal_pred"] == row["true_label"]))
        assert row["combined_correct"] == str(int(row["combined_pred"] == row["true_label"]))
        table[key] = row
    assert len(table) == 2560 and all((w, i) in table for w in WRITERS for i in range(640))
    assert sum(int(x["normal_correct"]) for x in pred) == 1602
    assert sum(int(x["combined_correct"]) for x in pred) == 2080
    return manifest, pred, table, raw, labels


def load_models(manifest):
    zpath = BASE / "trained_models.zip"
    models = {}
    with zipfile.ZipFile(zpath) as zf:
        for name, exp_sha in manifest["models_zip_members_sha256"].items():
            matches = [s for s in zf.namelist() if s.endswith("/" + name) or s == name]
            assert len(matches) == 1, (name, zf.namelist())
            data = zf.read(matches[0])
            assert hashlib.sha256(data).hexdigest() == exp_sha, name
            models[name] = joblib.load(io.BytesIO(data))
    assert len(models) == 5
    return models


def logits(pipe, x):
    net = pipe.named_steps["model"]
    assert net.activation == "relu" and len(net.coefs_) == 2
    xx = np.asarray(x, dtype=np.float64).reshape(-1, 784) / 255.0
    h = np.maximum(xx @ net.coefs_[0] + net.intercepts_[0], 0.0)
    return h @ net.coefs_[1] + net.intercepts_[1]


def geometry(image, threshold=32):
    im = np.asarray(image, dtype=float)
    mask = im >= threshold
    count = int(mask.sum())
    if count == 0:
        return dict(cx=np.nan, cy=np.nan, width=np.nan, height=np.nan,
                    ink_ratio=0.0, gray_mean=float(im.mean()),
                    foreground_mean=np.nan, ink_mass=0.0, stroke_proxy=np.nan,
                    bbox_aspect=np.nan, center_offset=np.nan)
    ys, xs = np.nonzero(mask)
    mass = float(im[mask].sum())
    cx = float((im * np.arange(28)[None, :]).sum() / max(float(im.sum()), 1e-12))
    cy = float((im * np.arange(28)[:, None]).sum() / max(float(im.sum()), 1e-12))
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    # Same thresholded, intensity-weighted CoM used in Normal's Spatial Response.
    wt = np.where(mask, im, 0.0)
    cx = float((wt * np.arange(28)[None, :]).sum() / mass)
    cy = float((wt * np.arange(28)[:, None]).sum() / mass)
    dist = distance_transform_edt(mask)
    return dict(cx=cx, cy=cy, width=width, height=height,
                ink_ratio=count/784.0, gray_mean=float(im.mean()),
                foreground_mean=float(im[mask].mean()), ink_mass=mass,
                stroke_proxy=float((2.0*dist[mask]).mean()),
                bbox_aspect=width/max(height, 1),
                center_offset=float(np.hypot(cx-13.5, cy-13.5)))


def prepare_features(pred, raw):
    feats = []
    for row in pred:
        w, i = row["writer"], row["index_in_writer"]
        f = geometry(raw[w][i], 32)
        f64, f128 = geometry(raw[w][i], 64), geometry(raw[w][i], 128)
        f = {**{k: row[k] for k in ("writer", "index_in_writer", "true_label",
                                   "normal_pred", "combined_pred", "normal_correct",
                                   "combined_correct")},
             "transition": ("both_correct" if row["normal_correct"] == "1" and row["combined_correct"] == "1"
                          else "corrected" if row["normal_correct"] == "0" and row["combined_correct"] == "1"
                          else "new_error" if row["normal_correct"] == "1" else
                          "persistent_same_error" if row["normal_pred"] == row["combined_pred"]
                          else "persistent_changed_error"),
             **f}
        for t, g in ((64, f64), (128, f128)):
            for k in ("cx", "cy", "width", "height", "ink_ratio"):
                f[k + "_t" + str(t)] = g[k]
        feats.append(f)
    assert len(feats) == 2560
    return feats


def make_class_tables(pred, out):
    c = np.zeros((16, 16), dtype=int)
    writer_cms = {}
    transitions = Counter()
    pairwriter = Counter()
    for r in pred:
        t, n, p = r["true_label"], r["normal_pred"], r["combined_pred"]
        w = r["writer"]
        c[LABELS.index(t), LABELS.index(p)] += 1
        pairwriter[(w, t, p)] += 1
        transitions[("N" if t == n else "W", "C" if t == p else "W",
                     "same" if n == p else "different")] += 1
    assert int(np.trace(c)) == 2080 and c.sum() == 2560
    p, rec, f1, support = precision_recall_fscore_support(
        [r["true_label"] for r in pred], [r["combined_pred"] for r in pred],
        labels=list(LABELS), zero_division=0)
    write_csv(out / "per_class_metrics.csv", [
        {"class": t, "support":int(support[i]), "precision":p[i], "recall":rec[i],
         "f1":f1[i], "correct":int(c[i, i])} for i,t in enumerate(LABELS)])
    wrows = []
    for w in WRITERS:
        rr = [r for r in pred if r["writer"] == w]
        mat = confusion_matrix([r["true_label"] for r in rr],
                               [r["combined_pred"] for r in rr], labels=list(LABELS))
        assert mat.sum() == 640
        writer_cms[w] = mat
        wrows.append({"writer":w,"n":len(rr),
                      "accuracy":float(np.trace(mat)/len(rr)),
                      "macro_recall":float(np.mean(np.diag(mat)/40.0)),
                      **{(t+"_to_"+p):int(mat[LABELS.index(t), LABELS.index(p)])
                         for t,p in (("0","D"),("D","0"),("8","B"),("B","8"))}})
    write_csv(out/"writer_metrics.csv", wrows)
    transrows = []
    for t in LABELS:
        for n in LABELS:
            for pnew in LABELS:
                count = sum(1 for r in pred if r["true_label"]==t and
                            r["normal_pred"]==n and r["combined_pred"]==pnew)
                if count:
                    transrows.append({"actual":t,"normal_prediction":n,
                                      "combined_prediction":pnew, "count":count,
                                      "transition": "corrected" if n!=t and pnew==t else
                                        "new_error" if n==t and pnew!=t else
                                        "both_correct" if n==t and pnew==t else
                                        "persistent_same_error" if n==pnew else
                                        "persistent_changed_error"})
    write_csv(out/"error_transitions.csv",transrows)
    return c, writer_cms, wrows, transrows


def save_cms(cm, writer_cms, out):
    target = out / "figures" / "writer_confusion_matrices"
    target.mkdir(parents=True, exist_ok=True)
    cases = [("pooled", cm, 160)] + [
        ("writer_" + str(w), writer_cms[w], 40) for w in WRITERS]
    for name, mat, maximum in cases:
        for kind in ("count", "normalized"):
            val = mat if kind=="count" else 100.0*mat/np.maximum(mat.sum(axis=1,keepdims=True),1)
            fig,ax = plt.subplots(figsize=(9,8))
            im = ax.imshow(val, cmap="viridis", vmin=0,vmax=maximum if kind=="count" else 100)
            fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
            ax.set(xticks=np.arange(16),yticks=np.arange(16),xticklabels=list(LABELS),
                   yticklabels=list(LABELS),xlabel="Predicted",ylabel="Actual",
                   title=name+" | "+kind+" | OOF")
            for i in range(16):
                for j in range(16):
                    if mat[i,j]:
                        ax.text(j,i,format(val[i,j],".1f") if kind=="normalized" else str(mat[i,j]),
                                ha="center",va="center",fontsize=6,
                                color="black" if val[i,j]>.55*(maximum if kind=="count" else 100) else "white")
            fig.tight_layout()
            fig.savefig(target/(name+"_"+kind+".png"),dpi=200)
            plt.close(fig)


def stratified_permutation(feats, out, repeats=2500):
    """Within each true-class and writer, shuffle the error label; BH across features."""
    rng = np.random.default_rng(20261116)
    rr = [r for r in feats if r["true_label"] in FOCUS]
    groups = defaultdict(list)
    for i,r in enumerate(rr):
        groups[(r["writer"],r["true_label"])].append(i)
    assert len(rr)==640 and len(groups)==16
    e = np.asarray([int(r["combined_correct"]=="0") for r in rr],dtype=float)
    X = np.asarray([[float(r[k]) for k in FEATURES] for r in rr])
    assert np.isfinite(X).all()
    Z = np.zeros_like(X)
    center_error = e.copy()
    for ix in groups.values():
        z = X[ix]
        Z[ix] = (z-z.mean(axis=0))/np.maximum(z.std(axis=0),1e-6)
        center_error[ix] -= e[ix].mean()
    observed = (Z.T@center_error)/len(rr)
    sims = np.zeros((repeats,len(FEATURES)))
    for b in range(repeats):
        pe=e.copy()
        for ix in groups.values():
            pe[ix]=rng.permutation(pe[ix])
        sims[b] = (Z.T @ (pe - (e-center_error)))/len(rr)
    pvals = (1+np.sum(np.abs(sims)>=np.abs(observed)[None,:],axis=0))/(repeats+1)
    # Benjamini-Hochberg q-values.
    order=np.argsort(pvals)
    qs=np.empty(len(pvals))
    running=1.0
    for rank_rev in range(len(pvals)-1,-1,-1):
        j=order[rank_rev]
        running=min(running,float(pvals[j])*len(pvals)/(rank_rev+1))
        qs[j]=running
    rows=[]
    for j,k in enumerate(FEATURES):
        low=Z[e==0,j]
        high=Z[e==1,j]
        rows.append({"feature":k,"n":len(rr),"n_errors":int(e.sum()),
                     "error_minus_correct_mean_z":float(high.mean()-low.mean()),
                     "centered_covariance":float(observed[j]),
                     "stratified_permutation_p":float(pvals[j]),
                     "bh_fdr_q":float(qs[j]),"permutations":repeats})
    write_csv(out/"feature_associations.csv",rows)
    groupsout=[]
    for (w,t),ix in sorted(groups.items()):
        actual = np.asarray(ix)
        z=X[actual]
        err=e[actual].astype(bool)
        for j,k in enumerate(FEATURES):
            groupsout.append({"writer":w,"true_class":t,"feature":k,
                 "n":len(ix),"errors":int(err.sum()),"correct":int((~err).sum()),
                 "mean_wrong":float(z[err,j].mean()) if err.any() else "",
                 "mean_correct":float(z[~err,j].mean()) if (~err).any() else "",
                 "mean_wrong_minus_correct":float(z[err,j].mean()-z[~err,j].mean())
                    if err.any() and (~err).any() else ""})
    write_csv(out/"feature_comparison.csv",groupsout)
    return rows


def select_focus(feats):
    selections=[]
    d=defaultdict(list)
    for r in feats:
        if r["true_label"] in FOCUS:
            d[(r["writer"],r["true_label"],r["combined_correct"])].append(r)
    for w in WRITERS:
        for cls in FOCUS:
            for corr in ("0","1"):
                arr=d[(w,cls,corr)]
                if arr:
                    # evenly distributed image indices; no model-score-based cherry picking
                    at=np.linspace(0,len(arr)-1,min(3,len(arr)),dtype=int)
                    selections.extend(arr[int(i)] for i in sorted(set(at)))
    keys={(r["writer"],r["index_in_writer"]) for r in selections}
    assert len(keys)==len(selections)
    return selections


def do_ig(pipe, x, cls, rival, steps=128):
    net=pipe.named_steps["model"]
    v=np.asarray(x,dtype=np.float64).reshape(784)/255.0
    w0,w1=(np.asarray(v,dtype=np.float64) for v in net.coefs_)
    b0,b1=(np.asarray(v,dtype=np.float64) for v in net.intercepts_)
    vclass=LABELS.index(cls); vrival=LABELS.index(rival)
    dw=w1[:,vclass]-w1[:,vrival]
    alpha=(np.arange(steps,dtype=np.float64)+.5)/steps
    h=alpha[:,None]*v[None,:]@w0 +b0
    grad=((h>0)*dw[None,:])@w0.T
    ig=v*grad.mean(axis=0)
    def margin(inp):
        z=np.maximum(inp@w0+b0,0.0)@w1+b1
        return float(z[vclass]-z[vrival])
    observed=margin(v);baseline=margin(np.zeros_like(v))
    residual=float(ig.sum()-(observed-baseline))
    return ig.reshape(28,28),observed,residual


def region_interventions(pipe, image, cls, competitor, seed=42):
    """Predeclared spatial regions vs same-count foreground removal (random controls).

    Region erasure may affect geometry; this tests coordinate sensitivity, NOT
    independent semantic stroke/connected-component structure.
    """
    xx=np.asarray(image,dtype=np.float32)
    mask=xx>=32
    yy,xxcoords=np.indices((28,28))
    regions={
        "left_middle":(xxcoords<=10)&(yy>=7)&(yy<=21),
        "right_middle":(xxcoords>=17)&(yy>=7)&(yy<=21),
        "center_vertical":(xxcoords>=11)&(xxcoords<=16)&(yy>=5)&(yy<=23),
        "center_horizontal":(yy>=10)&(yy<=17)&(xxcoords>=5)&(xxcoords<=22)
    }
    full=np.flatnonzero(mask)
    rng=np.random.default_rng(seed)
    base=logits(pipe, image)[0]
    ci,ri=LABELS.index(cls),LABELS.index(competitor)
    initial_margin=float(base[ci]-base[ri])
    rows=[]
    for name,region in regions.items():
        pos=np.flatnonzero(mask&region)
        if len(pos)<4 or len(full)-len(pos)<4:
            continue
        target=np.array(image,copy=True).reshape(784)
        target[pos]=0
        after=logits(pipe,target)[0]
        effect=float(after[ci]-after[ri]-initial_margin)
        rand=[]
        for _ in range(32):
            p=rng.choice(full,len(pos),replace=False)
            q=np.array(image,copy=True).reshape(784)
            q[p]=0
            margin=logits(pipe,q)[0]
            rand.append(float(margin[ci]-margin[ri]-initial_margin))
        rows.append({"region":name,"foreground_removed":len(pos),
                     "region_ink_mass_removed":float(np.asarray(image).reshape(784)[pos].sum()),
                     "region_delta_margin":effect,
                     "random_mean_delta_margin":float(np.mean(rand)),
                     "random_mean_abs_delta_margin":float(np.mean(np.abs(rand))),
                     "region_excess_abs_effect":float(abs(effect)-np.mean(np.abs(rand))),
                     "region_prediction_after":LABELS[int(after.argmax())],
                     "random_repetitions":len(rand)})
    return rows


def make_grids(feats, raw, out):
    figdir=out/"figures"
    figdir.mkdir(parents=True,exist_ok=True)
    for classes,name in ((("0","D"),"error_grid_0D"),(("8","B"),"error_grid_8B")):
        pool=[r for r in feats if r["true_label"] in classes]
        examples=[]
        for cls in classes:
            for incorrect in ("0","1"):
                sub=[r for r in pool if r["true_label"]==cls and r["combined_correct"]==incorrect]
                for w in WRITERS:
                    wr=[r for r in sub if r["writer"]==w]
                    if wr:examples.extend(wr[:2])
        examples=examples[:24]
        fig,axes=plt.subplots(4,6,figsize=(12,8))
        for ax,r in zip(axes.flat,examples):
            ax.imshow(raw[r["writer"]][r["index_in_writer"]],vmin=0,vmax=255,cmap="gray")
            ax.set_title(str(r["writer"])+"/"+str(r["index_in_writer"])+
                         ": "+r["true_label"]+" > "+r["combined_pred"],fontsize=8)
            ax.axis("off")
        for ax in axes.flat[len(examples):]:ax.axis("off")
        fig.suptitle("Real Hand-made | writer/index : actual > Combined prediction")
        fig.tight_layout()
        fig.savefig(figdir/(name+".png"),dpi=180)
        plt.close(fig)


def analysis(models, table, raw, feats, out):
    selections=select_focus(feats)
    igrows, regionrows=[],[]
    for r in selections:
        w,i,cls=r["writer"],r["index_in_writer"],r["true_label"]
        pipe=models["combined_heldout_"+str(w)+"_mlp.joblib"]
        normal=models["normal_mlp.joblib"]
        image=raw[w][i]
        rival=RIVAL[cls]
        for model_name,model in (("Normal",normal),("Combined",pipe)):
            ig,margin,residual=do_ig(model,image,cls,rival)
            score=logits(model,image)[0]
            igrows.append({"writer":w,"index_in_writer":i,"actual":cls,
                "rival":rival,"model":model_name,"correct":int(score.argmax()==LABELS.index(cls)),
                "prediction":LABELS[score.argmax()],"actual_minus_rival_logit_margin":margin,
                "IG_positive_sum":float(np.clip(ig,0,None).sum()),
                "IG_negative_sum":float(np.clip(ig,None,0).sum()),
                "IG_absolute_sum":float(np.abs(ig).sum()),
                "IG_completeness_residual":residual,
                "IG_max_coord_y":int(np.unravel_index(np.abs(ig).argmax(),ig.shape)[0]),
                "IG_max_coord_x":int(np.unravel_index(np.abs(ig).argmax(),ig.shape)[1])})
        for intervention in region_interventions(pipe,image,cls,rival,
                                                 seed=20261116+w*1000+i):
            regionrows.append({"writer":w,"index_in_writer":i,"actual":cls,
                "rival":rival,"baseline_prediction":r["combined_pred"],
                "baseline_correct":int(r["combined_correct"]),**intervention})
    write_csv(out/"margin_ig_diagnostics.csv",igrows)
    write_csv(out/"region_ablation_controls.csv",regionrows)
    return selections,igrows,regionrows


def summarize(manifest,pred,assoc,selections,igrows,regrows,wr,out,elapsed):
    corr = Counter()
    for r in pred:
        nc=r["normal_correct"]=="1";cc=r["combined_correct"]=="1"
        trans="both_correct" if nc and cc else "corrected" if not nc and cc else \
              "new_error" if nc and not cc else "both_wrong"
        corr[trans]+=1
    pairs={}
    for t,p in (("0","D"),("D","0"),("8","B"),("B","8")):
        rr=[r for r in pred if r["true_label"]==t and r["combined_pred"]==p]
        pairs[t+"->"+p]={"combined":len(rr),
                         "writer_breakdown":{str(w):sum(r["writer"]==w for r in rr)
                                             for w in WRITERS},
                         "new_error":sum(r["normal_pred"]==t for r in rr),
                         "same_wrong":sum(r["normal_pred"]==p for r in rr),
                         "other_wrong":sum(r["normal_pred"]!=t and r["normal_pred"]!=p
                                           for r in rr)}
    # A region ablation is an exploratory coordinate sensitivity signal only.
    region_group={}
    for region in sorted({r["region"] for r in regrows}):
        rr=[r for r in regrows if r["region"]==region]
        perimg=defaultdict(list)
        for r in rr:perimg[(r["writer"],r["index_in_writer"])].append(r["region_excess_abs_effect"])
        samplemeans=np.array([np.mean(v) for v in perimg.values()])
        if len(samplemeans)<2:continue
        rng=np.random.default_rng(20261116)
        boots=np.mean(rng.choice(samplemeans,(2000,len(samplemeans)),replace=True),axis=1)
        region_group[region]={"n_distinct_images":len(samplemeans),
            "mean_excess_abs_logit":float(samplemeans.mean()),
            "bootstrap95":np.quantile(boots,[.025,.975]).tolist(),
            "per_image_fraction_excess_positive":float(np.mean(samplemeans>0))}
    report={
        "status":"experimental_evidence_created; A/B go/no-go requires interpretation",
        "source":"EXP-CMB-001 original OOF and frozen models",
        "n_images":2560, "n_focus_images":640,
        "baseline_oof_accuracy":2080/2560,
        "transitions":dict(corr),"focus_pairs":pairs,
        "feature_associations":assoc,
        "A_IG_n_model_evaluations":len(igrows),
        "A_distinct_images":len(selections),
        "A_max_abs_IG_completeness_residual":max(abs(x["IG_completeness_residual"])
                                                for x in igrows),
        "A_region_ablation_exploratory":region_group,
        "writer_metrics":wr,
        "rules":{"B":"Look for within-writer-and-class BH q<0.05, nontrivial effect, across-writer consistency, and threshold robustness before recommending spatial normalization",
                 "A":"Logit/IG and prespecified region ablation show input coordinate relevance only. To claim meaningful stroke topology requires structure-preserving controls. Do not equate signed IG to semantic shape.",
                 "decision":"A/B are not mutually exclusive. If evidence cannot distinguish pathways, report INCONCLUSIVE and extend EXP-CMB-002, not advance."},
        "environment":{"numpy":np.__version__,"sklearn":sklearn.__version__,"scipy":scipy.__version__},
        "elapsed_seconds":elapsed,
        "original_test_used":False,"professor_test_used":False,
        "new_training_runs":0,"model_sha_members":manifest["models_zip_members_sha256"]
    }
    (out/"evidence_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    summary=[
        "# EXP-CMB-002 evidence — automatic execution results (research review)",
        "",
        "**Status:** analysis completed, final A/B determination pending interpretation and controls.",
        "",
        "- Original OOF revalidated: 2,560 Hand-made images, Combined 2,080 correct (81.25%).",
        "- Corrected "+str(corr["corrected"])+", newly wrong "+str(corr["new_error"])+
          ", both wrong "+str(corr["both_wrong"])+".",
        "- Focused true classes 0/D/8/B: 640 images.",
        "- Fixed-fold model IG diagnostics: "+str(len(selections))+" distinct images, "+
          str(len(igrows))+" model-condition rows.",
        "- Stratified writer+class permutation: 2,500 repetitions per global feature, BH-FDR across 7 features.",
        "- Region ablation: predefined regions and 32 random same-foreground-count removals per region.",
        "- **Interpretation limits:** region ablation modifies geometry and brightness mass; it is NOT a pure stroke-shape intervention. B correlation is NOT proof of causal improvement. No new training.",
        "",
        "## Feature association (pooled 0/D/8/B, conditioned on writer and actual class)",
        "",
        "| Feature | error - correct (within-group z) | permutation p | BH q |",
        "|---|---:|---:|---:|",
    ]
    summary.extend("| {} | {:.3f} | {:.4f} | {:.4f} |".format(x["feature"],
         x["error_minus_correct_mean_z"],x["stratified_permutation_p"],x["bh_fdr_q"])
         for x in assoc)
    summary.extend(["","## Region ablation (exploratory coordinate relevance only)","",
          "| Region | Images | Mean excess abs margin effect | image bootstrap 95% interval |",
          "|---|---:|---:|---|"])
    summary.extend("| {} | {} | {:.3f} | [{:.3f}, {:.3f}] |".format(k,v["n_distinct_images"],
                     v["mean_excess_abs_logit"],*v["bootstrap95"])
                   for k,v in region_group.items())
    summary.extend(["","See evidence_summary.json, feature_comparison.csv, margin_ig_diagnostics.csv, region_ablation_controls.csv and actual real-image grids.",
      "","Official original Test and professor Test were not accessed. Baseline EXP-CMB-001 files were not changed."])
    (out/"SUMMARY.md").write_text("\n".join(summary)+"\n",encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    manifest,pred,table,raw,labels=load_inputs()
    models=load_models(manifest)
    for w in WRITERS:
        X=raw[w].reshape(640,784).astype(np.float32)
        for key,model in (("normal_pred",models["normal_mlp.joblib"]),
                          ("combined_pred",models["combined_heldout_"+str(w)+"_mlp.joblib"])):
            obtained=np.array(list(LABELS))[model.predict(X).astype(int)]
            expected=np.array([table[w,i][key] for i in range(640)])
            assert np.array_equal(obtained,expected),("stored OOF prediction mismatch",w,key)
    print("REPRODUCED 2,560 OOF predictions with original frozen 5 models",flush=True)
    feats=prepare_features(pred,raw)
    write_csv(out/"image_features.csv",feats)
    c,writer_cms,wrows,trows=make_class_tables(pred,out)
    save_cms(c,writer_cms,out)
    assoc=stratified_permutation(feats,out)
    make_grids(feats,raw,out)
    selections,igrows,regrows=analysis(models,table,raw,feats,out)
    summarize(manifest,pred,assoc,selections,igrows,regrows,wrows,out,
              time.perf_counter()-start)
    print((out/"SUMMARY.md").read_text(encoding="utf-8"),flush=True)
    print("EXPERIMENT COMPLETED — FROZEN MODELS, NO TEST",flush=True)


if __name__=="__main__":
    main()
