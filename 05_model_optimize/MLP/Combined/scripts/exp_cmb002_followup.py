#!/usr/bin/env python3
"""EXP-CMB-002 continuation: writer-specific morphology + matched pixel swaps.

Purpose: decide whether further investigation of (A) pixel/stroke structure and
(B) global geometry/ink is justified. Frozen Combined 4-fold models only.
No Combined training, no Original Test, no training-time parameter selection.
Results are exploratory and remain on a review branch before user approval.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import ndimage
import exp_cmb002_evidence as core

FOCUS=core.FOCUS
RIVAL=core.RIVAL
FEATURES_MORPH=("holes_128","connected_components_128","left_mass_fraction",
  "right_mass_fraction","middle_mass_fraction","upper_mass_fraction",
  "central_row_mass_fraction","left_column_mass_fraction",
  "bilateral_asymmetry","upper_lower_asymmetry",
  "centerline_foreground_128")
TARGETS=((43,"0"),(41,"D"),(42,"D"),(43,"8"),(41,"B"))
ALL_GROUPS=tuple((w,c) for w in core.WRITERS for c in FOCUS)


def morphological(img):
    a=np.asarray(img,dtype=np.float64)
    mass=max(float(a.sum()),1e-12)
    mask=a>=128
    struct=ndimage.generate_binary_structure(2,1)
    components,ncomp=ndimage.label(mask,structure=struct)
    bg,nlab=ndimage.label(~mask,structure=struct)
    exterior=set(np.unique(np.r_[bg[0,:],bg[-1,:],bg[:,0],bg[:,-1]]).tolist())
    holes=sum(i not in exterior for i in range(1,nlab+1))
    return {
        "holes_128":holes,
        "connected_components_128":ncomp,
        "left_mass_fraction":float(a[:,:11].sum()/mass),
        "right_mass_fraction":float(a[:,17:].sum()/mass),
        "middle_mass_fraction":float(a[:,11:17].sum()/mass),
        "upper_mass_fraction":float(a[:14,:].sum()/mass),
        "central_row_mass_fraction":float(a[10:18,:].sum()/mass),
        "left_column_mass_fraction":float(a[6:23,7:13].sum()/mass),
        "bilateral_asymmetry":float(np.abs(a-a[:,::-1]).sum()/mass),
        "upper_lower_asymmetry":float(abs(a[:14,:].sum()-a[14:,:].sum())/mass),
        "centerline_foreground_128":int(mask[8:21,12:16].sum())
    }


def permutation_table(rows,features,groups,repeats,seed):
    """Within true class and writer, compare error and correct via shuffles."""
    rng=np.random.default_rng(seed)
    n=len(rows)
    x=np.array([[float(row[k]) for k in features] for row in rows],dtype=float)
    y=np.array([int(row["combined_correct"]=="0") for row in rows])
    by=defaultdict(list)
    for i,r in enumerate(rows):by[(r["writer"],r["true_label"])].append(i)
    assert set(by)==set(groups)
    # Z-score within writer/class to avoid writer/class mean confounding.
    z=np.zeros_like(x)
    ycenter=np.zeros(n)
    for ids in by.values():
        u=x[ids]
        z[ids]=(u-u.mean(0))/np.maximum(u.std(0),1e-6)
        ycenter[ids]=y[ids]-y[ids].mean()
    obs=z.T@ycenter/n
    sim=np.zeros((repeats,len(features)))
    for b in range(repeats):
        yy=np.zeros(n)
        for ids in by.values():
            yy[ids]=rng.permutation(y[ids])-y[ids].mean()
        sim[b]=z.T@yy/n
    p=(1+(np.abs(sim)>=np.abs(obs)).sum(0))/(repeats+1)
    order=np.argsort(p)
    q=np.ones(len(p))
    best=1.0
    for irank in range(len(order)-1,-1,-1):
        idx=order[irank]
        best=min(best,float(p[idx])*len(p)/(irank+1))
        q[idx]=best
    return [{"feature":ft,"n":n,"errors":int(y.sum()),
             "error_minus_correct_z":float(z[y==1,j].mean()-z[y==0,j].mean()),
             "p":float(p[j]),"q":float(q[j]),"strata":len(by)}
            for j,ft in enumerate(features)]


def targeted_permutations(rows,features,repeats=2000):
    raw=[]
    for w,cls in TARGETS:
        rr=[r for r in rows if r["writer"]==w and r["true_label"]==cls]
        assert len(rr)==40
        err=np.array([int(r["combined_correct"]=="0") for r in rr],dtype=float)
        assert 0<err.sum()<40
        x=np.array([[float(r[ft]) for ft in features] for r in rr])
        rng=np.random.default_rng(20261116+100*w+core.LABELS.index(cls))
        std=np.maximum(x.std(0),1e-6)
        z=(x-x.mean(0))/std
        obs=(z.T@(err-err.mean()))/40
        sims=np.zeros((repeats,len(features)))
        for i in range(repeats):
            sims[i]=(z.T@(rng.permutation(err)-err.mean()))/40
        p=(1+(np.abs(sims)>=np.abs(obs)).sum(0))/(repeats+1)
        for j,ft in enumerate(features):
            raw.append({"writer":w,"true_class":cls,"feature":ft,
                       "n":40,"n_wrong":int(err.sum()),"n_correct":int(40-err.sum()),
                       "wrong_minus_correct_raw":float(x[err==1,j].mean()-x[err==0,j].mean()),
                       "effect_wrong_minus_correct_z":float(z[err==1,j].mean()-z[err==0,j].mean()),
                       "p_unadjusted":float(p[j])})
    order=sorted(range(len(raw)),key=lambda i:raw[i]["p_unadjusted"])
    best=1.0
    for irank in range(len(order)-1,-1,-1):
        idx=order[irank]
        best=min(best,raw[idx]["p_unadjusted"]*len(raw)/(irank+1))
        raw[idx]["q_5_groups_all_features"]=best
    return raw


def n_edges(mask):
    a=np.asarray(mask,dtype=bool)
    return int(np.sum(a[:,:-1]&a[:,1:])+np.sum(a[:-1,:]&a[1:,:]))


def swap_candidates(image,seed,target=54):
    """Fixed 4 foreground/background exchanges per candidate; exact intensity histogram.

    Swaps are entirely within the original bbox and near neighbours. Accept only
    candidates whose bbox (threshold 128) remains exactly and CoM shift <=0.3px.
    Candidate selection uses no MLP outputs or ground-truth label.
    """
    x=np.asarray(image,dtype=np.uint8)
    m=x>=128
    ys,xs=np.nonzero(m)
    if len(xs)<10:return []
    xmin,xmax=int(xs.min()),int(xs.max())
    ymin,ymax=int(ys.min()),int(ys.max())
    bright=np.flatnonzero((x>=200)&np.indices(x.shape)[0].__ge__(ymin+1)&
        (np.indices(x.shape)[0]<=ymax-1)&
        (np.indices(x.shape)[1]>=xmin+1)&
        (np.indices(x.shape)[1]<=xmax-1))
    dark=np.flatnonzero((x<=30)&(np.indices(x.shape)[0]>=ymin+1)&
        (np.indices(x.shape)[0]<=ymax-1)&
        (np.indices(x.shape)[1]>=xmin+1)&
        (np.indices(x.shape)[1]<=xmax-1))
    if len(bright)<4 or len(dark)<4:return []
    yy,xx=np.unravel_index(bright,(28,28))
    yd,xd=np.unravel_index(dark,(28,28))
    pairs=[]
    for i,fg in enumerate(bright):
        dist=np.abs(xd-xx[i])+np.abs(yd-yy[i])
        for j in np.flatnonzero((dist>=1)&(dist<=3)):
            pairs.append((int(fg),int(dark[j]),int(dist[j])))
    if len(pairs)<16:return []
    rng=np.random.default_rng(seed)
    original_edges=n_edges(m)
    a=x.ravel()
    total=float(x.sum())
    if total<1:return []
    init_cx=float((x.astype(float)*np.arange(28)[None,:]).sum()/total)
    init_cy=float((x.astype(float)*np.arange(28)[:,None]).sum()/total)
    saved={}
    for attempt in range(2400):
        picks=rng.permutation(len(pairs))
        chosen=[]
        used=set()
        for pos in picks[:min(len(picks),160)]:
            fg,bg,d=pairs[pos]
            if fg not in used and bg not in used:
                chosen.append((fg,bg,d));used.add(fg);used.add(bg)
                if len(chosen)==4:break
        if len(chosen)<4:continue
        b=a.copy()
        for fg,bg,_ in chosen:b[fg],b[bg]=b[bg],b[fg]
        shifted=b.reshape(28,28)
        ys2,xs2=np.nonzero(shifted>=128)
        if not len(xs2) or (int(xs2.min()),int(xs2.max()),int(ys2.min()),int(ys2.max()))!=(xmin,xmax,ymin,ymax):continue
        new_total=float(b.sum())
        assert new_total==total
        cx=float((shifted.astype(float)*np.arange(28)[None,:]).sum()/total)
        cy=float((shifted.astype(float)*np.arange(28)[:,None]).sum()/total)
        shift=float(np.hypot(cx-init_cx,cy-init_cy))
        if shift>0.3:continue
        # Preserve complete intensity histogram and foreground count.
        assert np.array_equal(np.sort(a),np.sort(b))
        assert int((shifted>=128).sum())==int(m.sum())
        key=tuple(sorted((fg,bg) for fg,bg,_ in chosen))
        if key in saved:continue
        saved[key]={
            "array":shifted.copy(),
            "delta_4neighbour_connections":n_edges(shifted>=128)-original_edges,
            "mean_manhattan_swap_distance":float(np.mean([d for _,_,d in chosen])),
            "center_shift_px":shift
        }
        if len(saved)>=target:break
    return list(saved.values())


def select_samples(rows):
    d=defaultdict(list)
    for r in rows:
        if r["true_label"] in FOCUS:
            d[(r["writer"],r["true_label"],r["combined_correct"])].append(r)
    sel=[]
    for w,c in ALL_GROUPS:
        for correct in ("0","1"):
            v=d[(w,c,correct)]
            if v:
                at=np.linspace(0,len(v)-1,min(2,len(v)),dtype=int)
                sel.extend(v[int(j)] for j in sorted(set(at)))
    assert len({(r["writer"],r["index_in_writer"]) for r in sel})==len(sel)
    return sel


def margin(model,image,actual,rival):
    logits=core.logits(model,image)[0]
    return float(logits[core.LABELS.index(actual)]-logits[core.LABELS.index(rival)]),core.LABELS[int(logits.argmax())]


def topology_analysis(rows,raw,models,out):
    selected=select_samples(rows)
    trials,paired=[],[]
    for row in selected:
        w,idx,cls=row["writer"],row["index_in_writer"],row["true_label"]
        mod=models["combined_heldout_"+str(w)+"_mlp.joblib"]
        im=raw[w][idx]
        baseline,p0=margin(mod,im,cls,RIVAL[cls])
        proposals=swap_candidates(im,20261116+w*200+idx,target=54)
        if len(proposals)<16:continue
        # Candidate rank based only on adjacent-foreground graph, no MLP scores.
        ds=np.array([p["delta_4neighbour_connections"] for p in proposals])
        distances=np.array([p["mean_manhattan_swap_distance"] for p in proposals])
        order=np.argsort(ds,kind="stable")
        nselect=8
        disruptive=order[:nselect]
        controls=[]
        available=set(range(len(proposals)))-set(disruptive)
        for i in disruptive:
            # Match overall displacement (same 4 swaps, similar mean Manhattan length).
            j=min(available,key=lambda j:(abs(distances[j]-distances[i]),abs(ds[j]-np.median(ds)),j))
            controls.append(j)
            available.remove(j)
        this=[]
        for condition,indices in (("largest_connectivity_loss",disruptive),
                                  ("distance_matched_control",controls)):
            for j in indices:
                p=proposals[int(j)]
                after,pred=margin(mod,p["array"],cls,RIVAL[cls])
                trial={"writer":w,"index_in_writer":idx,"actual":cls,
                       "baseline_correct":int(row["combined_correct"]),
                       "condition":condition,"candidate_id":int(j),
                       "baseline_margin":baseline,"margin_delta":after-baseline,
                       "abs_margin_delta":abs(after-baseline),
                       "new_prediction":pred,"prediction_flip":int(pred!=p0),
                       "connectivity_change":p["delta_4neighbour_connections"],
                       "mean_swap_distance":p["mean_manhattan_swap_distance"],
                       "center_shift_px":p["center_shift_px"],
                       "foreground_swaps":4,"pixels_changed":8}
                trials.append(trial);this.append(trial)
        a=[x for x in this if x["condition"]=="largest_connectivity_loss"]
        b=[x for x in this if x["condition"]=="distance_matched_control"]
        paired.append({"writer":w,"index_in_writer":idx,"actual":cls,
          "baseline_correct":int(row["combined_correct"]),
          "mean_edges_loss_selected":float(np.mean([x["connectivity_change"] for x in a])),
          "mean_edges_loss_control":float(np.mean([x["connectivity_change"] for x in b])),
          "mean_distance_selected":float(np.mean([x["mean_swap_distance"] for x in a])),
          "mean_distance_control":float(np.mean([x["mean_swap_distance"] for x in b])),
          "mean_abs_delta_selected":float(np.mean([x["abs_margin_delta"] for x in a])),
          "mean_abs_delta_control":float(np.mean([x["abs_margin_delta"] for x in b])),
          "delta_abs_margin_selected_minus_control":float(np.mean([x["abs_margin_delta"] for x in a])-
                                                       np.mean([x["abs_margin_delta"] for x in b])),
          "flip_selected":float(np.mean([x["prediction_flip"] for x in a])),
          "flip_control":float(np.mean([x["prediction_flip"] for x in b]))})
    if not paired:raise RuntimeError("No valid strict histogram-preserving proposals")
    core.write_csv(out/"pixel_swap_trials.csv",trials)
    core.write_csv(out/"pixel_swap_per_image.csv",paired)
    summaries=[]
    for label,sel in (
        ("all",paired),
        ("0D",[r for r in paired if r["actual"] in ("0","D")]),
        ("8B",[r for r in paired if r["actual"] in ("8","B")])):
        if len(sel)<5:continue
        d=np.array([r["delta_abs_margin_selected_minus_control"] for r in sel])
        rng=np.random.default_rng(20261116)
        boot=np.mean(rng.choice(d,size=(4000,len(d)),replace=True),axis=1)
        summaries.append({"group":label,"n_independent_original_images":len(sel),
          "mean_excess_abs_margin":float(d.mean()),
          "bootstrap95_low":float(np.quantile(boot,.025)),
          "bootstrap95_high":float(np.quantile(boot,.975)),
          "fraction_per_image_positive":float(np.mean(d>0)),
          "mean_edges_selected":float(np.mean([r["mean_edges_loss_selected"] for r in sel])),
          "mean_edges_control":float(np.mean([r["mean_edges_loss_control"] for r in sel])),
          "mean_move_selected":float(np.mean([r["mean_distance_selected"] for r in sel])),
          "mean_move_control":float(np.mean([r["mean_distance_control"] for r in sel])),
          "n_candidates_per_image":54,"n_each_condition":8})
    core.write_csv(out/"topology_contrasts.csv",summaries)
    return len(selected),len(paired),summaries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest,preds,table,raw,labels=core.load_inputs()
    models=core.load_models(manifest)
    rows=core.prepare_features(preds,raw)
    for r in rows:
        r.update(morphological(raw[r["writer"]][r["index_in_writer"]]))
    core.write_csv(out/"morphology_features.csv",(
        {k:r[k] for k in ("writer","index_in_writer","true_label","combined_correct",
                          *FEATURES_MORPH)} for r in rows))
    focus=[r for r in rows if r["true_label"] in FOCUS]
    assert len(focus)==640
    morph=permutation_table(focus,FEATURES_MORPH,ALL_GROUPS,3000,20261116)
    core.write_csv(out/"morphology_associations.csv",morph)
    target=targeted_permutations(focus,(*core.FEATURES,*FEATURES_MORPH),repeats=2000)
    core.write_csv(out/"targeted_writer_class_tests.csv",target)
    selected,nvalid,topology=topology_analysis(focus,raw,models,out)
    summary={"status":"followup completed; interpret with explicit A/B decision",
       "n_focus_images":640,"n_morph_features":len(FEATURES_MORPH),
       "morphology_associations":morph,
       "targeted_unadjusted_q_significant":[x for x in target if x["q_5_groups_all_features"]<.05],
       "targeted_five_groups":[list(v) for v in TARGETS],
       "topology_selected_images":selected,"topology_valid_images":nvalid,
       "topology_contrasts":topology,
       "caveats":[
        "OOF repeatedly investigated as development data; not independent validation.",
        "Morphology descriptors overlap global geometry and do not isolate human semantic strokes.",
        "Pixel swaps preserve full intensity histogram, bbox and >=128 foreground count, with pixel-shift CoM <=0.3, but exact continuous CoM and all local geometry cannot be held fixed.",
        "An adjacency-edge count is a topology proxy, not a human visual recognition score.",
        "Targeted five strata were chosen after observing original OOF failure patterns; report them as exploratory and control FDR.",
        "The models were never retrained or tuned."
       ],
       "official_original_test_used":False,"professor_test_used":False}
    (out/"FOLLOWUP_EVIDENCE.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# EXP-CMB-002 continuation — morphology and controlled pixel swaps","",
      "- Original held-out OOF and 5 frozen models re-read; no MLP training.",
      "- Same 640 focus images. Global morphology feature permutations: 3,000 within-writer-and-class shuffles, BH across 11.",
      "- Five preselected writer/class strata: 2,000 shuffles, BH across all 5×18 tests.",
      "- Pixel swap samples: "+str(selected)+" selected, "+str(nvalid)+" with strict valid proposals, full 0–255 histogram and bbox preserved.",
      "",
      "## Global morphology association","",
      "| Feature | error-minus-correct z | p | q |","|---|---:|---:|---:|"]
    lines.extend("| {} | {:.3f} | {:.4f} | {:.4f} |".format(r["feature"],
         r["error_minus_correct_z"],r["p"],r["q"]) for r in morph)
    lines.extend(["","## Image-level topology intervention contrasts","",
      "| Group | N images | Selected - control absolute margin | Bootstrap 95% CI | Mean connection change (selected / control) |",
      "|---|---:|---:|---|---|"])
    lines.extend("| {} | {} | {:.3f} | [{:.3f}, {:.3f}] | {:.2f} / {:.2f} |".format(
        r["group"],r["n_independent_original_images"],r["mean_excess_abs_margin"],
        r["bootstrap95_low"],r["bootstrap95_high"],r["mean_edges_selected"],
        r["mean_edges_control"]) for r in topology)
    sig=[x for x in target if x["q_5_groups_all_features"]<.05]
    lines.extend(["","## Targeted writer-class tests","",
       "BH q<0.05 among 5 selected strata × 18 features: "+str(len(sig))+".",
       "This is **exploratory** even if significant; selected strata arise from baseline errors.",
       "See targeted_writer_class_tests.csv for all adjusted p-values.",
       "",
       "**Limits:** the pixel swaps are matched on 4 exchanges and displacement and preserve histogram, bounding box, thresholded foreground pixel count, with <=0.3px CoM movement. Four-neighbour connectivity is only a morphological proxy. No final causal claim of semantic shape recognition.",
       ""])
    (out/"FOLLOWUP_SUMMARY.md").write_text("\n".join(lines),encoding="utf-8")
    print((out/"FOLLOWUP_SUMMARY.md").read_text(encoding="utf-8"),flush=True)
    print("EXP-CMB-002 FOLLOWUP COMPLETE — NO TRAINING OR TEST",flush=True)


if __name__=="__main__":main()
