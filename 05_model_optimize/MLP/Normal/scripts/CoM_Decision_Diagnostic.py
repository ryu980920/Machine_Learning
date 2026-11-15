#!/usr/bin/env python3
"""EXP유004 follow-up (review only): Why CoM boosts digits but hurts letters.

Two *controlled* contrasts:
  1) Same frozen baseline weights, RAW vs CoM-shifted pixels (movement effect).
  2) Same CoM-shifted pixels, frozen Normal vs CoM-trained weights (fit effect).
The CoM-trained model fed RAW pixels is a diagnostic off-training-distribution
control, not a deployable policy.

Fixed class-logit margin Integrated Gradients on selected *real* transitions:
true 5 vs F, true B vs 8. Always same black reference and target sign.
No model fitting, no Test, no new hyperparameter choices, no publication.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from Pixcel_Attribution import read_handmade, LABELS
from Automatic_Centering_Transformer import center_batch, center_shifts
from Input_Sensitivity import get_output

BASE = ROOT/"05_model_optimize"/"MLP"/"Normal"
NORMAL_MODEL=BASE/"results"/"Input_Sensitivity"/"normal_mlp_refit_stage3.joblib"
COM_MODEL=BASE/"results"/"Automatic_Centering"/"exp004_mass.joblib"
RECORDS=BASE/"results"/"Automatic_Centering"/"exp004_predictions.csv.gz"
NORMAL_SHA="9b54ba9c60e389575200d88ad0b93c47ec47f9c673eebeb755ff47e3ebe23b0a"
COM_SHA="3c1aac362d75ff5c4d9652188c1c4fd97524525f5409ec00d5a6c56fa3ad27ef"
COLS=("Normal_raw","Normal_CoM","CoMModel_raw","CoMModel_CoM")


def sha256(path:Path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def savecsv(path,rows):
    assert rows
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def logits_for(net,raw):
    # raw pixels [0,255], exact existing pipeline's PixelScaler255 transformation.
    raw=np.asarray(raw,dtype=np.float32).reshape(-1,784)
    return get_output(type("Fake",(),{"named_steps":{"model":net}})(),raw)[0]


def true_vs_other_ig(net,raw,true,other,steps=128):
    """IG for logit(true) - logit(other), black baseline, fixed signed target."""
    x=np.asarray(raw,dtype=np.float64).reshape(784)/255.0
    w0,w1=(np.asarray(a,dtype=np.float64) for a in net.coefs_)
    b0=np.asarray(net.intercepts_[0],dtype=np.float64)
    b1=np.asarray(net.intercepts_[1],dtype=np.float64)
    assert len(net.coefs_)==2 and net.activation=="relu"
    cls=int(true); competitor=int(other)
    weight=w1[:,cls]-w1[:,competitor]
    aa=(np.arange(steps,dtype=np.float64)+.5)/steps
    z=(aa[:,None]*x[None,:])@w0+b0
    active=(z>0).astype(np.float64)
    grad=(active*weight[None,:])@w0.T
    ig=x*grad.mean(axis=0)
    def pairmargin(v):
        h=np.maximum(v@w0+b0,0)
        zz=h@w1+b1
        return float(zz[cls]-zz[competitor])
    delta=pairmargin(x)-pairmargin(np.zeros_like(x))
    residual=float(ig.sum()-delta)
    return ig.reshape(28,28),float(pairmargin(x)),residual


def metric(y,pred,mask,name,condition):
    truth=y[mask]; p=pred[mask]
    return {"group":name,"condition":condition,"n":int(mask.sum()),
            "correct":int((truth==p).sum()),
            "accuracy":float((truth==p).mean()),
            "true_digit_predictions":int((p<10).sum()),
            "true_letter_predictions":int((p>=10).sum())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    assert sha256(NORMAL_MODEL)==NORMAL_SHA
    assert sha256(COM_MODEL)==COM_SHA
    base_model=joblib.load(NORMAL_MODEL)
    com_model=joblib.load(COM_MODEL)
    assert base_model.named_steps["model"].get_params()==com_model.named_steps["model"].get_params()
    assert com_model.named_steps["centering"].method=="mass"
    x,y,writers,indices=read_handmade()
    assert len(y)==2560
    moved=center_batch(x,"mass")
    shifts=center_shifts(x,"mass")
    assert np.array_equal(com_model.named_steps["centering"].transform(x),moved)
    net_base=base_model.named_steps["model"]
    net_com=com_model.named_steps["model"]
    inputs={"Normal_raw":x, "Normal_CoM":moved,
            "CoMModel_raw":x, "CoMModel_CoM":moved}
    nets={"Normal_raw":net_base,"Normal_CoM":net_base,
          "CoMModel_raw":net_com,"CoMModel_CoM":net_com}
    output={name:logits_for(nets[name],inputs[name]) for name in COLS}
    pred={name:output[name].argmax(axis=1) for name in COLS}
    assert int((pred["Normal_raw"]==y).sum())==1609
    assert int((pred["CoMModel_CoM"]==y).sum())==1974
    assert np.array_equal(pred["Normal_raw"],base_model.predict(x))
    assert np.array_equal(pred["CoMModel_CoM"],com_model.predict(x))

    reference={}
    with gzip.open(RECORDS,"rt",encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"]=="Handmade":
                key=(int(row["writer"]),int(row["image_index"]))
                reference[key]=row
    assert len(reference)==2560
    for i,(writer,index,label) in enumerate(zip(writers,indices,y)):
        item=reference[(int(writer),int(index))]
        assert item["true"]==LABELS[int(label)]
        assert item["Normal_pred"]==LABELS[int(pred["Normal_raw"][i])]
        assert item["Center_of_Mass_pred"]==LABELS[int(pred["CoMModel_CoM"][i])]
    print("VERIFIED: 2560 previously published exact baseline/CoM predictions match",flush=True)

    groups=[("all",np.ones(len(y),dtype=bool)),
            ("digits_0-9",y<10),("letters_A-F",y>=10),
            ("true_5",y==5),("true_F",y==15),
            ("true_B",y==11),("true_8",y==8)]
    rows=[metric(y,pred[c],mask,g,c) for g,mask in groups for c in COLS]
    savecsv(out/"four_way_summary.csv",rows)
    print("FOUR-WAY CONTRAST: (fix weights for movement; fix input for fit)",flush=True)
    for group in ("all","digits_0-9","letters_A-F"):
        rs=[r for r in rows if r["group"]==group]
        print(f"{group}: "+", ".join(f"{r['condition']}={r['accuracy']:.6%}({r['correct']}/{r['n']})" for r in rs),flush=True)

    category_changes=[]
    for group,mask in groups:
        for a,b,label in [
            ("Normal_raw","Normal_CoM","Movement_same_normal_weights"),
            ("Normal_CoM","CoMModel_CoM","New_weights_same_centered_pixels"),
            ("Normal_raw","CoMModel_CoM","Actual_pipeline_delta"),
            ("CoMModel_raw","CoMModel_CoM","Movement_same_centered_weights_off_distribution"),
        ]:
            oldcorrect=(pred[a]==y)
            newcorrect=(pred[b]==y)
            m=mask
            category_changes.append({
                "group":group,"contrast":label,
                "source":a,"target":b,"n":int(m.sum()),
                "old_accuracy":float(oldcorrect[m].mean()),
                "new_accuracy":float(newcorrect[m].mean()),
                "delta_accuracy_pp":float(100*(newcorrect[m].mean()-oldcorrect[m].mean())),
                "corrected":int((m&~oldcorrect&newcorrect).sum()),
                "induced":int((m&oldcorrect&~newcorrect).sum()),
                "prediction_changes":int((m&(pred[a]!=pred[b])).sum()),
            })
    savecsv(out/"contrasts.csv",category_changes)

    # Actual 5->F correction, and true B that changes from correct B to wrong 8.
    wanted=[
        ("5_wrong_F_to_5",5,15,5),
        ("B_correct_to_wrong_8",11,11,8),
    ]
    cases=[]
    for scenario,true,before,after in wanted:
        hits=np.flatnonzero((y==true)&(pred["Normal_raw"]==before)&
                            (pred["CoMModel_CoM"]==after))
        assert len(hits)>0,(scenario,true,before,after)
        print(f"SCENARIO {scenario}: count={len(hits)}",flush=True)
        # Intentionally sample diverse writers when possible and include meaningful movements.
        picked=[]
        for writer in (41,42,43,44):
            matches=[int(i) for i in hits if writers[i]==writer]
            if matches:
                matches.sort(key=lambda i:(-np.linalg.norm(shifts[i]),int(indices[i])))
                picked.append(matches[0])
        for i in picked[:4]:
            cases.append((scenario,i,true,15 if true==5 else 8))
    print("SELECTED CASES:",[(sc,int(writers[i]),int(indices[i])) for sc,i,_,_ in cases],flush=True)

    igrows=[]
    gridfigs=[]
    for scenario,i,true,other in cases:
        raw=x[i].reshape(28,28);centered=moved[i].reshape(28,28)
        signed=[]
        for name,img,net in [
            ("Normal_raw",raw,net_base),
            ("Normal_CoM",centered,net_base),
            ("CoMModel_CoM",centered,net_com)]:
            ig,margin,resid=true_vs_other_ig(net,img,true,other)
            signed.append((name,ig,margin,resid))
            igrows.append({
                "scenario":scenario,
                "writer":int(writers[i]),"index":int(indices[i]),
                "true":LABELS[true],"competitor":LABELS[other],
                "model_condition":name,
                "prediction":LABELS[int(pred[name][i])],
                "true_minus_competitor_margin":margin,
                "IG_signed_sum":float(ig.sum()),
                "IG_positive_sum":float(np.clip(ig,0,None).sum()),
                "IG_negative_sum":float(np.clip(ig,None,0).sum()),
                "IG_max_abs":float(np.max(np.abs(ig))),
                "IG_completeness_residual":resid,
                "shift_x":int(shifts[i,0]),"shift_y":int(shifts[i,1])
            })
        gridfigs.append((scenario,i,raw,centered,signed))

    savecsv(out/"selected_ig_diagnostics.csv",igrows)

    # Image grids: one 5->F corrected and one B->8 induced per writer when available.
    for scenario in ("5_wrong_F_to_5","B_correct_to_wrong_8"):
        chosen=[x for x in gridfigs if x[0]==scenario]
        if not chosen:
            continue
        n=len(chosen)
        fig,axes=plt.subplots(n,5,figsize=(16,3.45*n),squeeze=False)
        for k,(_,i,raw,centered,signed) in enumerate(chosen):
            amps=max(float(np.abs(z[1]).max()) for z in signed)
            for col,view in ((0,raw),(1,centered)):
                ax=axes[k,col]
                ax.imshow(view,cmap="gray",vmin=0,vmax=255,interpolation="nearest")
                ax.set_xticks([]);ax.set_yticks([])
            for col,(name,ig,margin,resid) in enumerate(signed,start=2):
                ax=axes[k,col]
                ax.imshow(ig,cmap="RdBu_r",vmin=-amps,vmax=amps,
                          interpolation="nearest")
                ax.set_xticks([]);ax.set_yticks([])
                ax.set_title(f"{name}\npred {LABELS[int(pred[name][i])]}, "
                             f"margin {margin:+.2f}",fontsize=9)
            axes[k,0].set_title(f"Original W{writers[i]} #{indices[i]}\n"
                                f"pred {LABELS[int(pred['Normal_raw'][i])]}",fontsize=10)
            axes[k,1].set_title(f"CoM shifted dx={shifts[i,0]:+d} dy={shifts[i,1]:+d}\n"
                                f"pred {LABELS[int(pred['CoMModel_CoM'][i])]}",fontsize=10)
        true="5" if scenario.startswith("5") else "B"
        other="F" if true=="5" else "8"
        fig.suptitle(f"{scenario}: Signed IG for logit({true})−logit({other})\n"
                     "Red favors true label; blue favors competitor. "
                     "First compare col 3↔4 (movement), then col 4↔5 (training).",
                     fontsize=12)
        fig.tight_layout(rect=[0,0,1,.94])
        fig.savefig(out/f"IG_{scenario}.png",dpi=150)
        plt.close(fig)

    # 2x2 controlled prediction CSV; no labels used for interventions.
    perimage=[]
    for i in range(len(y)):
        record={"writer":int(writers[i]),"index":int(indices[i]),
                "true":LABELS[int(y[i])],
                "dx":int(shifts[i,0]),"dy":int(shifts[i,1])}
        for c in COLS:
            record[f"{c}_pred"]=LABELS[int(pred[c][i])]
            record[f"{c}_correct"]=int(pred[c][i]==y[i])
        perimage.append(record)
    savecsv(out/"four_way_predictions.csv",perimage)

    # True-5 and true-B class margin aggregate over all 160 true-class samples each.
    margins=[]
    for actual,other in ((5,15),(11,8)):
        ix=np.flatnonzero(y==actual)
        for name in COLS:
            val=output[name][ix,actual]-output[name][ix,other]
            margins.append({
                "true":LABELS[actual],"competitor":LABELS[other],
                "condition":name,"n":len(ix),
                "positive_pairwise_margin_fraction":float((val>0).mean()),
                "mean_true_minus_other_logit_margin":float(val.mean()),
                "median_true_minus_other_logit_margin":float(np.median(val)),
                "p10_margin":float(np.quantile(val,.1)),
                "p90_margin":float(np.quantile(val,.9)),
            })
    savecsv(out/"pairwise_margins.csv",margins)

    meta={"study":"EXP유004 CoM pixel decision diagnostic","mode":"review-only",
          "handmade_n":2560,"official_test_read":False,"professor_test_read":False,
          "model_fit_performed":False,"new_preprocessing_parameter_search":False,
          "models_sha256":{"Normal":NORMAL_SHA,"CoM":COM_SHA},
          "fixed_target_comparisons":["5_minus_F","B_minus_8"],
          "ig_reference":"all-zero input",
          "ig_approximation":"128 midpoint",
          "ig_count":len(igrows),
          "selected_examples":len(cases),
          "versions":{"python":platform.python_version(),"numpy":np.__version__,
                      "scikit_learn":sklearn.__version__}}
    (out/"metadata.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    overview=["# EXP유004 follow-up — CoM decision diagnostic (review only)",
              "",
              "- The same 2,560 Handmade files (not independent unseen-writer Test).",
              "- No new training, no Test, no hyperparameter tuning, no model deployment.",
              "- Compare Normal_raw vs Normal_CoM: input shift with identical weights.",
              "- Compare Normal_CoM vs CoMModel_CoM: weights difference on identical centered images.",
              "- CoMModel_raw is **off the new model training distribution**, diagnostic only.",
              "- IG uses signed margin true-class logit minus fixed rival class (5-F or B-8).",
              "- Red IG region supports true class *relative to rival*, blue supports rival.",
              "- A pixel IG map is not evidence of human-like shape understanding or proof of causality.",
              "",
              "| Group | Normal/raw | Normal/CoM shift | CoM-model/raw (diagnostic) | CoM-model/CoM |",
              "|---|---:|---:|---:|---:|"]
    for group in ("all","digits_0-9","letters_A-F","true_5","true_B"):
        rr=[next(r for r in rows if r["group"]==group and r["condition"]==name) for name in COLS]
        overview.append(f"| {group} | "+" | ".join(f"{r['accuracy']:.2%}" for r in rr)+" |")
    overview+=["","## Paired contrasts (causal controlled manipulations on the same samples)"]
    for group in ("digits_0-9","letters_A-F"):
        overview.append(f"### {group}")
        overview.append("| Comparison | Accuracy delta (pp) | Corrected | New errors |")
        overview.append("|---|---:|---:|---:|")
        for r in category_changes:
            if r["group"]==group and r["contrast"] in [
                "Movement_same_normal_weights","New_weights_same_centered_pixels","Actual_pipeline_delta"]:
                overview.append(f"| {r['contrast']} | {r['delta_accuracy_pp']:+.3f} | "
                                f"{r['corrected']} | {r['induced']} |")
    overview+=["","## Selected pairs",""]
    for scenario,_,_,_ in cases:
        pass
    overview.append(f"- True 5, baseline F → CoM model 5: "
                    f"{sum(1 for s,_,_,_ in cases if s=='5_wrong_F_to_5')} illustrated examples.")
    overview.append(f"- True B, baseline B → CoM model 8: "
                    f"{sum(1 for s,_,_,_ in cases if s=='B_correct_to_wrong_8')} illustrated examples.")
    overview+=["","## Files",
               "- four_way_summary.csv, contrasts.csv: exact paired group results",
               "- four_way_predictions.csv: same-image 2×2 controls for all 2,560",
               "- pairwise_margins.csv: true 5-vs-F and B-vs-8 logit margins",
               "- selected_ig_diagnostics.csv: per-example signed IG and completeness",
               "- IG_5_wrong_F_to_5.png; IG_B_correct_to_wrong_8.png: actual image/IG grids",
               "- metadata.json: inputs, checkpoint hashes, no-Test confirmation"]
    (out/"SUMMARY.md").write_text("\n".join(overview)+"\n",encoding="utf-8")
    print("\n".join(overview[:35]),flush=True)
    print("DIAGNOSTIC COMPLETE; generated files:",sorted(p.name for p in out.iterdir()),flush=True)


if __name__=="__main__":
    main()
