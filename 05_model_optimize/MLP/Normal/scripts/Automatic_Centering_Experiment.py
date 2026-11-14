#!/usr/bin/env python3
"""EXP유004 review-only: Normal vs centered MLPs, paired Official Val + Handmade.

Never reads official Test. Does not publish results or commit trained models.
Uses EXACT preexisting baseline (EXP유002 frozen joblib) and refits two
unchanged sklearn MLPClassifier(random_state=42) on the same 76,800 official
Training examples, adding only a stateless centering transform to each new
Pipeline. Handmade is NEVER used in fit or preprocessing parameter selection.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import scipy
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE))
from Pixcel_Attribution import (fixed_split, DEFAULT_DATA, read_handmade,
    prior_official_cm, prior_handmade, LABELS, build_baseline)
from Automatic_Centering_Transformer import (AutomaticCentering,
    raw_centers, center_shifts, center_batch, test_centering, TARGET, THRESHOLD)
from Input_Sensitivity import transform, Variant
# Pixcel_Attribution imports 'common' which exposes the actual baseline scaler.
from common import pipeline
from pixel_transform import PixelScaler255

STAGE3_MODEL=ROOT/"05_model_optimize"/"MLP"/"Normal"/"results"/"Input_Sensitivity"/"normal_mlp_refit_stage3.joblib"
EXPECTED_STAGE3_SHA="9b54ba9c60e389575200d88ad0b93c47ec47f9c673eebeb755ff47e3ebe23b0a"
CONDITIONS=("Normal","Center_of_Mass","Bounding_Box_Center")
CONFUSIONS=(("5","F"),("B","8"),("0","D"),("9","4"),("E","C"),("4","A"))
WRITERS=(41,42,43,44)

def checksum(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def write_csv(p,rows):
    if not rows:
        raise ValueError("No rows to write "+str(p))
    with p.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def score_row(domain,method,y,pred,mask,group_type,group,base_pred):
    y,pred,base_pred=y[mask],pred[mask],base_pred[mask]
    a=(pred==y)
    old=(base_pred==y)
    n=len(y)
    assert n>0
    cm=confusion_matrix(y,pred,labels=np.arange(16))
    return {
        "dataset":domain, "method":method,"group_type":group_type, "group":str(group),
        "n":n,"accuracy":float(a.mean()),
        "macro_f1":float(f1_score(y,pred,average="macro",labels=np.arange(16),zero_division=0)),
        "macro_precision":float(precision_score(y,pred,average="macro",labels=np.arange(16),zero_division=0)),
        "macro_recall":float(recall_score(y,pred,average="macro",labels=np.arange(16),zero_division=0)),
        "correct":int(a.sum()),
        "delta_correct_vs_baseline":int(a.sum()-old.sum()),
        "accuracy_delta_pp_vs_baseline":float(100*(a.sum()-old.sum())/n),
        "corrected_errors":int((~old & a).sum()),
        "induced_errors":int((old & ~a).sum()),
        "changed_predictions":int((base_pred!=pred).sum()),
        "prediction_change_rate":float((base_pred!=pred).mean()),
        "original_baseline_correct":int(old.sum()),
        "confusion_off_diagonal":int(n-np.trace(cm))
    }

def per_group(domain, y, writers, predictions):
    baseline=predictions["Normal"]
    ones=np.ones(len(y),dtype=bool)
    rows=[]
    for name,pred in predictions.items():
        rows.append(score_row(domain,name,y,pred,ones,"all","all",baseline))
        for c in range(16):
            rows.append(score_row(domain,name,y,pred,y==c,"class",LABELS[c],baseline))
        if domain=="Handmade":
            for kind,mask in [
                ("digits",y<10),("letters",y>=10)
            ]:
                rows.append(score_row(domain,name,y,pred,mask,"category",kind,baseline))
            for writer in WRITERS:
                m=writers==writer
                rows.append(score_row(domain,name,y,pred,m,"writer",str(writer),baseline))
                for typ,cm in [("digits",y<10),("letters",y>=10)]:
                    rows.append(score_row(domain,name,y,pred,m&cm,"writer_category",f"{writer}_{typ}",baseline))
    return rows

def alignment_rows(X, domain):
    """Quantify centering effects without labels or writer information."""
    a=np.asarray(X,dtype=np.float32).reshape(-1,784)
    rows=[]
    pixel_mass=a.sum(axis=1,dtype=np.float64)
    for method in ("mass","bbox"):
        shifts=center_shifts(a,method)
        aligned=center_batch(a,method)
        old_x,old_y=raw_centers(a,method)
        new_x,new_y=raw_centers(aligned,method)
        old_err=np.sqrt((old_x-TARGET[0])**2+(old_y-TARGET[1])**2)
        new_err=np.sqrt((new_x-TARGET[0])**2+(new_y-TARGET[1])**2)
        lost=np.maximum(0,pixel_mass-aligned.sum(axis=1,dtype=np.float64))
        fraction=lost/np.maximum(pixel_mass,1)
        rows.append({
            "dataset":domain,"centering_method":method,"n":len(a),
            "mean_center_error_before_px":float(old_err.mean()),
            "mean_center_error_after_px":float(new_err.mean()),
            "median_center_error_after_px":float(np.median(new_err)),
            "p95_center_error_after_px":float(np.quantile(new_err,.95)),
            "mean_absolute_dx":float(np.abs(shifts[:,0]).mean()),
            "mean_absolute_dy":float(np.abs(shifts[:,1]).mean()),
            "max_absolute_shift":int(np.abs(shifts).max()),
            "images_with_nonzero_ink_loss":int(np.count_nonzero(lost>0.01)),
            "fraction_with_nonzero_ink_loss":float(np.mean(lost>0.01)),
            "mean_lost_ink_fraction":float(fraction.mean()),
            "p95_lost_ink_fraction":float(np.quantile(fraction,.95)),
            "max_lost_ink_fraction":float(fraction.max()),
            "images_with_remaining_center_error_over_1px":int((new_err>1).sum()),
        })
    return rows

def paired_predictions(domain,y,writers,indices,preds,out_file):
    names=list(preds)
    assert names==list(CONDITIONS)
    with gzip.open(out_file,"wt",encoding="utf-8",newline="") as f:
        cols=["dataset","writer","image_index","true"]
        for name in names:
            cols.extend([f"{name}_pred",f"{name}_correct"])
        w=csv.DictWriter(f,fieldnames=cols)
        w.writeheader()
        for k in range(len(y)):
            r={"dataset":domain,"writer":int(writers[k]),
               "image_index":int(indices[k]),"true":LABELS[int(y[k])]}
            for name in names:
                r[f"{name}_pred"]=LABELS[int(preds[name][k])]
                r[f"{name}_correct"]=int(preds[name][k]==y[k])
            w.writerow(r)

def paired_confusions(y,preds,domain):
    rows=[]
    for name,values in preds.items():
        cm=confusion_matrix(y,values,labels=np.arange(16))
        for i,j in np.ndindex(cm.shape):
            rows.append({"dataset":domain,"method":name,
                         "true":LABELS[i],"predicted":LABELS[j],"count":int(cm[i,j])})
    return rows

def plot_scores(rows,out):
    allrows=[r for r in rows if r["group_type"]=="all"]
    fig, ax=plt.subplots(figsize=(9,4.7))
    x=np.arange(len(CONDITIONS))
    for j,domain in enumerate(("OfficialVal","Handmade")):
        vals=[next(r["accuracy"] for r in allrows if r["dataset"]==domain and r["method"]==name) for name in CONDITIONS]
        ax.bar(x+(-.19 if j==0 else .19), np.asarray(vals)*100, width=.37,label=domain)
        for xi,v in zip(x+(-.19 if j==0 else .19),vals):
            ax.text(xi,100*v+.35,f"{100*v:.2f}%",ha="center",fontsize=9)
    ax.set_xticks(x,["Normal","Center of Mass","Bounding Box"])
    ax.set_ylim(0,105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("EXP004 — fixed Normal vs centering-before-training MLPs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"accuracy_comparison.png",dpi=145)
    plt.close(fig)

def summary_text(rows,align,warnings_by_model,off_cm_delta):
    wanted=[r for r in rows if r["group_type"]=="all"]
    txt=["# EXP유004 — Automatic_Centering (review-only, measured results)","",
         "**No official Test, no professor Test, and no Handmade training or hyperparameter selection.**",
         "Same 76,800 Official Training examples and unchanged MLPClassifier(random_state=42).",
         "The current Normal model is the previously validated frozen Stage 3 checkpoint; the two centering models are freshly fit.",
         "Training and inference use the SAME centering method in each new Pipeline.",
         "Preset geometric target = (13.5,13.5) pixel coordinates; threshold=32; integer-pixel translation, no interpolation.",
         "",
         "## Official Val and Handmade — paired accuracy / corrections",
         "| Dataset | Model | Accuracy | Δ baseline (pp) | Corrected | Newly wrong | Macro F1 |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for domain in ("OfficialVal","Handmade"):
        for name in CONDITIONS:
            r=next(a for a in wanted if a["dataset"]==domain and a["method"]==name)
            txt.append(f"| {domain} | {name} | {r['accuracy']:.4%} | {r['accuracy_delta_pp_vs_baseline']:+.3f} | {r['corrected_errors']} | {r['induced_errors']} | {r['macro_f1']:.4f} |")
    txt+=["","## Alignment distortion and consistency",
          "| Data | Center | Mean center error before→after | Images with ink loss | Mean lost ink fraction |",
          "|---|---|---|---:|---:|"]
    for a in align:
        txt.append(f"| {a['dataset']} | {a['centering_method']} | {a['mean_center_error_before_px']:.3f}→{a['mean_center_error_after_px']:.3f}px | {a['images_with_nonzero_ink_loss']}/{a['n']} | {a['mean_lost_ink_fraction']:.5f} |")
    txt+=["","## Generalization/interpretation safeguards",
          "- This four-writer Handmade set has already been repeatedly inspected in EXP유001-003. Improvements here are **exploratory**, NOT evidence of unseen-writer Test accuracy.",
          "- Official Val was also used in historical research; do not claim completely untouched test performance.",
          "- Since the baseline is already strong on official Validation, inspect the domain gap and class/writer strata, not a tiny official accuracy gain alone.",
          "- Centering can shift, clip or reshape which input pixels the MLP sees. Here only integer translation with zero-fill is used: measure ink clipping separately.",
          "- Class labels/writer IDs are for evaluation ONLY. Neither centering transformer uses them.",
          "- EXP유004 does not determine a final deployed preprocessing policy; await review and independent writer evaluation.",
          f"- Historical baseline Official CM L1 difference: {off_cm_delta}.",
          "",
          "## Files",
          "- exp004_group_metrics.csv: official and Handmade full/class/writer/category paired measures.",
          "- exp004_predictions.csv.gz: per-image predictions (OfficialVal + Handmade) for all 3 models.",
          "- exp004_confusions.csv: 16×16 confusion entries for all models and both domains.",
          "- exp004_alignment_stats.csv: target-centering accuracy, translation and stroke clipping.",
          "- exp004_typical_confusions.csv: selected 5/F, B/8, 0/D confusion changes.",
          "- accuracy_comparison.png: performance chart.",
          "- exp004_run_metadata.json: SHA256 hashes, software versions, training times and fit boundaries.",
          "- *_{mass,bbox}.joblib: trained candidate Pipelines, user review artifact only.",
          ""]
    return "\n".join(txt)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    started=time.monotonic()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    test_centering()
    assert checksum(STAGE3_MODEL)==EXPECTED_STAGE3_SHA
    xtr,ytr,xva,yva,sha=fixed_split(DEFAULT_DATA)
    xhm,yhm,whm,ih=read_handmade()
    ntr,nva,nh=len(ytr),len(yva),len(yhm)
    assert (ntr,nva,nh)==(76800,19200,2560)
    print("DATA VERIFIED: Official Training 76800; Official Validation 19200; Handmade 2560, 4 writers",flush=True)
    baseline=joblib.load(STAGE3_MODEL)  # Checksum verified before deserialization
    assert baseline.named_steps["model"].get_params()==build_baseline()[0].get_params()
    predv0=baseline.predict(xva)
    predh0=baseline.predict(xhm)
    cm_l1=int(np.abs(confusion_matrix(yva,predv0,labels=np.arange(16))-prior_official_cm()).sum())
    hist=prior_handmade()
    hmis=sum(hist[(int(w),int(idx))]!=(LABELS[int(y)],LABELS[int(p)])
             for w,idx,y,p in zip(whm,ih,yhm,predh0))
    assert cm_l1==0 and hmis==0
    assert abs(accuracy_score(yva,predv0)-.9731770833333333)<1e-12
    assert abs(accuracy_score(yhm,predh0)-.628515625)<1e-12
    print("BASELINE VERIFIED: Official Val 97.317708%, Handmade 62.8515625%, CM L1=0, handwritten mismatch=0",flush=True)

    predict={"OfficialVal":{"Normal":predv0}, "Handmade":{"Normal":predh0}}
    training_times={}
    run_warnings={}
    for name,method in (("Center_of_Mass","mass"),("Bounding_Box_Center","bbox")):
        print(f"TRAINING {name} same split + unchanged MLP hyperparameters...",flush=True)
        estimator,scaled=build_baseline()
        assert scaled
        estimator_params=estimator.get_params()
        model=Pipeline([
            ("centering",AutomaticCentering(method=method)),
            ("scale_0_1",PixelScaler255()),
            ("model",estimator),
        ])
        t0=time.monotonic()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(xtr,ytr)
        elapsed=time.monotonic()-t0
        training_times[name]=float(elapsed)
        run_warnings[name]=[f"{w.category.__name__}: {w.message}" for w in captured]
        assert model.named_steps["model"].get_params()==estimator_params
        pv=model.predict(xva)
        ph=model.predict(xhm)
        predict["OfficialVal"][name]=pv
        predict["Handmade"][name]=ph
        joblib.dump(model,out/f"exp004_{method}.joblib",compress=3)
        print(f"TRAINED {name} in {elapsed:.1f}s: Official={accuracy_score(yva,pv):.6%}, Handmade={accuracy_score(yhm,ph):.6%}",flush=True)

    metrics=[]
    for domain,y,writers in (("OfficialVal",yva,np.full(nva,-1,dtype=np.int32)),
                              ("Handmade",yhm,whm)):
        metrics.extend(per_group(domain,y,writers,predict[domain]))
    write_csv(out/"exp004_group_metrics.csv",metrics)
    with gzip.open(out/"exp004_predictions.csv.gz","wt",encoding="utf-8",newline="") as f:
        cols=["dataset","writer","image_index","true"]
        for name in CONDITIONS:
            cols += [f"{name}_pred", f"{name}_correct"]
        csvwriter=csv.DictWriter(f,fieldnames=cols)
        csvwriter.writeheader()
        for domain,y,writers,indices in [
            ("OfficialVal",yva,np.full(nva,-1,dtype=np.int32),np.arange(nva)),
            ("Handmade",yhm,whm,ih)]:
            preds=predict[domain]
            for k in range(len(y)):
                row={"dataset":domain,"writer":int(writers[k]),"image_index":int(indices[k]),
                     "true":LABELS[int(y[k])]}
                for name in CONDITIONS:
                    row[f"{name}_pred"]=LABELS[int(preds[name][k])]
                    row[f"{name}_correct"]=int(preds[name][k]==y[k])
                csvwriter.writerow(row)
    confusions=[]
    for domain,y in (("OfficialVal",yva),("Handmade",yhm)):
        confusions.extend(paired_confusions(y,predict[domain],domain))
    write_csv(out/"exp004_confusions.csv",confusions)

    important=[]
    for domain,y in (("OfficialVal",yva),("Handmade",yhm)):
        for name in CONDITIONS:
            cm=confusion_matrix(y,predict[domain][name],labels=np.arange(16))
            for actual,predicted in CONFUSIONS:
                a,b=LABELS.index(actual),LABELS.index(predicted)
                important.append({"dataset":domain,"method":name,"actual":actual,
                                  "predicted":predicted,"errors":int(cm[a,b])})
    write_csv(out/"exp004_typical_confusions.csv",important)
    print("Calculating accuracy of centering and foreground ink-loss diagnostics...",flush=True)
    align=[]
    for domain,x in (("OfficialTrain",xtr),("OfficialVal",xva),("Handmade",xhm)):
        align.extend(alignment_rows(x,domain))
    write_csv(out/"exp004_alignment_stats.csv",align)
    plot_scores(metrics,out)
    meta={
        "study":"EXP유004 Automatic_Centering",
        "status":"review-only; NOT published to Normal results/logs",
        "baseline_model_sha256":EXPECTED_STAGE3_SHA,
        "official_npz_sha256":sha,
        "official_test_read":False,
        "professor_test_read":False,
        "handmade_used_in_fit":False,
        "handmade_used_for_preprocessing_policy":False,
        "official_train":ntr,"official_validation":nva,"handmade":nh,
        "handmade_writer_ids":list(WRITERS),
        "normal_previously_frozen":True,
        "new_models_fit_on_official_train":["Center_of_Mass","Bounding_Box_Center"],
        "baseline_historical_official_cm_l1":cm_l1,
        "baseline_historical_handmade_prediction_mismatches":hmis,
        "center_policy":{"methods":["mass","bbox"],"target_xy":list(TARGET),
                         "threshold":THRESHOLD,"translation":"integer np.rint shifts, zero-filled, no wrap or interpolation",
                         "handmade_tuning":False},
        "model_hyperparameters":{k:repr(v) for k,v in build_baseline()[0].get_params().items()},
        "candidate_training_seconds":training_times,
        "candidate_training_warnings":run_warnings,
        "software":{"python":platform.python_version(),"numpy":np.__version__,
                    "sklearn":sklearn.__version__,"scipy":scipy.__version__,
                    "matplotlib":matplotlib.__version__},
        "elapsed_seconds":float(time.monotonic()-started),
    }
    (out/"exp004_run_metadata.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"SUMMARY.md").write_text(summary_text(metrics,align,run_warnings,cm_l1),encoding="utf-8")
    print("EXP004 COMPARISON COMPLETE",flush=True)
    print("------ MAIN RESULTS ------",flush=True)
    for domain in ("OfficialVal","Handmade"):
        for name in CONDITIONS:
            r=next(rr for rr in metrics if rr["dataset"]==domain and rr["method"]==name and rr["group_type"]=="all")
            print(f"{domain} {name}: acc={r['accuracy']:.8%}, delta_pp={r['accuracy_delta_pp_vs_baseline']:+.4f}, corrected={r['corrected_errors']}, induced={r['induced_errors']}, macro_F1={r['macro_f1']:.5f}",flush=True)
    for method in ("mass","bbox"):
        for a in align:
            if a["centering_method"]==method:
                print(f"ALIGNMENT {a['dataset']} {method}: center_err_px {a['mean_center_error_before_px']:.4f}->{a['mean_center_error_after_px']:.4f}, ink_loss={a['images_with_nonzero_ink_loss']}/{a['n']}",flush=True)
    print(f"TRAINING TIMES={training_times}; TOTAL SEC={meta['elapsed_seconds']:.1f}",flush=True)

if __name__=="__main__":
    main()
