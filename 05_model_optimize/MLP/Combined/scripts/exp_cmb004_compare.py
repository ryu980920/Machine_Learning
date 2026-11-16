#!/usr/bin/env python3
"""Aggregate 4 newly trained geometry conditions into the preregistered comparison.

Never fit, tune, or select a model. This program presents paired evidence and
counterexamples, explicitly separating magnitude, error changes, official val
cost, and interaction from any final scientific judgement.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

ROOT=Path(__file__).resolve().parents[4]
CMB=ROOT/"05_model_optimize"/"MLP"/"Combined"
sys.path.insert(0,str(CMB/"scripts"))
import exp_cmb002_evidence as core

METHODS=("bbox_center","bbox_size","com_center","size_then_com")
LABELS=core.LABELS
WRITERS=core.WRITERS

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def write(path,rows):
    assert rows,path
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def num(x):
    return float(x)

def mean(rows,key):
    return sum(float(r[key]) for r in rows)/len(rows) if rows else float("nan")

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    out=args.out
    results={}
    for m in METHODS:
        part=out/m
        manifest=json.loads((part/"run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["method"]==m and manifest["no_test_used"]
        results[m]={
            "metrics":read(part/"metrics.csv"),
            "writer":read(part/"writer_metrics.csv"),
            "official":read(part/"official_val_metrics.csv"),
            "class":read(part/"class_metrics.csv"),
            "group":read(part/"group_metrics.csv"),
            "pairs":read(part/"pair_errors.csv"),
            "geo":read(part/"geometry_diagnostics.csv"),
            "focus":read(part/"focus_mechanism.csv"),
            "paired":read(part/"paired_predictions.csv"),
            "fit":read(part/"fit_metrics.csv"),
            "manifest":manifest}
        assert len(results[m]["paired"])==2560
        assert len(results[m]["focus"])==640
        assert len(results[m]["fit"])==4
    with (CMB/"results"/"EXP_CMB_001"/"metrics_summary.csv").open(encoding="utf-8-sig") as f:
        baseline=list(csv.DictReader(f))
    old_dev={int(r["held_out_writer"]):float(r["accuracy"]) for r in baseline
             if r["dataset"]=="official_dev" and r["model"]=="Combined"}
    assert set(old_dev)==set(WRITERS)
    _,predtable,oldtab,_,_=core.load_inputs()
    base_y=np.array([LABELS.index(r["true_label"]) for r in predtable])
    base_p=np.array([LABELS.index(r["combined_pred"]) for r in predtable])
    assert np.mean(base_y==base_p)==.8125

    summary=[{"method":"baseline","accuracy":.8125,"macro_f1":.8120430780652209,
      "correct":2080,"corrected":0,"induced":0,"delta_correct":0,"n":2560,
      "min_official_delta_pp":0,"max_official_delta_pp":0,
      "n_writers_improved":0,"n_writers_worsened":0,
      "mean_training_seconds_per_fold":"", "mean_n_iterations":""}]
    paired_all=[]
    writer_all=[];paired_by_method={}
    for method,d in results.items():
        r=d["metrics"][0]
        ov=[{"writer":int(z["writer"]),"delta_pp":100*(float(z["accuracy"])-old_dev[int(z["writer"])])}
             for z in d["official"]]
        wr=[{"writer":int(z["writer"]),"old":float(z["baseline_accuracy"]),
             "new":float(z["accuracy"])} for z in d["writer"]]
        summary.append({"method":method,
          "accuracy":r["accuracy"],"macro_f1":r["macro_f1"],"correct":r["correct"],
          "corrected":r["corrected"],"induced":r["induced"],
          "delta_correct":r["delta_correct"],"n":2560,
          "min_official_delta_pp":min(x["delta_pp"] for x in ov),
          "max_official_delta_pp":max(x["delta_pp"] for x in ov),
          "n_writers_improved":sum(x["new"]>x["old"] for x in wr),
          "n_writers_worsened":sum(x["new"]<x["old"] for x in wr),
          "mean_training_seconds_per_fold":mean(d["fit"],"fit_seconds"),
          "mean_n_iterations":mean(d["fit"],"n_iter")})
        writer_all.extend({"method":method,**z,"official_acc_delta_pp":
           next(x["delta_pp"] for x in ov if x["writer"]==int(z["writer"]))}
            for z in d["writer"])
        pp=d["paired"]
        for i,z in enumerate(pp):
            assert int(z["writer"])==int(predtable[i]["writer"])
            assert int(z["index_in_writer"])==int(predtable[i]["index_in_writer"])
            assert z["actual"]==predtable[i]["true_label"]
            assert z["baseline_pred"]==predtable[i]["combined_pred"]
        paired_by_method[method]=np.array([LABELS.index(x["prediction"]) for x in pp])
        paired_all.extend(pp)
    write(out/"method_comparison.csv",summary)
    write(out/"writer_comparison.csv",writer_all)

    # Aggregate all 4*4 directional pairs and 4 writers, preserving both errors and corrected/induced counts.
    errors=[]
    for method,d in results.items():
        for t,c in (("0","D"),("D","0"),("8","B"),("B","8")):
            z=[r for r in d["pairs"] if r["actual"]==t and r["rival"]==c]
            assert len(z)==4
            errors.append({"method":method,"actual":t,"rival":c,
                "baseline_errors":sum(int(r["baseline_errors"]) for r in z),
                "new_errors":sum(int(r["new_errors"]) for r in z),
                "delta_errors":sum(int(r["new_errors"])-int(r["baseline_errors"]) for r in z)})
    write(out/"main_error_pairs.csv",errors)

    group_rows=[]
    for method,d in results.items():
        for r in d["group"]:
            group_rows.append({"method":method,**r,
               "delta_accuracy_pp":100*(float(r["accuracy"])-float(r["baseline_accuracy"]))})
    write(out/"group_comparison.csv",group_rows)

    geometry=[]
    for method,d in results.items():
        for r in d["geo"]:
            geometry.append({"method":method,**r})
    write(out/"geometry_comparison.csv",geometry)

    # Scientific diagnostic: magnitude/outcome association within each
    # writer and actual-class; intentionally NOT using it to tune transforms.
    strata=[]
    for method,d in results.items():
        focus=d["focus"]
        for writer in WRITERS:
            for t in ("0","D","8","B"):
                subset=[r for r in focus if int(r["writer"])==writer and r["actual"]==t]
                assert len(subset)==40
                for state in ("corrected","new_error","both_correct","persistent_error"):
                    selected=[r for r in subset if r["transition"]==state]
                    if not selected:continue
                    abs_ratio=[abs(float(r["scale_factor"])-1) for r in selected]
                    abs_com=[abs(float(r["com_shift_px"])) for r in selected]
                    strata.append({"method":method,"writer":writer,"actual":t,
                        "transition":state,"count":len(selected),
                        "mean_abs_scale_factor_change":float(np.mean(abs_ratio)),
                        "mean_com_shift_px":float(np.mean(abs_com)),
                        "mean_bbox_long_change_px":mean(selected,"bbox_change"),
                        "mean_logit_margin_change":mean(selected,"margin_change"),
                        "mean_ink_mass_change_ratio":mean(selected,"mass_change_ratio")})
    write(out/"mechanism_by_stratum.csv",strata)

    # Changes are not caused solely by the image transform because weights are
    # re-fitted. This factorial contrast is descriptive of full train/infer pipelines.
    p0=base_p
    c=paired_by_method["bbox_size"]
    d=paired_by_method["com_center"]
    e=paired_by_method["size_then_com"]
    interaction=[]
    for kind,z in (("all",np.ones(2560,dtype=bool)),
                   ("digits",base_y<10),("letters",base_y>=10),
                   ("0",base_y==0),("D",base_y==13),
                   ("8",base_y==8),("B",base_y==11)):
        acc=lambda pred:float(np.mean(pred[z]==base_y[z]))
        interaction.append({"group":kind,"baseline":acc(p0),"size":acc(c),
                            "com":acc(d),"size_then_com":acc(e),
                            "descriptive_interaction_E_minus_C_minus_D_plus_A":
                            acc(e)-acc(c)-acc(d)+acc(p0)})
    write(out/"factorial_interaction.csv",interaction)

    fig,ax=plt.subplots(figsize=(9.5,4.8))
    xs=np.arange(len(summary))
    ax.plot(xs,[float(r["accuracy"])*100 for r in summary],marker="o",label="Hand-made OOF")
    ax.set(xticks=xs,xticklabels=[r["method"] for r in summary],
       ylabel="OOF Accuracy (%)",title="EXP-CMB-004 | same 4-Fold writer-held-out protocol")
    plt.setp(ax.get_xticklabels(),rotation=20,ha="right")
    ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/"accuracy_comparison.png",dpi=170);plt.close(fig)

    # Report transparency: no numerical gate, no automatic 'winner' label.
    lines=[
        "# EXP-CMB-004 — BBox Size vs CoM Center: 실측 비교 요약",
        "",
        "**통과 기준:** 사전등록 문서에 따라 특정 점수 이상이 아니라 어느 방법이 어떤 오류에 왜 효과적인지 **근거와 반례를 함께 설명**하는 것. 이 자동 보고서는 숫자만 제시하며 최종 연구 판정은 별도 실험일지에 작성한다.",
        "",
        "## 1. 전체 OOF (16 classes, 4 heldout writers)",
        "",
        "| Method | Accuracy | Macro F1 | Δ correct vs baseline | Corrected | Newly wrong | Writers improved | Official Val worst Δpp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append("| {} | {:.3%} | {:.4f} | {} | {} | {} | {} | {} |".format(
            r["method"],float(r["accuracy"]),float(r["macro_f1"]),
            r["delta_correct"],r["corrected"],r["induced"],r["n_writers_improved"],
            "—" if r["method"]=="baseline" else "{:+.3f}".format(float(r["min_official_delta_pp"]))))
    lines+=["","## 2. 주요 오답 방향별 Count","","| Method | 0→D | D→0 | 8→B | B→8 | 합계 |",
           "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        arr=[next(r for r in errors if r["method"]==method and r["actual"]==a and r["rival"]==b)
            for a,b in (("0","D"),("D","0"),("8","B"),("B","8"))]
        counts=[int(r["new_errors"]) for r in arr]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(method,*counts,sum(counts)))
    lines+=["","## 3. Mechanism: 입력에서 실제로 변화한 특징",
           "",
           "| Method | Official train BBox long-side std before→after | Hand-made mean CoM offset before→after | Mean image mass change (Hand-made pooled by writers) |",
           "|---|---|---|---:|"]
    for method in METHODS:
        g=[r for r in geometry if r["method"]==method]
        off=next(x for x in g if x["sample"]=="official_train")
        hh=[x for x in g if x["sample"].startswith("writer_")]
        ink=sum(float(x["mean_abs_ink_mass_fraction_change"]) for x in hh)/4
        before=sum(float(x["mean_com_offset_before"]) for x in hh)/4
        after=sum(float(x["mean_com_offset_after"]) for x in hh)/4
        lines.append("| {} | {:.3f}→{:.3f} | {:.3f}→{:.3f}px | {:.4f} |".format(
           method,float(off["std_long_before"]),float(off["std_long_after"]),before,after,ink))
    lines+=["","## 4. Interaction (descriptive: 4 separate retrained models)",
        "",
        "| Group | E − Size − CoM + Baseline (accuracy percentage points) |",
        "|---|---:|"]
    for r in interaction:
        lines.append("| {} | {:+.3f} |".format(
           r["group"],100*r["descriptive_interaction_E_minus_C_minus_D_plus_A"]))
    lines+=["","### Interpretations that still need to be made",
        "",
        "- Size vs CoM must be interpreted using error-direction and writer-level transitions together with the actual size/center displacement and margin changes, not Accuracy rank alone.",
        "- Mean input mass change during scaling is NOT pure clipping loss: bilinear resizing changes area and intensity mass. Interpretation must separate scaling from cropping.",
        "- The combined-condition contrast is descriptive of retrained pipelines, NOT a clean causal estimate that holds model weights fixed.",
        "- Hypothesis evidence is developmental OOF on the same four previously examined writers; new-writer external verification is still missing.",
        "- Original Test and professor Test were never accessed.",
        "",
        "See: writer_comparison.csv, group_comparison.csv, main_error_pairs.csv, mechanism_by_stratum.csv, geometry_comparison.csv, factorial_interaction.csv and each method subfolder's actual PNG/CSV/models.",
        ""]
    (out/"SUMMARY.md").write_text("\n".join(lines),encoding="utf-8")
    (out/"AGGREGATE_MANIFEST.json").write_text(json.dumps({
        "experiment_id":"EXP-CMB-004",
        "methods":METHODS,"comparison_criterion":"explanatory evidence, no accuracy/F1 gate",
        "writer_heldout":[41,42,43,44],"official_test_used":False,
        "professor_final_test_used":False,"completed_method_count":len(results),
        "trained_model_count":len(results)*4,
        "baseline_retrained":False,
        "size_target":results["bbox_size"]["manifest"]["target_longside_pixels"],
        "method_model_member_hashes":{m:results[m]["manifest"]["trained_models_sha256"] for m in METHODS}
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print((out/"SUMMARY.md").read_text(encoding="utf-8"),flush=True)

if __name__=="__main__":
    main()
