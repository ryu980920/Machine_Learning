#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP013/014: aggregate independent three-seed measurements and family-probability robustness."""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path
def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
def save(path,rows):
    if not rows:raise RuntimeError(f"No rows for {path}")
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
def avg(nums):
    return sum(nums)/len(nums)
def run(source,out):
    source=Path(source);out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    exp013=next(source.glob("exp013-binary*/binary_probe_metrics.csv"))
    binary=read(exp013)
    binary_final=[r for r in binary if r["group"]=="all"]
    save(out/"exp013_binary_probe_scores.csv",binary_final)
    root=source
    exps=[root/f"exp014-seed{s}" for s in (42,43,44)]
    for p in exps:
        assert (p/"accuracy_metrics.csv").is_file(),p
        assert (p/"scale_image_family_probabilities.csv").is_file(),p
        m=json.loads((p/"manifest.json").read_text(encoding="utf-8"))
        assert m["original_checkpoint_hash_and_predictions_verified"]
        assert not m["original_test_used"] and not m["professor_test_used"]
        assert not m["handmade_used_in_fit"] and not m["validation_used_in_fit"]
    all_m=[row for p in exps for row in read(p/"accuracy_metrics.csv")]
    save(out/"exp014_all_seed_metrics.csv",all_m)
    by={}
    for r in all_m:
        if r["group"] not in ("all","digits","letters"):continue
        key=(r["model"],r["dataset"],r["group"])
        by.setdefault(key,[]).append(float(r["accuracy"]))
    summary=[]
    for (model,dataset,group),values in sorted(by.items()):
        assert len(values)==3,(model,dataset,group,len(values))
        summary.append({"model":model,"dataset":dataset,"group":group,
            "n_seeds":3,"mean_accuracy":avg(values),
            "std_accuracy":statistics.stdev(values),
            "seed42":values[0],"seed43":values[1],"seed44":values[2]})
    save(out/"exp014_accuracy_summary.csv",summary)
    # The same source image and model seed are paired across five scaling interventions.
    per_seed={}
    for p in exps:
        records=read(p/"scale_image_family_probabilities.csv")
        idx={}
        for r in records:
            key=(int(r["seed"]),r["model"],int(r["writer"]),int(r["index"]))
            idx.setdefault(key,{})[float(r["scale"])]=r
        for key,v in idx.items():
            assert set(v)=={.8,.9,1.,1.1,1.2},key
            seed,model,writer,i=key
            truth=v[1.]["true"]
            family="digits" if truth in "0123456789" else "letters"
            for scale in (.8,.9,1.1,1.2):
                p0=float(v[1.]["p_letter"]);ps=float(v[scale]["p_letter"])
                m0=p0>=.5;ms=ps>=.5
                block=(seed,model,family,scale)
                s=per_seed.setdefault(block,{"n":0,"delta_sum":0.,"absolute_sum":0.,"flips":0})
                s["n"]+=1;s["delta_sum"]+=ps-p0
                s["absolute_sum"]+=abs(ps-p0);s["flips"]+=int(ms!=m0)
    sensitivity=[]
    for (seed,model,family,scale),v in sorted(per_seed.items()):
        sensitivity.append({"seed":seed,"model":model,"family":family,"scale":scale,
           "n_images":v["n"],"mean_signed_delta_p_letter":v["delta_sum"]/v["n"],
           "mean_abs_delta_p_letter":v["absolute_sum"]/v["n"],
           "family_prediction_flip_rate":v["flips"]/v["n"]})
    save(out/"exp014_family_scale_sensitivity_by_seed.csv",sensitivity)
    grouped={}
    for r in sensitivity:
        key=(r["model"],r["family"],r["scale"])
        grouped.setdefault(key,[]).append(r)
    agg=[]
    for (model,family,scale),v in sorted(grouped.items()):
        assert len(v)==3
        agg.append({"model":model,"family":family,"scale":scale,
          "n_seeds":3,"n_unique_images_per_seed":v[0]["n_images"],
          "mean_signed_delta_p_letter":avg([q["mean_signed_delta_p_letter"] for q in v]),
          "mean_abs_delta_p_letter":avg([q["mean_abs_delta_p_letter"] for q in v]),
          "mean_family_flip_rate":avg([q["family_prediction_flip_rate"] for q in v])})
    save(out/"exp014_family_scale_sensitivity_summary.csv",agg)
    md=["# EXP013·014 — Actual measured results (3 seeds)","",
      "Source is GitHub Actions for this experiment branch. No original/professor Test.",
      "EXP013 binary-probe models were trained on Official Train ONLY.",
      "EXP014 MLP variants were trained on Official Train 76,800 samples each.",
      "Hand-made is a repeatedly used development set, not an independent final test.",
      "",
      "## EXP013 geometry-only binary classification (Official Validation)",
      "",
      "| Features after | Feature set | Probe | Balanced Acc | ROC AUC |",
      "|---|---|---|---:|---:|"]
    for r in binary_final:
        if r["dataset"]=="OfficialVal" and r["features"] in ("all_four","size_only","ink_mass_only") and r["classifier"]=="logreg":
            md.append(f"| {r['preprocessing']} | {r['features']} | {r['classifier']} | "
                      f"{100*float(r['balanced_accuracy']):.3f}% | {float(r['roc_auc']):.4f} |")
    md+=["","## EXP014 Official Validation and Handmade (three-seed means)","",
         "| Model | Official Val | Handmade digits | Handmade letters |",
         "|---|---:|---:|---:|"]
    names=sorted({r["model"] for r in summary})
    lookup={(r["model"],r["dataset"],r["group"]):r["mean_accuracy"] for r in summary}
    for name in names:
        md.append(f"| {name} | {100*lookup[name,'OfficialVal','all']:.3f}% | "
                  f"{100*lookup[name,'Handmade','digits']:.3f}% | "
                  f"{100*lookup[name,'Handmade','letters']:.3f}% |")
    md+=["","## EXP014 Size response of full preprocessing pipelines",
         "Mean absolute change in P(letter) vs 1.0x; smaller indicates more robust behavior.",
         "Also see per-seed family flip rate; robustness alone is not accuracy.",
         "",
         "| Model | Class family | Scale | Mean |Delta P(letter)| | Flip rate |",
         "|---|---|---:|---:|---:|"]
    for r in agg:
        if r["scale"] in (.8,1.2):
            md.append(f"| {r['model']} | {r['family']} | {r['scale']:.1f} | "
                      f"{r['mean_abs_delta_p_letter']:.4f} | "
                      f"{100*r['mean_family_flip_rate']:.2f}% |")
    md+=["","## Interpretation boundaries",
      "- EXP013: geometry-only binary accuracy is evidence a shortcut feature exists in the dataset; it is not by itself proof that the MLP uses it.",
      "- EXP014: compare augmentation against frozen original EXP006 CoM and SizeCoM controls at the same preprocessing and original developer data.",
      "- CoM augmentation models and SizeCoM augmentation model should be compared to the correspondingly preprocessed archived controls.",
      "- Scale perturbations alter thickness, mass, and pixels; improvements do not prove a unique isolated feature caused the change.",
      "- Original Test/professor Test were never used. Do not adopt new pipeline without approval.",
      ""]
    (out/"SUMMARY.md").write_text("\n".join(md),encoding="utf-8")
    (out/"manifest.json").write_text(json.dumps({"source_seeds":[42,43,44],
      "source_exp013":str(exp013.relative_to(source)),"independent_test_used":False,
      "main_pipeline_changed":False},indent=2),encoding="utf-8")
    print("EXP013_EXP014_REVIEW_COMPLETE",len(binary_final),len(summary),len(agg),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--inputs",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.inputs,a.out)
