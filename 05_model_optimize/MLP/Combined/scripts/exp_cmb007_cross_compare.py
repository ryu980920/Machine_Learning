#!/usr/bin/env python3
"""EXP-CMB-007: compare frozen EXP-CMB-005/006 frozen 0-50% against EXP-CMB-007 new 60-100%.

All original OOF predictions come from previously merged main results; no
retraining, renormalization, or test-set inspection takes place here.
"""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS="0123456789ABCDEF"
ORIG_ARMS=["com_only"]+[f"scale{pct:02d}_stroke{stroke:02d}" for pct in (10,20,30,40,50) for stroke in (0,14,28)]
NEW_ARMS=[f"scale{pct:02d}_stroke{stroke:02d}" for pct in (60,70,80,90,100) for stroke in (0,14,28)]
REFERENCES=("scale30_stroke00","scale30_stroke28","scale50_stroke00","scale50_stroke28")
SEEDS=(42,43,44)
WRITERS=(41,42,43,44)

def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as file:
        return list(csv.DictReader(file))

def csvout(path,data):
    assert data
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--old",type=Path,required=True)
    parser.add_argument("--old-low",type=Path,required=True)
    parser.add_argument("--old-high",type=Path,required=True)
    parser.add_argument("--new",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    old,new,out=args.old,args.new,args.out
    out.mkdir(parents=True,exist_ok=True)

    a=rows(old/"all_16_preprocessing_conditions.csv");b=rows(new/"comparison.csv")
    assert [x["arm"] for x in a]==ORIG_ARMS, [x["arm"] for x in a]
    assert [x["arm"] for x in b]==NEW_ARMS, [x["arm"] for x in b]
    combined=a+b
    assert len(combined)==31 and len({x["arm"] for x in combined})==31
    csvout(out/"all_31_preprocessing_conditions.csv",combined)

    classes=rows(old/"all_16_class_error_rates_across_16_conditions.csv")+rows(new/"all_16_class_error_changes.csv")
    dirs=rows(old/"all_240_directed_errors_across_16_conditions.csv")+rows(new/"all_240_direction_changes.csv")
    assert len(classes)==31*16 and len(dirs)==31*240
    csvout(out/"all_16_class_error_rates_across_31_conditions.csv",classes)
    csvout(out/"all_240_directed_errors_across_31_conditions.csv",dirs)

    a_pred=rows(args.old_low/"oof_raw_predictions.csv")+rows(args.old_high/"oof_raw_predictions.csv")
    b_pred=rows(new/"oof_raw_predictions.csv")
    assert len(a_pred)==16*3*2560 and len(b_pred)==15*3*2560
    def organize(data):
        result=defaultdict(dict)
        for r in data:
            arm=r["arm"];seed=int(r["seed"]);writer=int(r["writer"]);idx=int(r["index"])
            k=(writer,idx);g=(arm,seed)
            assert writer in WRITERS and seed in SEEDS
            assert k not in result[g],(g,k)
            result[g][k]=(r["actual"],r["pred"])
        for key,v in result.items():
            assert len(v)==2560,(key,len(v))
        return result
    original=organize(a_pred)
    extended=organize(b_pred)
    paired=[];paired_class=[];paired_writer=[]
    for reference in REFERENCES:
        for arm in NEW_ARMS:
            for seed in SEEDS:
                prior=original[(reference,seed)];current=extended[(arm,seed)]
                assert prior.keys()==current.keys() and len(prior)==2560
                old_ok=[];new_ok=[]
                for key in sorted(prior):
                    true,old_pred=prior[key];true2,new_pred=current[key]
                    assert true==true2
                    old_ok.append(old_pred==true)
                    new_ok.append(new_pred==true)
                old_ok=np.array(old_ok,dtype=bool);new_ok=np.array(new_ok,dtype=bool)
                item={"reference":reference,"new_arm":arm,"seed":seed,
                      "n":2560,"old_correct":int(old_ok.sum()),
                      "new_correct":int(new_ok.sum()),
                      "corrected":int(np.sum(~old_ok&new_ok)),
                      "induced":int(np.sum(old_ok&~new_ok)),
                      "delta_correct":int(new_ok.sum()-old_ok.sum()),
                      "delta_accuracy_pp":float(100*(new_ok.mean()-old_ok.mean()))}
                assert item["corrected"]-item["induced"]==item["delta_correct"]
                paired.append(item)
                for actual in LABELS:
                    keys=[k for k in prior if prior[k][0]==actual]
                    assert len(keys)==160,(actual,len(keys))
                    corrected=sum(prior[k][1]!=actual and current[k][1]==actual for k in keys)
                    induced=sum(prior[k][1]==actual and current[k][1]!=actual for k in keys)
                    prior_err=sum(prior[k][1]!=actual for k in keys)
                    new_err=sum(current[k][1]!=actual for k in keys)
                    paired_class.append({"reference":reference,"new_arm":arm,"seed":seed,
                       "actual":actual,"n":160,"old_errors":prior_err,"new_errors":new_err,
                       "corrected":corrected,"induced":induced,
                       "change_error_rate_pp":float(100*(new_err-prior_err)/160)})
                for writer in WRITERS:
                    keys=[k for k in prior if k[0]==writer]
                    assert len(keys)==640
                    old_w=sum(prior[k][1]==prior[k][0] for k in keys)
                    new_w=sum(current[k][1]==current[k][0] for k in keys)
                    paired_writer.append({"reference":reference,"new_arm":arm,"seed":seed,
                       "writer":writer,"n":640,"old_correct":old_w,"new_correct":new_w,
                       "delta_accuracy_pp":float(100*(new_w-old_w)/640)})
    assert len(paired)==len(REFERENCES)*len(NEW_ARMS)*3
    csvout(out/"paired_30_50_vs_60_100_corrections.csv",paired)
    csvout(out/"paired_30_50_vs_60_100_per_class.csv",paired_class)
    csvout(out/"paired_30_50_vs_60_100_per_writer.csv",paired_writer)

    # Compare *mean* directed counts per 2560 held-out OOF across 3 seeds,
    # keeping nonzero and zero directions as separate, explicit rows.
    directed_delta=[]
    dmap={(x["arm"],x["actual"],x["predicted"]):float(x["mean_errors"]) for x in dirs}
    for reference in REFERENCES:
        for arm in NEW_ARMS:
            for aclass in LABELS:
                for predicted in LABELS:
                    if aclass==predicted:continue
                    older=dmap[(reference,aclass,predicted)]
                    newer=dmap[(arm,aclass,predicted)]
                    directed_delta.append({"reference":reference,"new_arm":arm,
                      "actual":aclass,"predicted":predicted,
                      "old_mean_errors":older,"new_mean_errors":newer,
                      "delta_errors":newer-older,
                      "delta_actual_class_rate_pp":float(100*(newer-older)/160)})
    csvout(out/"all_240_pair_changes_vs_30_50.csv",directed_delta)
    assert len(directed_delta)==len(REFERENCES)*len(NEW_ARMS)*240


    # Family-flip stability covers all 16 classes, both data sources, 0.8-1.2.
    # 1.0 is the frozen model's per-image reference. Keep source class and
    # writer-by-seed worst-case separate from class-mean flip.
    raw_class=(rows(args.old_low/"five_scale_16class_responses.csv")
             +rows(args.old_high/"five_scale_16class_responses.csv")
             +rows(new/"five_scale_16class_responses.csv"))
    expected_raw=31*3*4*2*5*16
    assert len(raw_class)==expected_raw,(len(raw_class),expected_raw)
    class_group=defaultdict(list)
    for r in raw_class:
        arm,source,actual,scale=r["arm"],r["dataset"],r["actual"],float(r["scale"])
        assert arm in (ORIG_ARMS+NEW_ARMS) and source in ("OOF","OfficialVal")
        assert actual in LABELS
        class_group[(arm,source,actual,scale)].append(float(r["family_flip"]))
    class_flip=[]
    for (arm,source,actual,scale),v in class_group.items():
        assert len(v)==3*4,(arm,source,actual,scale,len(v))
        class_flip.append({"arm":arm,"dataset":source,"actual":actual,"scale":scale,
                           "mean_family_flip":float(np.mean(v)),
                           "max_fold_seed_family_flip":float(max(v)),
                           "min_fold_seed_family_flip":float(min(v))})
    assert len(class_flip)==31*2*16*5
    class_flip.sort(key=lambda x:((ORIG_ARMS+NEW_ARMS).index(x["arm"]),
                              x["dataset"],x["actual"],x["scale"]))
    csvout(out/"all_31_condition_16class_five_scale_flip.csv",class_flip)
    worst=[]
    for arm in (ORIG_ARMS+NEW_ARMS):
        for source in ("OOF","OfficialVal"):
            d=[x for x in class_flip if x["arm"]==arm and x["dataset"]==source
               and x["scale"] in (.8,1.2)]
            focus=[x for x in d if x["actual"] in "B8D"]
            assert len(d)==32 and len(focus)==6
            worst.append({"arm":arm,"dataset":source,
                          "max_16class_mean_flip":float(max(x["mean_family_flip"] for x in d)),
                          "max_B8D_mean_flip":float(max(x["mean_family_flip"] for x in focus)),
                          "max_16class_fold_seed_flip":float(max(x["max_fold_seed_family_flip"] for x in d)),
                          "max_B8D_fold_seed_flip":float(max(x["max_fold_seed_family_flip"] for x in focus))})
    csvout(out/"all_31_condition_16class_worst_flip.csv",worst)

    # Include Normal-style family flip and official development Validation on
    # the same chart as OOF. These are *observed*, not selection thresholds.
    fig,ax=plt.subplots(figsize=(14,6))
    xs=np.arange(len(combined))
    ax.plot(xs,[100*float(r["mean_accuracy"]) for r in combined],
            marker="o",label="Held-out writer OOF")
    ax.plot(xs,[100*float(r["official_val_accuracy"]) for r in combined],
            marker="s",label="Official development Validation")
    ax.set(xticks=xs,xticklabels=[r["arm"] for r in combined],
           ylabel="Accuracy (%)",title="EXP-CMB-005/006+006 | frozen MLP | Size 0–100%")
    plt.setp(ax.get_xticklabels(),rotation=42,ha="right")
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/"size0to100_oof_vs_official.png",dpi=160)
    plt.close(fig)

    notes=["# Combined Size 0–100% × Stroke 0/14/28 — 31조건 실측 비교",
      "",
      "**모든 모델 MLP HP 동일. EXP-CMB-005/006의 16조건 원본 실측 재사용, EXP-CMB-007의 15조건만 새로 학습.**",
      "연구 판정은 숫자 Accuracy/F1 하한이 아니라 원인·상충·반례 설명이 가능한지로 결정.",
      "",
      "| 조건 | OOF | 숫자 | 문자 | OfficialVal | 숫자×1.2 계열 Flip | 문자×0.8 계열 Flip |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for r in combined:
        notes.append("| {} | {:.3%} | {:.3%} | {:.3%} | {:.3%} | {:.2%} | {:.2%} |".format(
            r["arm"],float(r["mean_accuracy"]),float(r["mean_digits_acc"]),
            float(r["mean_letters_acc"]),float(r["official_val_accuracy"]),
            float(r["digits_1p2_family_flip"]),float(r["letters_0p8_family_flip"])))
    notes+=["","## 30%/50% 기준 대비 새 오답·교정 진단 (동일 Seed × Writer × Image 대응)"]
    for reference in REFERENCES:
        notes+=["",f"### 기준 {reference}","",
                "| 새 조건 | 3Seed 평균 교정 | 3Seed 평균 신규 오답 | 평균 OOF Δ(pp) |",
                "|---|---:|---:|---:|"]
        for arm in NEW_ARMS:
            pp=[r for r in paired if r["reference"]==reference and r["new_arm"]==arm]
            notes.append("| {} | {:.1f} | {:.1f} | {:+.3f} |".format(
                arm,np.mean([r["corrected"] for r in pp]),
                np.mean([r["induced"] for r in pp]),
                np.mean([r["delta_accuracy_pp"] for r in pp])))
    notes+=["","**모든 16클래스 및 240개 오분류 방향별 악화·개선과 0.8~1.2배 개별 클래스 최악 Flip은 CSV로 확인.**",
        "새 실험이 기존 30/50%보다 OOF 점수가 높더라도 OfficialVal·숫자/문자·필체별 손실과 반례가 있으면 별도 서술.",
        "",
        "원본 Test/교수님 Test/Normal 신규 480장 미사용. 반복 사용한 Hand-made 4명은 독립 신규 작성자 Test가 아님."]
    (out/"SUMMARY.md").write_text("\n".join(notes)+"\n",encoding="utf-8")
    manifest={"experiment":"EXP-CMB-007","reference":"EXP-CMB-005/006",
      "old_condition_count":len(ORIG_ARMS),"new_condition_count":len(NEW_ARMS),
      "total_condition_count":len(combined),
      "reference_arms":REFERENCES,"seeds":SEEDS,"writers":WRITERS,
      "expected_new_fits":180,"hyperparameters_changed":False,
      "original_test_used":False,"professor_test_used":False,
      "status":"numeric comparison complete; explanatory audit pending"}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print((out/"SUMMARY.md").read_text(encoding="utf-8"),flush=True)

if __name__=="__main__":
    main()
