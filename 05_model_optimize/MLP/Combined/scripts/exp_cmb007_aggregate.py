#!/usr/bin/env python3
"""Aggregate all 15 EXP-CMB-007 Size 60-100 x Stroke 0/14/28 runs (no training).

Each completed run has four held-out writer model predictions, 16 class error
rates, 240 directed misclassification counts, and five-scale family responses.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
ARMS={f"scale{pct:02d}_stroke{strength:02d}":(pct/100,strength/100)
      for pct in (60,70,80,90,100) for strength in (0,14,28)}
SEEDS=(42,43,44)
WRITERS=(41,42,43,44)
LABELS="0123456789ABCDEF"
def loadcsv(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def writecsv(p,rows):
    rows=list(rows);assert rows,p
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--incoming",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--stage",choices=("extended",),required=True)
    args=parser.parse_args()
    expected=list(ARMS)
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    original=out/"raw"
    allsummary=[];fold=[];pairs=[];class_error_rows=[];family_error_rows=[];responses=[];classes=[];geometry=[];perimage=[]
    for arm in expected:
        for seed in SEEDS:
            folder=args.incoming/f"cmb007-{arm}-s{seed}"
            manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
            assert manifest["arm"]==arm and int(manifest["seed"])==seed
            assert manifest["experiment"]=="EXP-CMB-007" and manifest["fixed_model_hyperparameters"]
            assert not manifest["original_test_used"] and not manifest["professor_test_used"]
            assert not manifest["new_handmade_480_used"]
            assert manifest["no_mlp_hyperparameter_search"] if "no_mlp_hyperparameter_search" in manifest else True
            target=original/arm/f"seed_{seed}"
            target.mkdir(parents=True,exist_ok=True)
            for item in folder.iterdir():
                if item.is_file():
                    shutil.copyfile(item,target/item.name)
            pred=loadcsv(folder/"oof_predictions.csv")
            assert len(pred)==2560
            pp={int(x["writer"]):[] for x in pred}
            for x in pred:
                pp[int(x["writer"])].append(x)
            assert all(len(v)==640 for v in pp.values())
            yy=np.array([LABELS.index(x["actual"]) for x in pred])
            predicted=np.array([LABELS.index(x["pred"]) for x in pred])
            pr,re,f1,_=precision_recall_fscore_support(yy,predicted,
                labels=np.arange(16),zero_division=0)
            summary={"arm":arm,"seed":seed,
                "ratio":ARMS[arm][0],"stroke":ARMS[arm][1],
                "n":len(yy),"correct":int(np.sum(yy==predicted)),
                "accuracy":float(np.mean(yy==predicted)),
                "macro_f1":float(f1.mean()),
                "digits_acc":float(np.mean(yy[yy<10]==predicted[yy<10])),
                "letters_acc":float(np.mean(yy[yy>=10]==predicted[yy>=10]))}
            allsummary.append(summary)
            for f in loadcsv(folder/"fold_metrics.csv"):fold.append(f)
            for f in loadcsv(folder/"pair_confusions.csv"):pairs.append(f)
            for f in loadcsv(folder/"per_class_error_rates.csv"):class_error_rows.append(f)
            for f in loadcsv(folder/"family_confusion_rates.csv"):family_error_rows.append(f)
            for f in loadcsv(folder/"scale_sensitivity.csv"):responses.append(f)
            for f in loadcsv(folder/"per_class_and_scale.csv"):classes.append(f)
            for f in loadcsv(folder/"geometry_distributions.csv"):geometry.append(f)
            for x in pred:perimage.append(x)
            assert len(loadcsv(folder/"training_time.csv"))==4
            assert len(loadcsv(folder/"scale_sensitivity.csv"))==4*2*5*3
    assert len(allsummary)==len(expected)*3
    assert len(expected)==15
    writecsv(out/"per_seed_overall.csv",allsummary)
    writecsv(out/"per_fold_group_metrics.csv",fold)
    writecsv(out/"pairwise_confusion_directions.csv",pairs)
    writecsv(out/"all_class_error_rates_raw.csv",class_error_rows)
    writecsv(out/"family_error_rates_raw.csv",family_error_rows)
    writecsv(out/"five_scale_group_responses.csv",responses)
    writecsv(out/"five_scale_16class_responses.csv",classes)
    writecsv(out/"trained_geometry.csv",geometry)
    writecsv(out/"oof_raw_predictions.csv",perimage)
    stats=[]
    for arm in expected:
        v=[x for x in allsummary if x["arm"]==arm]
        total=np.mean([x["accuracy"] for x in v])
        val=[x for x in fold if x["arm"]==arm and x["dataset"]=="OfficialVal"
            and x["group"]=="all"]
        normal=[x for x in responses if x["arm"]==arm and x["dataset"]=="OOF"
            and x["group"] in ("digits","letters")]
        risk=[]
        for group,scale in (("digits",1.2),("letters",.8)):
            current=[float(x["family_flip"]) for x in normal
                     if x["group"]==group and abs(float(x["scale"])-scale)<1e-6]
            assert len(current)==len(SEEDS)*len(WRITERS)
            risk.append(float(np.mean(current)))
        fcs=[x for x in classes if x["arm"]==arm and x["dataset"]=="OOF"
             and x["actual"] in ("B","8","D") and float(x["scale"]) in (.8,1.2)]
        worst=max(float(x["family_flip"]) for x in fcs) # single writer/seed, disclose
        fco=[x for x in classes if x["arm"]==arm and x["dataset"]=="OfficialVal"
             and x["actual"] in ("B","8","D") and float(x["scale"]) in (.8,1.2)]
        ov_worst=max(float(x["family_flip"]) for x in fco)
        stats.append({"arm":arm,"size_ratio":ARMS[arm][0],
            "stroke_strength":ARMS[arm][1],
            "mean_accuracy":total,
            "std_accuracy_seed":float(np.std([x["accuracy"] for x in v])),
            "mean_digits_acc":float(np.mean([x["digits_acc"] for x in v])),
            "mean_letters_acc":float(np.mean([x["letters_acc"] for x in v])),
            "official_val_accuracy":float(np.mean([float(x["accuracy"]) for x in val])),
            "digits_1p2_family_flip":risk[0],
            "letters_0p8_family_flip":risk[1],
            "max_writer_seed_B8D_family_flip":worst,
            "max_official_fold_seed_B8D_family_flip":ov_worst})
    writecsv(out/"comparison.csv",stats)
    # All 240 directional errors: aggregate across 4 held-out writers,
    # then average across 3 seeds. Percentages always use the source class
    # denominator (160 Hand-made per class per seed).
    baselinepath=Path(__file__).resolve().parents[4]/"05_model_optimize"/"MLP"/"Combined"/"results"/"EXP_CMB_001"/"OOF_predictions.csv"
    baseline=loadcsv(baselinepath)
    assert len(baseline)==2560
    baseline_cm=np.zeros((16,16),dtype=int)
    for r in baseline:
        aa=LABELS.index(r["true_label"]);bb=LABELS.index(r["combined_pred"])
        baseline_cm[aa,bb]+=1
    assert (np.trace(baseline_cm),baseline_cm.sum())==(2080,2560)
    direction_comparisons=[]
    class_comparisons=[]
    for arm in expected:
        baseline_wrong={LABELS[i]:int(baseline_cm[i].sum()-baseline_cm[i,i]) for i in range(16)}
        arr=[r for r in pairs if r["arm"]==arm and r["dataset"]=="OOF_original"]
        rr=[r for r in class_error_rows if r["arm"]==arm and r["dataset"]=="OOF_original"]
        assert len(arr)==3*4*16*15,(arm,len(arr))
        assert len(rr)==3*4*16
        for i,letter in enumerate(LABELS):
            cr=[r for r in rr if r["actual"]==letter]
            assert len(cr)==12
            current=float(np.mean([float(r["class_error_rate"]) for r in cr]))
            base=float(baseline_wrong[letter]/160)
            class_comparisons.append({"arm":arm,"actual":letter,"n_actual_per_seed":160,
               "baseline_wrong":baseline_wrong[letter],
               "baseline_error_rate":base,"mean_wrong":float(np.mean(
                    [sum(int(z["wrong"]) for z in cr if int(z["seed"])==seed) for seed in SEEDS])),
               "mean_class_error_rate":current,"delta_error_rate_pp":100*(current-base)})
            for j,target in enumerate(LABELS):
                if i==j:continue
                group=[r for r in arr if r["actual"]==letter and r["predicted"]==target]
                assert len(group)==12,(arm,letter,target,len(group))
                seedtot=[sum(int(r["errors"]) for r in group if int(r["seed"])==seed) for seed in SEEDS]
                baseline_count=int(baseline_cm[i,j])
                avg=float(np.mean(seedtot))
                direction_comparisons.append({"arm":arm,"actual":letter,"predicted":target,
                    "n_actual_per_seed":160,"baseline_errors":baseline_count,
                    "baseline_rate":baseline_count/160,
                    "mean_errors":avg,"mean_error_rate":avg/160,
                    "delta_errors_vs_baseline":avg-baseline_count,
                    "delta_error_rate_pp":100*(avg-baseline_count)/160})
    writecsv(out/"all_240_direction_changes.csv",direction_comparisons)
    writecsv(out/"all_16_class_error_changes.csv",class_comparisons)
    # Every direction, even absent, remains in the CSV; top20 for quick review.
    top20=sorted(direction_comparisons,key=lambda r:(-r["mean_errors"],r["arm"],r["actual"],r["predicted"]))[:20]
    writecsv(out/"top_20_error_directions_across_conditions.csv",top20)
    family_summary=[]
    for arm in expected:
        for fam in ("digit_to_letter","letter_to_digit","digit_to_digit","letter_to_letter"):
            x=[r for r in family_error_rows if r["arm"]==arm and r["dataset"]=="OOF_original"
               and r["direction"]==fam]
            assert len(x)==12
            family_summary.append({"arm":arm,"direction":fam,
              "mean_error_count":float(np.mean([sum(int(z["errors"]) for z in x
                                            if int(z["seed"])==seed) for seed in SEEDS])),
              "mean_error_rate_by_actual_family":float(np.mean([float(z["error_rate"]) for z in x]))})
    writecsv(out/"family_error_direction_comparison.csv",family_summary)

    # Paired effects: retain same seed/writer/image index for all 40/50 conditions.
    corrections=[]
    for i in range(len(expected)):
        for j in range(i+1,len(expected)):
            a=expected[i];b=expected[j]
            for seed in SEEDS:
                def get(am):
                    return {(int(x["writer"]),int(x["index"])):x
                         for x in perimage if x["arm"]==am and int(x["seed"])==seed}
                pa,pb=get(a),get(b)
                assert len(pa)==len(pb)==2560 and pa.keys()==pb.keys()
                old=np.array([pa[k]["pred"]==pa[k]["actual"] for k in sorted(pa)])
                new=np.array([pb[k]["pred"]==pb[k]["actual"] for k in sorted(pb)])
                corrections.append({"a":a,"b":b,"seed":seed,
                    "corrected":int(np.sum(~old&new)),
                    "induced":int(np.sum(old&~new)),
                    "net":int(new.sum()-old.sum())})
    writecsv(out/"paired_corrected_induced.csv",corrections)
    lines=[f"# EXP-CMB-007 — {args.stage} 실측 집계","",
      "**전처리 효과 분석용. MLP HP는 모든 조건에 고정. 특정 Accuracy/F1 숫자를 넘기면 합격시키는 연구가 아니다.**",
      "", "| 조건 | OOF Accuracy | 숫자 | 문자 | 공식 Val | 숫자×1.2 Family flip | 문자×0.8 Family flip |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for a in stats:
        lines.append("| {} | {:.3%} | {:.3%} | {:.3%} | {:.3%} | {:.2%} | {:.2%} |".format(
          a["arm"],a["mean_accuracy"],a["mean_digits_acc"],a["mean_letters_acc"],
          a["official_val_accuracy"],a["digits_1p2_family_flip"],
          a["letters_0p8_family_flip"]))
    lines+=["","## 16개 전체 클래스의 오분류율 (OOF, 각 실제 클래스 160장/Seed)",
      "","| 조건 | 실제 클래스 | Baseline 오분류율 | 새 모델 오분류율 | 변화 (pp) |",
      "|---|---|---:|---:|---:|"]
    for r in class_comparisons:
        lines.append("| {} | {} | {:.2%} | {:.2%} | {:+.2f} |".format(
            r["arm"],r["actual"],r["baseline_error_rate"],
            r["mean_class_error_rate"],r["delta_error_rate_pp"]))
    lines+=["","## 그 밖의 주요 오류 방향 (기존 Baseline에서 0↔D/8↔B 제외 상위 15개)",
      "","| 실제→예측 | 기존 건수/160 | 기존 방향 오류율 |"]
    lines+=["|---|---:|---:|"]
    other=[]
    for i,a in enumerate(LABELS):
        for j,b in enumerate(LABELS):
            if i==j or (a,b) in (("0","D"),("D","0"),("8","B"),("B","8")):
                continue
            other.append((int(baseline_cm[i,j]),a,b))
    for count,a,b in sorted(other,reverse=True)[:15]:
        lines.append("| {}→{} | {} | {:.2%} |".format(a,b,count,count/160))
    lines+=["","**모든 방향(16×15=240개)과 전처리 전후 변화는 [all_240_direction_changes.csv](all_240_direction_changes.csv).**",
      "**전체 클래스별 오류율과 0→D/8→B 외 새롭게 늘어난 혼동은 [all_16_class_error_changes.csv](all_16_class_error_changes.csv), [family_error_direction_comparison.csv](family_error_direction_comparison.csv)에도 기록.**",
      ""]
    lines+=["","**주의:** 표의 숫자/문자 Family flip은 이미지는 같은데 문자계열 P(letter)≥0.5 여부가 바뀐 비율. Accuracy와 다르다. 개별 클래스/작성자에서 반대 방향의 오류와 공식 Val 하락이 존재할 수 있다.",
      "",
      "관련 원본: per_seed_overall.csv, per_fold_group_metrics.csv, five_scale_16class_responses.csv, trained_geometry.csv, pairwise_confusion_directions.csv, paired_corrected_induced.csv.",
      "",
      "사전등록된 비교 질문에 대한 기전 해석/반례/연구 목적 달성 판정은 별도 최종 연구일지에서 한다. 원본 Test와 교수님 Test는 접근하지 않았다.", ""]
    (out/"SUMMARY.md").write_text("\n".join(lines),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(10,5.2))
    xs=np.arange(len(stats))
    ax.plot(xs,[100*a["mean_accuracy"] for a in stats],marker="o",label="Held-out writer OOF")
    ax.plot(xs,[100*a["official_val_accuracy"] for a in stats],marker="s",label="Official Development Val")
    ax.set(xticks=xs,xticklabels=[a["arm"] for a in stats],
       ylabel="Accuracy (%)",title=f"EXP-CMB-007 {args.stage} | Frozen Combined MLP HP")
    plt.setp(ax.get_xticklabels(),rotation=30,ha="right")
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/"comparison.png",dpi=170);plt.close(fig)
    (out/"manifest.json").write_text(json.dumps({"experiment":"EXP-CMB-007",
      "stage":args.stage,"arms":expected,"seeds":SEEDS,"folds":WRITERS,
      "expected_fits":len(expected)*len(SEEDS)*len(WRITERS),
      "no_hp_search":True,"no_test_used":True,
      "research_gate":"explanation not absolute accuracy",
      "status":"numeric comparison complete, explanatory decision pending",
      "reference_experiment":"EXP-CMB-005/006",
      "frozen_mlp_parameters":True,
      "original_0_to_50_not_retrained":True},
      ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print((out/"SUMMARY.md").read_text(encoding="utf-8"),flush=True)
if __name__=="__main__":
    main()
