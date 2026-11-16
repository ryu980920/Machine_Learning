#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate EXP016 60 high-ratio new MLPs vs frozen EXP015 30/40/50 reference MLPs.

Never retrain or read any Original Test/Professor Test data.
"""
from __future__ import annotations
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from exp006_size_com_comparison import LABELS

SEEDS=(42,43,44)
HIGH=(60,70,80,90,100)
STROKES=(0,14,28,42)
OLD=(30,40,50)
FOCUS=("8","B","D")
SC=(.8,.9,1.,1.1,1.2)
EXTREMES=(.8,1.2)
GATES={"digit12flip":.10,"letter08flip":.10,"digit12p":.12,"letter08p":.12,
       "seedgroup":.15,"allscale":.12,"digitsacc":.85,"lettersacc":.80,
       "focusmax":.20,"allclassmax":.25,"seedfocusmax":.30}
EPS=1e-10
def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
def save(path,rows):
    assert rows
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def avg(x):return sum(x)/len(x)
def run(hi,old,out):
    hi=Path(hi);old=Path(old);out=Path(out)
    assert not out.exists()
    out.mkdir(parents=True)
    models_new={f"size{r}_stroke{v:02d}" for r in HIGH for v in STROKES}
    models_old={f"size{r}_stroke28" for r in OLD}
    names=sorted(models_new|models_old)
    assert len(names)==23
    record={};acc={};sens={};pairs=[];src={}
    for s in SEEDS:
        for high,ratios,root in ((True,HIGH,hi),(False,OLD,old)):
            for ratio in ratios:
                f=root/(f"exp016-s{s}-r{ratio}" if high else f"exp015-s{s}-r{ratio}")
                manifest=json.loads((f/"manifest.json").read_text(encoding="utf-8"))
                assert manifest["seed"]==s and manifest["actual_training_rows_per_variant"]==76800
                assert manifest["experiment"]==("EXP016" if high else "EXP015")
                assert manifest["original_predictions_verified"]
                assert manifest["official_train_n"]==76800 and manifest["official_val_n"]==19200
                assert manifest["handmade_n"]==2560
                assert manifest["nested_sampling_from_exp014_half"]
                assert not any(manifest[k] for k in ("official_test_used","professor_test_used",
                              "handmade_used_in_fit","official_val_used_in_fit"))
                if high:
                    assert manifest["augmentation_above_50_with_stable_stroke_parity"]
                    assert abs(manifest["ratio_actual"]-ratio/100)<.002
                desired={f"size{ratio}_stroke{v:02d}" for v in STROKES} if high else {f"size{ratio}_stroke28"}
                for row in read(f/"training_records.csv"):
                    if row["variant"] in desired:
                        key=(row["variant"],s)
                        assert key not in record,key
                        record[key]=row;src[key]=f
                for row in read(f/"accuracy_metrics.csv"):
                    if row["variant"] not in desired or row["group"] not in ("all","digits","letters"):continue
                    key=(row["variant"],s,row["dataset"],row["group"])
                    assert key not in acc,key
                    acc[key]=float(row["accuracy"])
                for row in read(f/"scale_sensitivity.csv"):
                    if row["variant"] not in desired:continue
                    key=(row["variant"],s,row["group"],float(row["scale"]))
                    assert key not in sens,key
                    sens[key]=row
                pairs += [x for x in read(f/"pair_errors.csv") if x["variant"] in desired]
    assert len(record)==69,len(record)
    assert all((name,s) in src for name in names for s in SEEDS)
    save(out/"all_candidates_seed_pair_errors.csv",pairs)
    cls_data={};perclass=[];seed_focus={}
    for name in names:
        scratch=defaultdict(list)
        for s in SEEDS:
            path=src[name,s]/f"response_{name}.npz"
            with np.load(path,allow_pickle=False) as z:
                p=np.array(z["p_letter"])
                truth=np.array(z["y_true"],dtype=int)
                scales=np.array(z["scales"],dtype=float)
            assert p.shape==(5,2560) and len(truth)==2560
            assert np.allclose(scales,SC,atol=1e-6)
            assert set(truth)==set(range(16))
            baseline=p[2]>=.5
            for c,label in enumerate(LABELS):
                mask=truth==c
                assert np.sum(mask)==160,(name,s,label)
                for k,scale in enumerate(SC):
                    if scale==1:continue
                    flip=float(np.mean((p[k,mask]>=.5)!=baseline[mask]))
                    drift=float(np.mean(np.abs(p[k,mask]-p[2,mask])))
                    scratch[label,scale].append((flip,drift))
                    if label in FOCUS and scale in EXTREMES:
                        key=(name,s)
                        seed_focus[key]=max(seed_focus.get(key,0.),flip)
        for (label,scale),values in sorted(scratch.items()):
            assert len(values)==3
            row={"model":name,"class":label,"group":"digits" if LABELS.index(label)<10 else "letters",
                "scale":scale,"n_per_seed":160,"seeds":3,
                "flip_mean":avg([a for a,b in values]),
                "flip_max_seed":max(a for a,b in values),
                "abs_delta_p_letter_mean":avg([b for a,b in values])}
            perclass.append(row);cls_data[name,label,scale]=row
    save(out/"all_16class_size_flip_3seed.csv",perclass)
    ranked=[]
    for name in names:
        def accuracy(ds,group):return [acc[name,s,ds,group] for s in SEEDS]
        def metric(group,scale,field):
            return [float(sens[name,s,group,scale][field]) for s in SEEDS]
        v=accuracy("OfficialVal","all")
        d=accuracy("Handmade","digits");l=accuracy("Handmade","letters")
        d12=metric("digits",1.2,"family_flip_rate")
        l08=metric("letters",.8,"family_flip_rate")
        dp=metric("digits",1.2,"mean_abs_delta_p_letter")
        lp=metric("letters",.8,"mean_abs_delta_p_letter")
        allgroup=max(avg(metric(g,sc,"family_flip_rate")) for g in ("digits","letters") for sc in (.8,.9,1.1,1.2))
        focusmax,cl,scale=max((cls_data[name,c,sc]["flip_mean"],c,sc) for c in FOCUS for sc in EXTREMES)
        anymax=max(cls_data[name,c,sc]["flip_mean"] for c in LABELS for sc in EXTREMES)
        seedsmax=max(seed_focus[name,s] for s in SEEDS)
        failed=[]
        if avg(d12)>GATES["digit12flip"]+EPS:failed.append("digit_1p2_flip")
        if avg(l08)>GATES["letter08flip"]+EPS:failed.append("letter_0p8_flip")
        if avg(dp)>GATES["digit12p"]+EPS:failed.append("digit_1p2_prob_delta")
        if avg(lp)>GATES["letter08p"]+EPS:failed.append("letter_0p8_prob_delta")
        if max(d12+l08)>GATES["seedgroup"]+EPS:failed.append("seed_group_flip")
        if allgroup>GATES["allscale"]+EPS:failed.append("all_scale_group_flip")
        if avg(d)<GATES["digitsacc"]-EPS:failed.append("digit_accuracy")
        if avg(l)<GATES["lettersacc"]-EPS:failed.append("letter_accuracy")
        old_ok=not failed
        if focusmax>GATES["focusmax"]+EPS:failed.append("B8D_worst_flip")
        if anymax>GATES["allclassmax"]+EPS:failed.append("any16class_worst_flip")
        if seedsmax>GATES["seedfocusmax"]+EPS:failed.append("B8D_worst_per_seed_flip")
        ranked.append({"variant":name,"cohort":"EXP016_NEW" if name in models_new else "EXP015_FROZEN",
             "size_augmentation_pct":int(name.split("_")[0][4:]),
             "stroke_blending_pct":int(name.split("_")[1][6:]),
             "passes_legacy_group_gates":old_ok,"passes_all_predeclared_gates":not failed,
             "failed_gates":" | ".join(failed),
             "official_val_mean":avg(v),"official_val_stdev":statistics.stdev(v),
             "official_val_seed42":v[0],"official_val_seed43":v[1],"official_val_seed44":v[2],
             "handmade_digits_mean":avg(d),"handmade_letters_mean":avg(l),
             "handmade_all_mean":avg(accuracy("Handmade","all")),
             "digits_1p2_flip":avg(d12),"letters_0p8_flip":avg(l08),
             "digits_1p2_mean_abs_delta_p":avg(dp),"letters_0p8_mean_abs_delta_p":avg(lp),
             "max_all_group_scale_flip":allgroup,
             "max_seed_group_flip":max(d12+l08),
             "B8D_max_class_flip":focusmax,"B8D_worst_class":cl,"B8D_worst_scale":scale,
             "all16_max_class_flip":anymax,"B8D_max_single_seed_class_flip":seedsmax,
             "size_only_geometry_bal_acc":float(record[name,42]["size_only_val_bal_acc"]),
             "size_only_geometry_auc":float(record[name,42]["size_only_val_roc_auc"]),
             "actual_augmented_fraction":float(record[name,42]["actual_fraction"])})
    eligible=sorted([x for x in ranked if x["passes_all_predeclared_gates"]],
       key=lambda x:(x["all16_max_class_flip"],x["B8D_max_class_flip"],
                     -x["official_val_mean"],-min(x["handmade_digits_mean"],x["handmade_letters_mean"])))
    rejected=sorted([x for x in ranked if not x["passes_all_predeclared_gates"]],
       key=lambda x:(x["all16_max_class_flip"],x["B8D_max_class_flip"],-x["official_val_mean"]))
    ordered=eligible+rejected
    for j,x in enumerate(ordered):
        x["eligible_rank"]=j+1 if j<len(eligible) else ""
        x["decision"]="ELIGIBLE" if j<len(eligible) else "INELIGIBLE"
    save(out/"all_23_conditions_ranked_gates.csv",ordered)
    best_focus=min(ranked,key=lambda x:(x["B8D_max_class_flip"],x["all16_max_class_flip"]))
    metadata={"experiment":"EXP016","seeds":list(SEEDS),"new_MLPs":60,"new_conditions":20,
         "historical_frozen_controls":sorted(models_old),"candidates":len(ranked),
         "legacy_gate_pass":sum(x["passes_legacy_group_gates"] for x in ranked),
         "strict_all_gates_pass":len(eligible),"strict_top_candidate":eligible[0]["variant"] if eligible else None,
         "best_B8D_diagnostic":best_focus["variant"],"unchanged_preregistered_gates":GATES,
         "original_test_used":False,"professor_test_used":False,"validation_used_in_fit":False,
         "handmade_used_in_fit":False,"main_changed":False}
    (out/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# EXP유016 — 고비율 증강 60개 MLP 및 고정 3조건 통합 실측","",
           "모든 값은 성공한 3 Seed 학습 Artifact와 고정 EXP015 모델의 실제 CSV/NPZ에서 집계함.",
           f"23조건 중 과거 집단 Shortcut Gate 통과: {metadata['legacy_gate_pass']}개; "
           f"이번 개별 클래스 Gate 포함 전체 통과: **{len(eligible)}개**.","",
           "| 조건 | 적격 | 공식 Val | HM 숫자 | HM 문자 | 숫자 1.2 Flip | 문자 0.8 Flip | B8D 최악 | 16클래스 최악 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in ordered:
        lines.append("| "+x["variant"]+" | "+("통과" if x["passes_all_predeclared_gates"] else "탈락")+" | "+
            " | ".join(f"{100*x[k]:.2f}%" for k in
            ("official_val_mean","handmade_digits_mean","handmade_letters_mean",
             "digits_1p2_flip","letters_0p8_flip","B8D_max_class_flip","all16_max_class_flip"))+" |")
    lines+=["","## 사전 선택 기준 판정"]
    if eligible:
        w=eligible[0]
        lines.append(f"- 모든 안전 Gate 통과 후보 중 전체 16클래스 최악 Flip 최소: {w['variant']}.")
        lines.append(f"- 공식 Val {100*w['official_val_mean']:.4f}%; HM 숫자 {100*w['handmade_digits_mean']:.4f}%, 문자 {100*w['handmade_letters_mean']:.4f}%.")
    else:
        lines.append("- 안전 Gate를 전부 통과한 조건 없음. **최종 모델 채택 금지**.")
    lines += [f"- 참고: B8D 여섯 극단 변화 최저 조건 {best_focus['variant']}: {100*best_focus['B8D_max_class_flip']:.2f}% (전체 Gate 통과와 구분).",
              "- Family flip은 참값 예측 Accuracy와 동일하지 않음. Hand-made 반복 개발용 데이터이므로 독립 테스트를 의미하지 않음.",
              "- 0.8~1.2 배율 변형은 픽셀 질량과 보간도 바꾸므로 순수 Size 한 요소만 조작했다는 주장은 지양.",
              "- 기존 Main, 최종 모델, Original Test 및 교수님 Test 유지/미사용.",""]
    (out/"SUMMARY.md").write_text("\n".join(lines),encoding="utf-8")
    print("EXP016_AGGREGATE_SUCCESS",len(ranked),len(eligible),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--high-inputs",type=Path,required=True)
    p.add_argument("--historical-inputs",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.high_inputs,a.historical_inputs,a.out)
