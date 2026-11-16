#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-1 EXP017 predeclared 3-seed gate-based OfficialVal aggregation.

No training and no handwriting loading. Returns up to four DIFFERENT hidden
layer architectures for stage-2 fine hyperparameter search.
"""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path
from collections import defaultdict
from exp017_hparam_coarse import CONF,SEEDS,config_name
GATE={"min_official_val":.9431076388888888, # frozen 80-14 0.9481076389 - 0.005
      "max_focus_flip":.20,"max_any16_flip":.25,"max_single_seed_focus_flip":.30,
      "max_digits12_flip":.10,"max_letters08_flip":.10,
      "max_digits12_pdrift":.12,"max_letters08_pdrift":.12,
      "min_digit_acc":.85,"min_letter_acc":.85}
FOCUS=("8","B","D")
CLASSES=tuple("0123456789ABCDEF")
SCALES=(.8,1.,1.2)
def read_csv(path):
    with Path(path).open("r",encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    assert rows
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def avg(values):return sum(values)/len(values)
def run(inputs,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    config={config_name(h,a,lr):{"hidden":list(h),"alpha":a,"lr":lr}
            for h,a,lr in CONF}
    assert len(config)==26
    raw=defaultdict(list);train=defaultdict(list);hashes=set()
    for seed in SEEDS:
      for part in (0,1,2):
        path=Path(inputs)/f"exp017-coarse-s{seed}-p{part}"
        manifest=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
        assert manifest["experiment"]=="EXP017" and manifest["seed"]==seed and manifest["part"]==part
        assert manifest["training_rows"]==76800 and manifest["official_val_rows"]==19200
        assert manifest["augmented_rows"]==61440
        assert manifest["size_aug_ratio"]==.8 and manifest["stroke_blend"]==.14
        assert not any(manifest[k] for k in ("team_handwriting_loaded","handmade_used","original_test_used","professor_test_used"))
        hashes.add(manifest["original_official_sha256"])
        for r in read_csv(path/"training_records.csv"):
            name=r["variant"];key=(name,seed)
            assert name in config and not train[key]
            train[key].append(r)
        for r in read_csv(path/"metrics.csv"):
            key=(r["variant"],seed,r["group"],float(r["scale"]))
            assert not raw[key],key
            raw[key].append(float(r["accuracy"]))
            raw[key].append(float(r["family_flip"]))
            raw[key].append(float(r["abs_delta_p_letter"]))
    assert len(hashes)==1 and len(train)==26*3
    assert all((name,seed) in train for name in config for seed in SEEDS)
    def vals(name,group,scale,idx):
        return [raw[name,seed,group,scale][idx] for seed in SEEDS]
    ranked=[]
    for name,v in config.items():
        v_acc=vals(name,"all",1.,0)
        d_acc=vals(name,"digits",1.,0); l_acc=vals(name,"letters",1.,0)
        dflip=vals(name,"digits",1.2,1); lflip=vals(name,"letters",.8,1)
        dp=vals(name,"digits",1.2,2);lp=vals(name,"letters",.8,2)
        focal=max((avg(vals(name,"class_"+c,scale,1)),c,scale) for c in FOCUS for scale in (.8,1.2))
        all16=max((avg(vals(name,"class_"+c,scale,1)),c,scale) for c in CLASSES for scale in (.8,1.2))
        focus_seed=max(raw[name,seed,"class_"+c,scale][1] for seed in SEEDS
                       for c in FOCUS for scale in (.8,1.2))
        failed=[]
        if avg(v_acc)<GATE["min_official_val"]:failed.append("val_below_anchor_minus_0p5pp")
        if focal[0]>GATE["max_focus_flip"]:failed.append("B8D_above_20pct")
        if all16[0]>GATE["max_any16_flip"]:failed.append("anyclass_above_25pct")
        if focus_seed>GATE["max_single_seed_focus_flip"]:failed.append("B8D_seed_above_30pct")
        if avg(dflip)>GATE["max_digits12_flip"]:failed.append("digits_1p2_flip_above_10pct")
        if avg(lflip)>GATE["max_letters08_flip"]:failed.append("letters_0p8_flip_above_10pct")
        if avg(dp)>GATE["max_digits12_pdrift"]:failed.append("digits_probability_drift")
        if avg(lp)>GATE["max_letters08_pdrift"]:failed.append("letters_probability_drift")
        if avg(d_acc)<GATE["min_digit_acc"]:failed.append("digit_accuracy_below_85pct")
        if avg(l_acc)<GATE["min_letter_acc"]:failed.append("letter_accuracy_below_85pct")
        row={"variant":name,"hidden":str(v["hidden"]),"alpha":v["alpha"],"learning_rate_init":v["lr"],
          "pass_all_gates":not failed,"failed_gates":" | ".join(failed),"n_failed_gates":len(failed),
          "officialval_acc_mean":avg(v_acc),"officialval_acc_sd":statistics.stdev(v_acc),
          "officialval_seed42":v_acc[0],"officialval_seed43":v_acc[1],"officialval_seed44":v_acc[2],
          "officialval_digits_mean":avg(d_acc),"officialval_letters_mean":avg(l_acc),
          "officialval_scale08_acc":avg(vals(name,"all",.8,0)),
          "officialval_scale12_acc":avg(vals(name,"all",1.2,0)),
          "digits12_flip":avg(dflip),"letters08_flip":avg(lflip),
          "digits12_pdrift":avg(dp),"letters08_pdrift":avg(lp),
          "B8D_max_flip":focal[0],"B8D_worst_class":focal[1],"B8D_worst_scale":focal[2],
          "max16class_flip":all16[0],"max16_worst_class":all16[1],
          "max16_worst_scale":all16[2],"max_single_seed_B8D_flip":focus_seed,
          "mean_n_iter":avg([float(train[name,seed][0]["n_iter"]) for seed in SEEDS])}
        ranked.append(row)
    eligible=sorted([x for x in ranked if x["pass_all_gates"]],
        key=lambda x:(-x["officialval_acc_mean"],x["B8D_max_flip"],x["max16class_flip"]))
    rejected=sorted([x for x in ranked if not x["pass_all_gates"]],
        key=lambda x:(x["n_failed_gates"],x["B8D_max_flip"],-x["officialval_acc_mean"]))
    ordered=eligible+rejected
    selected=[];seen=set()
    for row in ordered:
        if row["hidden"] in seen:continue
        selected.append(row)
        seen.add(row["hidden"])
        if len(selected)==4:break
    assert len(selected)==4
    names={r["variant"] for r in selected}
    for j,row in enumerate(ordered):
        row["overall_rank"]=j+1
        row["stage2_selected"]=row["variant"] in names
    write_csv(out/"all_26_configs_3seed_gates.csv",ordered)
    metadata={"experiment":"EXP017","stage":"coarse","models":78,"configurations":26,
      "official_train_n":76800,"official_val_n":19200,
      "team_handwriting_used":False,"original_test_used":False,"professor_test_used":False,
      "official_source_sha256":list(hashes)[0],"gate_predeclared":GATE,
      "passing_configs":len(eligible),"selected_4_architectures":[
           {"variant":x["variant"],"hidden":config[x["variant"]]["hidden"],
            "alpha":config[x["variant"]]["alpha"],"lr":config[x["variant"]]["lr"],
            "passes_all_gates":x["pass_all_gates"]} for x in selected]}
    (out/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    summary=["# EXP유017 — OfficialVal 전용 26구성 MLP 3-Seed 탐색",
       "",f"총 **78개 모델**; 안전 Gate 통과 구성: **{len(eligible)}/26**.",
       "Hand-made/팀 수집 손글씨를 읽거나 모델 선택에 쓰지 않음.",
       "",
       "| 조건 | 합격 | OfficialVal | 숫자×1.2 Flip | 문자×0.8 Flip | B8D 최악 | 16클래스 최악 |",
       "|---|---|---:|---:|---:|---:|---:|"]
    for r in ordered:
        summary.append(f"| {r['variant']} | {'O' if r['pass_all_gates'] else 'X'} | "+
          " | ".join(f"{r[k]*100:.2f}%" for k in 
              ("officialval_acc_mean","digits12_flip","letters08_flip","B8D_max_flip","max16class_flip"))+" |")
    summary+=["","## 다음 단계 후보(서로 다른 4개 구조)"]+[
       f"- {s['variant']} — {'Gate 통과' if s['pass_all_gates'] else 'Gate 미달·대체 후보'}"
       for s in metadata["selected_4_architectures"]]
    (out/"SUMMARY.md").write_text("\n".join(summary)+"\n",encoding="utf-8")
    print("EXP017_COARSE_AGGREGATE_OK",len(eligible),"stage2",metadata["selected_4_architectures"],flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--inputs",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    x=a.parse_args();run(x.inputs,x.out)
