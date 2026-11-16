#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP017 final coarse+fine joint ranking; OfficialVal-only, no handwritten data.

Original Test/professor Test unaccessed. No subsequent tuning against Hand-made.
"""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path
from exp017_hparam_coarse_aggregate import GATE,FOCUS,CLASSES,avg
from exp017_hparam_coarse import CONF,config_name,SEEDS
from exp017_hparam_fine import configs as fine_configs

def read(path):
    with Path(path).open("r",encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))
def save(path,rows):
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run(coarse_inputs,fine_inputs,coarse_json,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    cset={config_name(h,a,lr):{"hidden":list(h),"alpha":a,"lr":lr,
        "early_stopping":False,"source":"coarse"} for h,a,lr in CONF}
    fset={x["variant"]:{**x,"source":"fine"} for x in fine_configs(coarse_json)}
    names={**cset,**fset}
    assert len(names)==46 and len(cset)==26 and len(fset)==20
    data={};paths={};fit={};sha=set()
    for seed in SEEDS:
      for stage,root,splits in (("coarse",coarse_inputs,3),("fine",fine_inputs,4)):
        for j in range(splits):
            tag=f"exp017-coarse-s{seed}-p{j}" if stage=="coarse" else f"exp017-fine-s{seed}-slot{j}"
            f=Path(root)/tag
            man=json.loads((f/"manifest.json").read_text(encoding="utf-8"))
            assert man["experiment"]=="EXP017" and man["stage"]==stage and man["seed"]==seed
            assert not any(man[k] for k in ("team_handwriting_loaded","handmade_used","original_test_used","professor_test_used"))
            assert man["official_val_rows"]==19200 and man["augmented_rows"]==61440
            assert (man["training_rows"] if stage=="coarse" else man["training_rows_source"])==76800
            sha.add(man["original_official_sha256"] if stage=="coarse" else man["official_source_sha256"])
            for record in read(f/"training_records.csv"):
                name=record["variant"]
                assert name in names and names[name]["source"]==stage
                assert (name,seed) not in fit
                fit[name,seed]=record
                paths[name,seed]=str(f)
            for r in read(f/"metrics.csv"):
                name=r["variant"]
                key=(name,seed,r["group"],float(r["scale"]))
                assert key not in data
                data[key]=(float(r["accuracy"]),float(r["family_flip"]),float(r["abs_delta_p_letter"]))
    assert len(sha)==1 and len(fit)==46*3,(sha,len(fit))
    def vals(name,grp,scale,index):return [data[name,s,grp,scale][index] for s in SEEDS]
    rows=[]
    for name,c in names.items():
        val=vals(name,"all",1.,0)
        digit=vals(name,"digits",1.,0);letter=vals(name,"letters",1.,0)
        d12=vals(name,"digits",1.2,1);l08=vals(name,"letters",.8,1)
        dp=vals(name,"digits",1.2,2);lp=vals(name,"letters",.8,2)
        focus=max((avg(vals(name,"class_"+cls,sc,1)),cls,sc) for cls in FOCUS for sc in (.8,1.2))
        all16=max((avg(vals(name,"class_"+cls,sc,1)),cls,sc) for cls in CLASSES for sc in (.8,1.2))
        seedmax=max(data[name,seed,"class_"+cls,sc][1] for seed in SEEDS for cls in FOCUS for sc in (.8,1.2))
        failed=[]
        if avg(val)<GATE["min_official_val"]:failed.append("val_below_anchor_minus_0p5pp")
        if focus[0]>GATE["max_focus_flip"]:failed.append("B8D_above_20pct")
        if all16[0]>GATE["max_any16_flip"]:failed.append("anyclass_above_25pct")
        if seedmax>GATE["max_single_seed_focus_flip"]:failed.append("B8D_seed_above_30pct")
        if avg(d12)>GATE["max_digits12_flip"]:failed.append("digits_1p2_flip_above_10pct")
        if avg(l08)>GATE["max_letters08_flip"]:failed.append("letters_0p8_flip_above_10pct")
        if avg(dp)>GATE["max_digits12_pdrift"]:failed.append("digits_probability_drift")
        if avg(lp)>GATE["max_letters08_pdrift"]:failed.append("letters_probability_drift")
        if avg(digit)<GATE["min_digit_acc"]:failed.append("digit_accuracy_below_85pct")
        if avg(letter)<GATE["min_letter_acc"]:failed.append("letter_accuracy_below_85pct")
        rows.append({"variant":name,"source":c["source"],
          "hidden":str(c["hidden"]),"alpha":c["alpha"],
          "learning_rate_init":c["lr"] if c["source"]=="coarse" else c["learning_rate_init"],
          "early_stopping":c["early_stopping"],
          "passes_all_gates":not failed,"fail_count":len(failed),"failed_gates":" | ".join(failed),
          "OfficialVal_mean":avg(val),"OfficialVal_stdev":statistics.stdev(val),
          "OfficialVal_seed42":val[0],"OfficialVal_seed43":val[1],"OfficialVal_seed44":val[2],
          "Val_digit_accuracy":avg(digit),"Val_letter_accuracy":avg(letter),
          "Val_scale08_accuracy":avg(vals(name,"all",.8,0)),
          "Val_scale12_accuracy":avg(vals(name,"all",1.2,0)),
          "digits_1p2_flip":avg(d12),"letters_0p8_flip":avg(l08),
          "digits_1p2_pdrift":avg(dp),"letters_0p8_pdrift":avg(lp),
          "B8D_max_flip":focus[0],"B8D_class":focus[1],"B8D_scale":focus[2],
          "all16_max_flip":all16[0],"all16_class":all16[1],
          "all16_scale":all16[2],"B8D_max_one_seed":seedmax,
          "training_iterations_mean":avg([float(fit[name,s]["n_iter"]) for s in SEEDS])})
    eligible=sorted([x for x in rows if x["passes_all_gates"]],
          key=lambda x:(-x["OfficialVal_mean"],x["B8D_max_flip"],x["all16_max_flip"]))
    excluded=sorted([x for x in rows if not x["passes_all_gates"]],
          key=lambda x:(x["fail_count"],x["B8D_max_flip"],-x["OfficialVal_mean"]))
    ranked=eligible+excluded
    for j,r in enumerate(ranked):
        r["gate_rank"]=j+1 if j<len(eligible) else ""
        r["status"]="ELIGIBLE" if j<len(eligible) else "NOT_ELIGIBLE"
    save(out/"all_46_configs_3seed_final_ranking.csv",ranked)
    def meta(r):
        return None if r is None else {"variant":r["variant"],"source":r["source"],
          "hidden":r["hidden"],"alpha":r["alpha"],"learning_rate_init":r["learning_rate_init"],
          "early_stopping":r["early_stopping"],"OfficialVal_mean":r["OfficialVal_mean"],
          "B8D_max_flip":r["B8D_max_flip"],"all16_max_flip":r["all16_max_flip"]}
    leader=eligible[0] if eligible else None
    runner=eligible[1] if len(eligible)>1 else None
    fallback=excluded[0] if excluded else None
    md={"experiment":"EXP017","stage":"joint_final","n_coarse":26,"n_fine":20,
        "n_models":46*3,"n_total_conditions":46,
        "seed_count":3,"gates":GATE,"official_val_source_sha256":list(sha)[0],
        "all_gates_passing":len(eligible),"selected_frozen_candidate":meta(leader),
        "runner_up":meta(runner),"best_noneligible_diagnostic":meta(fallback),
        "team_handwriting_evaluated":False,"original_test_evaluated":False,
        "professor_test_evaluated":False,
        "stage1_frozen_80_14_anchor_officialval":.9481076388888888,
        "stage1_frozen_80_14_anchor_B8D_max_officialval":.25556}
    (out/"metadata.json").write_text(json.dumps(md,ensure_ascii=False,indent=2),encoding="utf-8")
    summary=["# EXP유017 MLP HP — OfficialVal-only 통합 최종 결과",
      "",f"1차 26개 + 2차 20개 = **46조합×3 Seed=138개 학습 완료**. 통과 조합 **{len(eligible)}개**.",
      "**팀 손글씨/기존 Hand-made/Original Test/교수님 Test는 튜닝에 일절 사용하지 않음.**",
      "기존 80-14 OfficialVal 94.8108%, B·D·8 최악 확대·축소 판단전환 25.56%.",
      "",
      "| 모델 | 단계 | 통과 | OfficialVal | 숫자×1.2 Flip | 문자×0.8 Flip | B8D 최악 | 16클래스 최악 |",
      "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in ranked:
        summary.append("| "+r["variant"]+" | "+r["source"]+" | "+
         ("Yes" if r["passes_all_gates"] else "No")+" | "+
         " | ".join(f"{100*r[k]:.2f}%" for k in
           ("OfficialVal_mean","digits_1p2_flip","letters_0p8_flip","B8D_max_flip","all16_max_flip"))+" |")
    summary+=["","## 완료 판정"]
    if leader:
        summary += [f"- **동결 후보: {leader['variant']}** (기존 80-14 전처리 그대로 유지)",
           f"- OfficialVal {100*leader['OfficialVal_mean']:.4f}%, B8D 최악 {100*leader['B8D_max_flip']:.2f}%, 16클래스 최악 {100*leader['all16_max_flip']:.2f}%.",
           f"- 97.30% 목표: {'충족' if leader['OfficialVal_mean']>=.973 else '미달'}.",
           "- 팀원 손글씨 최종 Validation 전까지 배포/메인 반영 금지."]
    else:
        summary += ["- **사전등록된 Gate 전부 통과 조건 없음**. 최종 모델 채택·동결 금지.",
            f"- 최상위 차선책은 {fallback['variant'] if fallback else '없음'}이나 기준 미달이며 최종 후보로 선언하지 않는다."]
    (out/"SUMMARY.md").write_text("\n".join(summary)+"\n",encoding="utf-8")
    print("EXP017_FINAL_AGGREGATE_OK",len(eligible),"leader",leader["variant"] if leader else None,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--coarse-inputs",required=True,type=Path)
    p.add_argument("--fine-inputs",required=True,type=Path)
    p.add_argument("--coarse-json",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    run(a.coarse_inputs,a.fine_inputs,a.coarse_json,a.out)
