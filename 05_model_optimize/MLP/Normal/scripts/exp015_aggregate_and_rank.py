#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP015: rank candidates ONLY after preregistered size-shortcut safeguards.

Three independent MLP random initializations (seeds 42/43/44); the original
Official Train, Official Val & Handmade datasets/splits are immutable.
Gate established BEFORE model results; no post hoc relaxing of thresholds.
"""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path
from collections import defaultdict
import numpy as np
from exp006_size_com_comparison import LABELS

SEEDS=(42,43,44)
PCTS=(10,20,30,40,50)
STROKES=(0,14,28)
# These are PREDECLARED from EXP014's successful, conservative mean responses.
GATE={
 "digit_1p2_flip_mean_max":0.10,
 "letter_0p8_flip_mean_max":0.10,
 "digit_1p2_probability_drift_mean_max":0.12,
 "letter_0p8_probability_drift_mean_max":0.12,
 "each_seed_critical_flip_max":0.15,
 "all_scale_family_flip_mean_max":0.12,
 "handmade_digits_mean_min":0.85,
 "handmade_letters_mean_min":0.80,
}
def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
def save(path,rows):
    if not rows:raise AssertionError(f"No data {path}")
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
def mean(values):
    assert values
    return float(sum(values)/len(values))
def run(inp,out):
    inp=Path(inp);out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    folders=[inp/f"exp015-s{s}-r{pct}" for s in SEEDS for pct in PCTS]
    for f in folders:
        assert (f/"manifest.json").exists(), f
        m=json.loads((f/"manifest.json").read_text(encoding="utf-8"))
        assert m["seed"] in SEEDS and int(round(100*m["ratio_requested"])) in PCTS
        assert m["original_predictions_verified"]
        assert m["balanced_per_class_per_target"] and m["nested_sampling_from_exp014_half"]
        assert m["actual_training_rows_per_variant"]==76800
        assert not m["official_test_used"] and not m["professor_test_used"]
        assert not m["handmade_used_in_fit"] and not m["official_val_used_in_fit"]
    metrics=[r for f in folders for r in read(f/"accuracy_metrics.csv")]
    sensitivity=[r for f in folders for r in read(f/"scale_sensitivity.csv")]
    training=[r for f in folders for r in read(f/"training_records.csv")]
    geometry=[r for f in folders for r in read(f/"training_geometry.csv")]
    pair=[r for f in folders for r in read(f/"pair_errors.csv")]
    save(out/"all_seed_accuracy_metrics.csv",metrics)
    save(out/"all_seed_scale_sensitivity.csv",sensitivity)
    save(out/"all_seed_training_records.csv",training)
    save(out/"all_seed_training_geometry.csv",geometry)
    save(out/"all_seed_pair_errors.csv",pair)
    names=sorted({r["variant"] for r in training})
    assert len(names)==len(PCTS)*len(STROKES),len(names)
    lookup_acc={}
    for r in metrics:
        if r["group"] not in ("all","digits","letters"):continue
        key=(r["variant"],int(r["seed"]),r["dataset"],r["group"])
        if key in lookup_acc:
            # Baseline repeats in ratio==10? only appears in the 10-percent runs.
            raise AssertionError(("duplicate",key))
        lookup_acc[key]=float(r["accuracy"])
    lookup_robust={}
    for r in sensitivity:
        if r["group"] not in ("digits","letters"):continue
        key=(r["variant"],int(r["seed"]),r["group"],float(r["scale"]))
        assert key not in lookup_robust,key
        lookup_robust[key]=r
    lookup_train={}
    for r in training:
        key=(r["variant"],int(r["seed"]))
        assert key not in lookup_train,key
        lookup_train[key]=r
    allrows=[]
    for name in names:
        parts=name.split("_")
        ratio=int(parts[0][4:])/100.
        stroke=int(parts[1][6:])/100.
        def scores(dataset,group):
            return [lookup_acc[name,s,dataset,group] for s in SEEDS]
        val=scores("OfficialVal","all")
        digits=scores("Handmade","digits")
        letters=scores("Handmade","letters")
        def q(group,scale,field):
            return [float(lookup_robust[name,s,group,scale][field]) for s in SEEDS]
        d12=q("digits",1.2,"family_flip_rate")
        l08=q("letters",.8,"family_flip_rate")
        d12p=q("digits",1.2,"mean_abs_delta_p_letter")
        l08p=q("letters",.8,"mean_abs_delta_p_letter")
        allmeanflip=max(mean(q(group,scale,"family_flip_rate"))
            for group in ("digits","letters") for scale in (.8,.9,1.1,1.2))
        reasons=[]
        if mean(d12)>GATE["digit_1p2_flip_mean_max"]:reasons.append("digit_1p2_flip_mean")
        if mean(l08)>GATE["letter_0p8_flip_mean_max"]:reasons.append("letter_0p8_flip_mean")
        if mean(d12p)>GATE["digit_1p2_probability_drift_mean_max"]:reasons.append("digit_1p2_p_drift")
        if mean(l08p)>GATE["letter_0p8_probability_drift_mean_max"]:reasons.append("letter_0p8_p_drift")
        if max(d12+l08)>GATE["each_seed_critical_flip_max"]:reasons.append("per_seed_flip")
        if allmeanflip>GATE["all_scale_family_flip_mean_max"]:reasons.append("full_range_flip")
        if mean(digits)<GATE["handmade_digits_mean_min"]:reasons.append("digits_accuracy")
        if mean(letters)<GATE["handmade_letters_mean_min"]:reasons.append("letters_accuracy")
        train_sizes=[lookup_train[name,s]["size_only_val_roc_auc"] for s in SEEDS]
        assert len(set(train_sizes))==1,"Augmented geometry should not depend on MLP seed"
        record={"variant":name,"augmentation_fraction":ratio,"stroke_strength":stroke,
            "pass_shortcut_and_accuracy_gates":not reasons,"failed_gates":" | ".join(reasons),
            "official_val_mean":mean(val),"official_val_seed_sd":statistics.stdev(val),
            "handmade_digits_mean":mean(digits),"handmade_letters_mean":mean(letters),
            "handmade_min_family_mean":min(mean(digits),mean(letters)),
            "handmade_all_mean":mean(scores("Handmade","all")),
            "digits_1p2_flip_mean":mean(d12),"letters_0p8_flip_mean":mean(l08),
            "digits_1p2_mean_abs_probability_drift":mean(d12p),
            "letters_0p8_mean_abs_probability_drift":mean(l08p),
            "critical_max_seed_flip":max(d12+l08),
            "all_scale_max_family_flip_mean":allmeanflip,
            "train_geometry_only_size_auc_on_official_val":float(train_sizes[0]),
            "official_val_seed42":val[0],"official_val_seed43":val[1],
            "official_val_seed44":val[2],
            "handmade_digits_seed42":digits[0],"handmade_digits_seed43":digits[1],
            "handmade_digits_seed44":digits[2],
            "handmade_letters_seed42":letters[0],"handmade_letters_seed43":letters[1],
            "handmade_letters_seed44":letters[2]}
        allrows.append(record)
    # Strict binary eligibility gate, THEN rank OfficialVal, THEN worst family.
    winners=sorted((r for r in allrows if r["pass_shortcut_and_accuracy_gates"]),
                   key=lambda x:(-x["official_val_mean"],-x["handmade_min_family_mean"],
                                 x["all_scale_max_family_flip_mean"]))
    failures=sorted((r for r in allrows if not r["pass_shortcut_and_accuracy_gates"]),
                    key=lambda x:(len(x["failed_gates"].split(" | ")),-x["official_val_mean"]))
    ordered=winners+failures
    for i,r in enumerate(ordered):
        r["rank_among_eligible"]=i+1 if i<len(winners) else ""
        r["selection_status"]="ELIGIBLE" if i<len(winners) else "INELIGIBLE"
    save(out/"preregistered_gate_and_ranking.csv",ordered)
    # Additional class-specific mean family-flip of the direct paired counterfactual.
    class_rows=[]
    for name in names:
        ratio=int(name.split("_")[0][4:])
        for scale in (.8,1.2):
            for cl in ("0","8","5","B","D","F"):
                seed_flips=[];seed_abs=[]
                for seed in SEEDS:
                    f=inp/f"exp015-s{seed}-r{ratio}"
                    data=np.load(f/f"response_{name}.npz",allow_pickle=False)
                    ss=data["scales"].astype(np.float64)
                    ix=int(np.argmin(np.abs(ss-scale)));base=int(np.argmin(np.abs(ss-1.)))
                    p=data["p_letter"];lbl=data["y_true"]
                    mask=lbl==LABELS.index(cl)
                    assert np.sum(mask)==160 and abs(float(ss[ix])-scale)<1e-6
                    seed_flips.append(float(np.mean((p[ix,mask]>=.5)!=(p[base,mask]>=.5))))
                    seed_abs.append(float(np.mean(np.abs(p[ix,mask]-p[base,mask]))))
                class_rows.append({"variant":name,"class":cl,"scale":scale,
                    "n_unique_images_per_seed":160,
                    "family_flip_mean_over_seeds":mean(seed_flips),
                    "mean_abs_delta_p_letter":mean(seed_abs)})
    save(out/"class_specific_size_flip.csv",class_rows)
    # The frozen archive control is measured in the 10% jobs only.
    baseline={"OfficialVal":{},"Handmade":{}}
    for ds in baseline:
        for group in ("all","digits","letters"):
            scores=[lookup_acc["baseline_com",s,ds,group] for s in SEEDS]
            baseline[ds][group]={"mean":mean(scores),"stdev":statistics.stdev(scores)}
    control={}
    for group,scale in (("digits",1.2),("letters",.8)):
        control[group+"_"+str(scale)]={
          "mean_family_flip":mean(
               [float(lookup_robust["baseline_com",s,group,scale]["family_flip_rate"])
                for s in SEEDS]),
          "mean_abs_delta_p_letter":mean(
               [float(lookup_robust["baseline_com",s,group,scale]["mean_abs_delta_p_letter"])
                for s in SEEDS])}
    (out/"metadata.json").write_text(json.dumps({
        "experiment":"EXP015","three_seeds":list(SEEDS),"total_grid_conditions":len(names),
        "all_conditions_fit_on_original_train_only":True,"gate":GATE,
        "eligible_count":len(winners),"selected_best":winners[0]["variant"] if winners else None,
        "fixed_original_exp006_com_accuracy":baseline,"fixed_original_exp006_com_scale":control,
        "original_test_used":False,"professor_test_used":False,
        "main_or_final_checkpoint_changed":False
    },ensure_ascii=False,indent=2),encoding="utf-8")
    md=["# EXP유015 — 완전 검증된 3-Seed Grid 결과","",
        "**주의: 수치는 완료된 GitHub Actions Artifact의 실제 결과다.**",
        "Original Test 및 교수님 Test 미사용, Hand-made는 재사용 개발용 데이터.",
        f"후보 수: **{len(names)}**, Shortcut·숫자/문자 Accuracy 사전 게이트 통과: **{len(winners)}**.",
        "",
        "## 사전에 고정한 Shortcut 안전 게이트",
        "- 1.2배 숫자 및 0.8배 문자 Family flip 평균 각각 **10% 이하**",
        "- 같은 개입의 평균 절대 문자 확률 변화 각각 **0.12 이하**",
        "- 세 Seed 중 어느 한 Seed도 핵심 개입 Family flip이 **15% 초과 불가**",
        "- 0.8·0.9·1.1·1.2배의 모든 숫자/문자 집단에서 평균 flip **12% 이하**",
        "- 기본 Hand-made 숫자 **85% 이상**, 문자 **80% 이상** (각 3 Seed 평균)",
        "위 조건을 **모두 통과한 경우에만** OfficialVal 평균이 높은 순으로 비교한다.",
        "",
        "## 모든 후보",
        "| 조건 | 사전 Gate | OfficialVal | Handmade 숫자 | Handmade 문자 | 숫자×1.2 Flip | 문자×0.8 Flip |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in ordered:
        md.append(f"| {row['variant']} | {'통과' if row['pass_shortcut_and_accuracy_gates'] else '탈락'} | "
           f"{100*row['official_val_mean']:.3f}% | "
           f"{100*row['handmade_digits_mean']:.3f}% | "
           f"{100*row['handmade_letters_mean']:.3f}% | "
           f"{100*row['digits_1p2_flip_mean']:.2f}% | "
           f"{100*row['letters_0p8_flip_mean']:.2f}% |")
    md+=["","## 사전 선정 규칙에 따른 결론"]
    if winners:
        best=winners[0]
        md += [f"- 게이트 통과 최상위 후보: **{best['variant']}**.",
               f"- OfficialVal: **{100*best['official_val_mean']:.4f}%**; "
               f"Handmade 숫자 **{100*best['handmade_digits_mean']:.4f}%**, "
               f"문자 **{100*best['handmade_letters_mean']:.4f}%**.",
               f"- 숫자×1.2 Family flip **{100*best['digits_1p2_flip_mean']:.3f}%**, "
               f"문자×0.8 Family flip **{100*best['letters_0p8_flip_mean']:.3f}%**.",
               f"- OfficialVal 97.30% 목표 {'충족' if best['official_val_mean']>=.973 else '미달'}."]
    else:
        md+=["- **통과 후보 없음:** Shortcut 억제 기준을 완화하거나 실패 후보를 임의 채택하지 않음."]
    md+=["",
      "### 해석상 주의",
      "- Family flip 감소는 크기 Shortcut 의존성 *완화*의 행동 근거이지 완전 제거의 인과적 증명은 아니다.",
      "- CoM-only 입력 전처리와 Augmentation 학습만 바뀌며 평가 입력에 증강 적용하지 않음.",
      "- OfficialVal과 Hand-made는 이미 반복 사용되었기 때문에 이 3-seed Grid의 최고 조건은 개발용 최적화 결과다. 새로운 작성자의 독립 평가 전까지 일반화 보증은 아님.",
      "- 원본 데이터와 main/최종 모델을 변경하지 않는다. 승인 전에 새 후보를 배포·적용하지 않는다.",
      ""]
    (out/"SUMMARY.md").write_text("\n".join(md),encoding="utf-8")
    print("EXP015_AGGREGATE_SUCCESS",len(names),"eligible",len(winners),
          "winner",winners[0]["variant"] if winners else None,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.inputs,a.out)
