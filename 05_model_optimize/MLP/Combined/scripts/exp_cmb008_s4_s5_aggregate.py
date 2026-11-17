#!/usr/bin/env python3
"""EXP-CMB-008 S4/S5 sequential, evidence-first audit.

S4: LR=.0005/.001/.002 with frozen Size30%-Stroke28%, CoM, (256,), alpha=.0001,
    Adam/ReLU, early_stopping=False, with S1 original LR=.001 as exact baseline.
S5: keep LR selected by S4; compare early_stopping=False to True (training-only
    internal validation). Reuse S4/S1 ES=False models. Never touch independent Test.
"""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path
from exp_cmb008_s1_aggregate import (
    SEEDS,WRITERS,readcsv,summarize,check,target_under_5percent,paired,pct
)

PAIRS=("D->0","B->8","8->B","0->D")
LR_KEYS=("0.0005","0.001","0.002")
REQUIRED_FILES=("manifest.json","oof_predictions.csv","fold_metrics.csv",
                "per_class_and_scale.csv","training_time.csv","training_loss.csv")
def read_run(directory,seed,lr,stage,early_stopping):
    assert directory.is_dir(),f"Artifact absent: {directory}"
    for required in REQUIRED_FILES:assert (directory/required).is_file(),directory/required
    m=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    assert m["experiment"]=="EXP-CMB-008" and m["arm"]=="scale30_stroke28"
    assert int(m["seed"])==seed and m["augmentation_seed"]==20261116
    assert m["handmade_train_preprocess"]=="CoM-only"
    assert m["heldout_and_official_val_preprocess"]=="CoM-only"
    assert m["official_train_only_augmentation"]
    assert not any(m[x] for x in ("original_test_used","professor_test_used","new_handmade_480_used"))
    assert m["hidden_layer_sizes"]==[256] if stage!="S1_width" else m["width"]==256
    expected={"activation":"relu","solver":"adam","alpha":.0001,
              "learning_rate_init":float(lr),"max_iter":200,"early_stopping":early_stopping}
    if stage=="S1_width":
        assert lr=="0.001" and not early_stopping
        assert m["stage"]=="S1_width" and m["only_changed_parameter"]=="hidden_layer_sizes"
    else:
        expected["hidden_layer_sizes"]=[256]
        assert m["stage"]==stage and m["only_changed_parameter"]==(
            "learning_rate_init" if stage=="S4_learning_rate" else "early_stopping")
        assert abs(float(m["learning_rate_init"])-float(lr))<1e-12
        if stage=="S5_early_stopping":
            assert early_stopping
            assert m["early_stopping_internal_validation_fraction"]==.1
            assert m["early_stopping_n_iter_no_change"]==10
    assert m["fixed_params"]==expected,(directory,m["fixed_params"],expected)
    oof=readcsv(directory/"oof_predictions.csv")
    assert len(oof)==2560 and len({(int(x["writer"]),int(x["index"])) for x in oof})==2560
    assert {int(x["writer"]) for x in oof}==set(WRITERS)
    assert all(x["actual"] in "0123456789ABCDEF" and x["pred"] in "0123456789ABCDEF" for x in oof)
    vals=[x for x in readcsv(directory/"fold_metrics.csv") if x["dataset"]=="OfficialVal" and x["group"]=="all"]
    assert len(vals)==4 and all(int(x["n"])==19200 for x in vals)
    scales=[x for x in readcsv(directory/"per_class_and_scale.csv") if
            x["dataset"] in ("OOF","OfficialVal") and float(x["scale"]) in (.8,1.2)]
    assert scales
    times=readcsv(directory/"training_time.csv")
    assert len(times)==4 and all(int(x["train_n"])==78720 for x in times)
    loss=readcsv(directory/"training_loss.csv")
    assert loss
    byfold={}
    for r in loss:
        k=int(r["heldout"])
        byfold.setdefault(k,[]).append((int(r["iteration"]),float(r["loss"])))
    assert set(byfold)==set(WRITERS)
    assert all(byfold[w][0][0]==1 and len(byfold[w])==int(next(x for x in times if int(x["heldout"])==w)["iterations"]) for w in WRITERS)
    return {"m":m,"oof":oof,"val":vals,"scales":scales,"times":times,
            "loss_start":statistics.mean(byfold[w][0][1] for w in WRITERS),
            "loss_end":statistics.mean(byfold[w][-1][1] for w in WRITERS)}
def load_group(paths,lr,stage,es):
    return {seed:read_run(paths[seed],seed,lr,stage,es) for seed in SEEDS}
def verify_sha(groups):
    baseline=next(iter(groups.values()))
    for key,data in groups.items():
        for seed in SEEDS:
            assert data[seed]["m"]["official_sha256"]==baseline[seed]["m"]["official_sha256"],(key,seed,"official SHA mismatch")
            assert data[seed]["m"]["handmade_sha256"]==baseline[seed]["m"]["handmade_sha256"],(key,seed,"handmade SHA mismatch")
            assert data[seed]["m"]["augmentation"]["requested_ratio"]==.3
def rank(base,metrics):
    guards={key:check(base,val) for key,val in metrics.items()}
    strict=[k for k in metrics if target_under_5percent(metrics[k]) and all(guards[k].values())]
    safe=[k for k in metrics if all(guards[k].values())]
    assert safe, "The baseline itself must pass non-regression gates"
    pool=strict or safe
    maximum=max(metrics[k]["oof"] for k in pool)
    near=[k for k in pool if metrics[k]["oof"]>=maximum-.002-1e-12]
    near.sort(key=lambda k:(metrics[k]["pairs"]["D->0"],
        sum(metrics[k]["pairs"][x] for x in PAIRS if x!="D->0"),
        -metrics[k]["oof"],metrics[k]["seed_std"],-metrics[k]["val"],
        float(k) if k in LR_KEYS else (0 if k=="False" else 1)))
    selected=near[0]
    return guards,strict,safe,selected
def build_rows(metrics,guards):
    results=[]
    for key,val in metrics.items():
        gate=guards[key]
        record={"candidate":key,"oof_accuracy":val["oof"],
          "oof_misclassification":1-val["oof"],"official_val_accuracy":val["val"],
          "seed_std":val["seed_std"],"worst_writer_accuracy":min(val["writers"].values()),
          "D_to_0":val["pairs"]["D->0"],"B_to_8":val["pairs"]["B->8"],
          "8_to_B":val["pairs"]["8->B"],"0_to_D":val["pairs"]["0->D"],
          "D_to_0_rate":val["pairs"]["D->0"]/480,"B_to_8_rate":val["pairs"]["B->8"]/480,
          "8_to_B_rate":val["pairs"]["8->B"]/480,"0_to_D_rate":val["pairs"]["0->D"]/480,
          "class_0_error_rate":val["classerror"]["0"],"class_8_error_rate":val["classerror"]["8"],
          "class_B_error_rate":val["classerror"]["B"],"class_D_error_rate":val["classerror"]["D"],
          "oof_max_family_flip":val["flip"]["OOF"],"val_max_family_flip":val["flip"]["OfficialVal"],
          "mean_iterations":val["mean_n_iter"],"mean_fit_seconds":val["mean_fit_seconds"],
          "warning_folds":val["warnings"],"all_four_directions_under_5pct":target_under_5percent(val),
          "all_fallback_safeguards_pass":all(gate.values()),
          "failed_gates":";".join(k for k,v in gate.items() if not v)}
        results.append(record)
    return results
def save_csv(p,rows):
    with p.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
def format_details(stage,baseline_key,selected_key,groups,metrics,guards,strict,safe,paired_errors,loss_by_key):
    return {"experiment":"EXP-CMB-008","stage":stage,"selected":selected_key,
      "status":"TARGET_MET" if strict else ("BASELINE_RETAINED" if selected_key==baseline_key else "FALLBACK_IMPROVED"),
      "selected_is_independent_final_test":False,
      "all_four_directions_under_5pct_candidates":strict,
      "safe_fallback_candidates":safe,
      "target_rate_each_direction":"strictly below 5%", "max_count_per_direction":23,
      "actual_class_denominator":480,
      "criterion":"if none <5% with gates, maximize OOF accuracy with no regression in D->0/B->8/8->B/0->D, errors of 0/8/B/D, or overfit safeguards",
      "scores":metrics,"gates":guards,"paired_corrected_and_induced":paired_errors,
      "average_first_and_last_training_loss":loss_by_key,
      "same_official_and_handmade_sha_confirmed":True,
      "no_independent_test_used":True,"training_seeds":list(SEEDS),
      "heldout_writers":list(WRITERS)}
def write_report(stage,baseline_key,selected,rows,metrics,guards,strict,paired_errors,loss,extra):
    name="S4 Learning Rate" if stage=="S4_learning_rate" else "S5 Early Stopping"
    why=(
      "S3에서 L2 강도를 높이자 일부 정확도는 증가했으나 D→0 등 핵심 오분류가 악화되었다. alpha=1e-4, (256,)을 유지한 채 Adam 초기 학습률을 변화시켜 수렴·정확도·오분류를 분리 비교했다."
      if stage=="S4_learning_rate" else
      "S4에서 실제 선정된 LR을 고정하고 Training Fold 내부 10%만 사용하는 Early Stopping을 적용했다. 같은 LR의 ES=False 원본 모델을 재사용해 조기 종료 효과 및 과소·과적합을 비교했다.")
    md=[f"# EXP-CMB-008 — {name} 실측·결정",
        "",f"> **단계: {stage} · 판단: {'4방향 5% 목표 달성' if strict else '4방향 5% 목표 미달'}**",
        f"> 기존 기준 {baseline_key} / 선택 {selected}; 이는 개발용 OOF 기반 선택이며 최종 독립 Test의 성능이 아니다.","",
        "## 이전 실험 → 이번 실험: 후보를 정한 이유",why,
        "고정: Official Train의 Size30%/Stroke28% 증강, CoM, 단층 (256,), alpha=1e-4, ReLU/Adam, 3 Seed × 4 Writer CV, 데이터 SHA. "
        +( "ES=False 고정; LR만 0.0005/0.001/0.002 비교." if stage=="S4_learning_rate" else
           "S4에서 선정한 LR을 고정하고 ES=False/True만 변경."),"",
        "## 전체 정확도와 중요 혼동 방향 (실제 클래스당 480회 예측)",
        "| 후보 | OOF 정확도 | OOF 오분류율 | OfficialVal | D→0 | B→8 | 8→B | 0→D | 네 방향 <5% | 비악화·일반화 Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in rows:
        def count(k):return str(r[k])+" ("+pct(r[k+"_rate"])+")"
        md.append("| "+str(r["candidate"])+" | "+pct(r["oof_accuracy"])+" | "+pct(r["oof_misclassification"])+" | "+
           pct(r["official_val_accuracy"])+" | "+" | ".join(count(k) for k in ("D_to_0","B_to_8","8_to_B","0_to_D"))+
           " | "+("통과" if r["all_four_directions_under_5pct"] else "미달")+
           " | "+(r["failed_gates"] or "통과")+" |")
    md+=["","## 오버피팅·클래스 오류 및 수렴 증거",
         "| 후보 | 실제0 오류 | 실제8 오류 | 실제B 오류 | 실제D 오류 | 최저 Writer OOF | Seed 표준편차 | 최대 OOF Flip | 최대 OfficialVal Flip | 반복횟수 | 경고 Fold |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append("| "+str(r["candidate"])+" | "+" | ".join(pct(r[f"class_{cl}_error_rate"]) for cl in ("0","8","B","D"))+
          " | "+pct(r["worst_writer_accuracy"])+" | "+pct(r["seed_std"])+
          " | "+pct(r["oof_max_family_flip"])+" | "+pct(r["val_max_family_flip"])+
          " | "+f'{r["mean_iterations"]:.1f}'+" | "+str(r["warning_folds"])+" |")
    md+=["","### 4명의 보류 Writer별 OOF 정확도",
        "| 후보 | W41 | W42 | W43 | W44 | Training loss 초기 평균 | Training loss 마지막 평균 | 평균 Fold 시간(초) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in metrics:
        m=metrics[key]
        md.append("| "+str(key)+" | "+" | ".join(pct(m["writers"][str(w)]) for w in WRITERS)+
          " | "+f'{loss[key]["first"]:.6f}'+" | "+f'{loss[key]["last"]:.6f}'+
          " | "+f'{m["mean_fit_seconds"]:.1f}'+" |")
    md+=["","### 같은 이미지: 기존 오답 교정과 새 오답",
         "| 후보 | 기존 오답 교정 | 기존 정답 신규 오답 | 순개선 |",
         "|---|---:|---:|---:|"]
    for key,pair in paired_errors.items():
        corr,induced=pair
        md.append("| "+key+" | "+str(corr)+" | "+str(induced)+" | "+str(corr-induced)+" |")
    md+=["","## 채택·탈락의 명시적 근거",
        f"- **이번 단계 채택값: {selected}**.",
        f"- **4방향 각각 5% 미만 및 Gate 통과:** {', '.join(strict) if strict else '없음'}. 목표는 아직 달성하지 못했다면 이를 우회해 성공으로 표기하지 않는다.",
        "- **단계별 후보 판정:**"]
    for r in rows:
        k=r["candidate"];why="비교 기준선" if k==baseline_key else (r["failed_gates"] or "안전 Gate 모두 통과")
        md.append(f"  - **{k}:** {why}. D→0 {r['D_to_0']}건·OOF {pct(r['oof_accuracy'])}·OfficialVal {pct(r['official_val_accuracy'])}.")
    md+=["",("- **다음 S5:** 방금 선택한 LR을 고정하고 Training-only Early Stopping=False/True를 비교. 이 단계의 결과를 보지 않고 ES를 먼저 결정하지 않는다."
        if stage=="S4_learning_rate" else
        "- **S1~S5 이후:** Size30/Stroke28/CoM·(256,)·alpha=1e-4·이번 LR 및 ES 선택값을 잠정 동결. 전체 실험의 개발용 결과로 정리하며 최종 Test는 여전히 미사용."),
        "- **과적합 주의:** 같은 4 Writer를 반복 비교해 선택 편향 가능성이 남는다. Internal Early Stopping과 ConvergenceWarning 무발생만으로 과적합 부재를 증명하지 않는다.",
        "- **데이터 경계:** Original Test·교수님 Test·신규 Hand-made 480장 모두 HP 선택 과정에서 미사용.",
        ""]
    return "\n".join(md)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("s4","s5"),required=True)
    ap.add_argument("--s1-reference",type=Path,required=True)
    ap.add_argument("--s4-new",type=Path)
    ap.add_argument("--s5-new",type=Path)
    ap.add_argument("--s4-decision",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    s1={s:a.s1_reference/f"cmb008-w256-s{s}" for s in SEEDS}
    if a.stage=="s4":
        assert a.s4_new is not None
        groups={}
        for lr in LR_KEYS:
            paths=s1 if lr=="0.001" else {s:a.s4_new/f"cmb008-s4-lr{lr}-s{s}" for s in SEEDS}
            groups[lr]=load_group(paths,lr,"S1_width" if lr=="0.001" else "S4_learning_rate",False)
        verify_sha(groups)
        metrics={key:summarize(key,run) for key,run in groups.items()}
        baseline_key="0.001"
        assert abs(metrics[baseline_key]["oof"]-.9221354166666667)<1e-9
        assert abs(metrics[baseline_key]["val"]-.9704427083333333)<1e-9
        gates,strict,safe,selected=rank(metrics[baseline_key],metrics)
        paired_errors={key:paired(groups[baseline_key],g) for key,g in groups.items() if key!=baseline_key}
        loss={key:{"first":statistics.mean(groups[key][s]["loss_start"] for s in SEEDS),
                   "last":statistics.mean(groups[key][s]["loss_end"] for s in SEEDS)} for key in groups}
        stage="S4_learning_rate"
        tag="S4"
        decision_extra={"selected_learning_rate":selected,"alpha_fixed":.0001,
            "structure_fixed":[256],"early_stopping_fixed":False,
            "new_fits":24,"baseline_reused_models":12}
    else:
        assert a.s4_new is not None and a.s5_new is not None and a.s4_decision is not None
        s4=json.loads(a.s4_decision.read_text(encoding="utf-8"))
        assert s4["stage"]=="S4_learning_rate"
        lr=s4["selected_learning_rate"]
        assert lr in LR_KEYS
        paths=s1 if lr=="0.001" else {s:a.s4_new/f"cmb008-s4-lr{lr}-s{s}" for s in SEEDS}
        groups={"False":load_group(paths,lr,"S1_width" if lr=="0.001" else "S4_learning_rate",False),
                "True":load_group({s:a.s5_new/f"cmb008-s5-lr{lr}-s{s}" for s in SEEDS},lr,"S5_early_stopping",True)}
        verify_sha(groups)
        metrics={key:summarize(key,run) for key,run in groups.items()}
        baseline_key="False"
        expected=s4["scores"][lr]
        assert abs(metrics[baseline_key]["oof"]-expected["oof"])<1e-12, "S5 baseline must match S4 selected archive"
        assert abs(metrics[baseline_key]["val"]-expected["val"])<1e-12
        gates,strict,safe,selected=rank(metrics[baseline_key],metrics)
        paired_errors={"True":paired(groups["False"],groups["True"])}
        loss={key:{"first":statistics.mean(groups[key][s]["loss_start"] for s in SEEDS),
                   "last":statistics.mean(groups[key][s]["loss_end"] for s in SEEDS)} for key in groups}
        stage="S5_early_stopping"
        tag="S5"
        decision_extra={"selected_learning_rate":lr,
                        "selected_early_stopping":selected=="True",
                        "s4_selected_learning_rate_from_verified_decision":True,
                        "internal_training_validation_fraction_if_enabled":.1,
                        "alpha_fixed":.0001,"structure_fixed":[256],
                        "new_fits":12,"baseline_reused_models":12}
    report=format_details(stage,baseline_key,selected,groups,metrics,gates,strict,safe,paired_errors,loss)
    report.update(decision_extra)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/f"{tag}_details.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    rows=build_rows(metrics,gates)
    save_csv(a.out/f"{tag}_comparison.csv",rows)
    text=write_report(stage,baseline_key,selected,rows,metrics,gates,strict,paired_errors,loss,decision_extra)
    (a.out/f"{tag}_Stage_Results.md").write_text(text,encoding="utf-8")
    print(f"{tag}_DECISION",json.dumps({"status":report["status"],"selected":selected,
            "strict":strict,"safe":safe,"rows":rows},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
