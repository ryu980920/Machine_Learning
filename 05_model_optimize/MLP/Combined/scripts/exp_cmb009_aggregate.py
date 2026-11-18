#!/usr/bin/env python3
"""Aggregate exactly paired 12-fold CMB009 runs, gate arms, then assess gated combo."""
from __future__ import annotations
import argparse,csv,json,pathlib,statistics,hashlib
from collections import defaultdict

ARMS=("base","warp","rotate3","rotate5","shear005","shear010")
SEEDS=(42,43,44)
WRITERS=(41,42,43,44)
LABELS="0123456789ABCDEF"
WATCH=("7_to_9","9_to_7","D_to_0","0_to_D","B_to_8","8_to_B")

def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def gather(directory,arms):
    metrics={}
    raw={}
    for arm in arms:
        for seed in SEEDS:
            for held in WRITERS:
                key=(arm,seed,held)
                d=directory/f"cmb009-{arm}-s{seed}-w{held}"
                if not d.exists():raise AssertionError(f"missing folder: {d}")
                m=json.loads((d/"manifest.json").read_text())
                assert (m["arm"],m["seed"],m["heldout"])==key
                assert m["training_samples"]==78720 and m["heldout_samples"]==640
                assert m["official_val_samples"]==19200 and m["model_batch"]==128
                assert not m["external_438_used_in_training_or_selection"]
                assert not m["original_test_used"] and not m["professor_test_used"]
                assert m["frozen"]["early_stopping"] is False
                assert m["sklearn_version"]=="1.8.0"
                assert m["new_train_only_aug"]["ratio"]==0 or abs(m["new_train_only_aug"]["ratio"]-.2)<.002
                assert hashlib.sha256((d/"trained_model.joblib").read_bytes()).hexdigest()==m["model_sha256"]
                with (d/"oof_predictions.csv").open(newline="",encoding="utf-8-sig") as f:
                    pp=list(csv.DictReader(f))
                assert len(pp)==640
                assert [int(x["index"]) for x in pp]==list(range(640))
                assert [x["actual"] for x in pp]==[LABELS[i%16] for i in range(640)]
                metrics[key]=m;raw[key]=pp
    for seed in SEEDS:
        for held in WRITERS:
            official={metrics[(a,seed,held)]["official_sha256"] for a in arms}
            hand={json.dumps(metrics[(a,seed,held)]["writer_sha256"],sort_keys=True) for a in arms}
            assert len(official)==len(hand)==1
    return metrics,raw

def summarize(metrics,arms):
    out={}
    for arm in arms:
        rows=[m for (a,_,_),m in metrics.items() if a==arm]
        def mean(k):
            return sum(r[k]["accuracy"] for r in rows)/len(rows)
        def macro(k):
            return sum(r[k]["macro_f1"] for r in rows)/len(rows)
        oc={}
        for pair in WATCH:
            oc[pair]=sum(r["oof"]["pairs"][pair] for r in rows)
        e={c:sum(r["oof"]["class_errors"][c] for r in rows) for c in LABELS}
        writer={str(w):sum(metrics[(arm,s,w)]["oof"]["accuracy"] for s in SEEDS)/3 for w in WRITERS}
        seeds={str(s):sum(metrics[(arm,s,w)]["oof"]["accuracy"] for w in WRITERS)/4 for s in SEEDS}
        out[arm]={"oof_accuracy":mean("oof"),"oof_macro_f1":macro("oof"),
            "official_val_accuracy":mean("official_val"),
            "official_val_macro_f1":macro("official_val"),
            "writer_oof":writer,"seed_oof":seeds,
            "oof_directional_counts":oc,"oof_class_errors":e,
            "oof_correct_sum":sum(r["oof"]["correct"] for r in rows),
            "oof_trials":7680,"official_val_trials":12*19200,
            "n_models":len(rows),
            "new_train_only_aug_accepted_mean":sum(r["new_train_only_aug"]["accepted"] for r in rows)/len(rows),
            "new_train_only_aug_attempted_mean":sum(r["new_train_only_aug"]["attempted"] for r in rows)/len(rows),
            "avg_epochs":sum(r["n_iter"] for r in rows)/len(rows)}
    return out

def paired(pred,arm):
    improved=broken=0
    for seed in SEEDS:
        for held in WRITERS:
            one=pred[("base",seed,held)]
            two=pred[(arm,seed,held)]
            assert len(one)==len(two)
            for a,b in zip(one,two):
                assert a["actual"]==b["actual"] and a["index"]==b["index"]
                was=a["actual"]==a["pred"];now=b["actual"]==b["pred"]
                improved+=int(not was and now)
                broken+=int(was and not now)
    return {"corrected":improved,"new_errors":broken,"net":improved-broken}

def evaluate(arms,summary,pred):
    ref=summary["base"]
    decisions={}
    for arm in arms:
        if arm=="base":continue
        x=summary[arm]
        val_drop_pp=100*(ref["official_val_accuracy"]-x["official_val_accuracy"])
        acc_gain_pp=100*(x["oof_accuracy"]-ref["oof_accuracy"])
        worst_writer_pp=min(100*(x["writer_oof"][w]-ref["writer_oof"][w]) for w in ref["writer_oof"])
        p=x["oof_directional_counts"];q=ref["oof_directional_counts"]
        focus_growth=sum(p[k]-q[k] for k in ("D_to_0","0_to_D","B_to_8","8_to_B"))
        # Predeclared exploratory gates. Do not treat these as validated independent-test thresholds.
        gates={"oof_gain_at_least_0p1pp":acc_gain_pp>=.10-1e-12,
               "official_val_drop_max_0p3pp":val_drop_pp<=.30+1e-12,
               "writer_worst_drop_max_1pp":worst_writer_pp>=-1.0-1e-12,
               "seven_to_nine_no_increase":p["7_to_9"]<=q["7_to_9"],
               "nine_to_seven_no_increase":p["9_to_7"]<=q["9_to_7"],
               "old_four_directions_net_growth_max_6":focus_growth<=6}
        decisions[arm]={"gates":gates,"pass":all(gates.values()),
          "oof_gain_pp":acc_gain_pp,"val_drop_pp":val_drop_pp,
          "worst_writer_delta_pp":worst_writer_pp,"old_four_count_delta":focus_growth,
          "paired_vs_baseline":paired(pred,arm)}
    return decisions

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--incoming",required=True,type=pathlib.Path)
    a.add_argument("--out",required=True,type=pathlib.Path)
    a.add_argument("--phase",choices=("standalone","combo"),required=True)
    a.add_argument("--selection",type=pathlib.Path)
    x=a.parse_args()
    x.out.mkdir(parents=True,exist_ok=True)
    if x.phase=="standalone":
        metrics,pred=gather(x.incoming,ARMS)
        summary=summarize(metrics,ARMS)
        decision=evaluate(ARMS,summary,pred)
        selected=[]
        for armset in (("warp",),("rotate3","rotate5"),("shear005","shear010")):
            ok=[arm for arm in armset if decision[arm]["pass"]]
            if ok: selected.append(max(ok,key=lambda z:summary[z]["oof_accuracy"]))
        selection={"selected":selected,"criteria":"predeclared exploratory gates","combo_allowed":len(selected)>=2,
            "reason":"Only independently qualified arms combined; same 20 percent official cohort, composed transforms",
            "no_external438_or_original_professor_test_used":True}
        report={"experiment":"EXP-CMB-009","stage":"localwarp_rotation_shear",
            "frozen_batch_size":128,"n_models":72,"summary":summary,"decision":decision,"selection":selection}
        save(x.out/"standalone_summary.json",report)
        save(x.out/"selection.json",selection)
        with (x.out/"standalone_table.csv").open("w",newline="",encoding="utf-8-sig") as f:
            writer=csv.DictWriter(f,fieldnames=["arm","oof_accuracy","official_val_accuracy","oof_macro_f1","7_to_9","9_to_7","D_to_0","0_to_D","B_to_8","8_to_B","paired_corrected","paired_new_errors","paired_net","passes_gates"])
            writer.writeheader()
            for arm in ARMS:
                m=summary[arm];d=decision.get(arm,{})
                writer.writerow({"arm":arm,"oof_accuracy":m["oof_accuracy"],
                    "official_val_accuracy":m["official_val_accuracy"],
                    "oof_macro_f1":m["oof_macro_f1"],
                    **{k:m["oof_directional_counts"][k] for k in WATCH},
                    **{"paired_"+key:value for key,value in d.get("paired_vs_baseline",{}).items()},
                    "passes_gates":d.get("pass",True)})
        for group, names in (("1_Local_Warp",("warp",)),("2_Rotation",("rotate3","rotate5")),("3_Shear",("shear005","shear010"))):
            for name in names:
                s=summary[name];d=decision[name]
                print("STAGE_RESULT",json.dumps({"stage":group,"arm":name,
                     "oof_accuracy":s["oof_accuracy"],"official_val_accuracy":s["official_val_accuracy"],
                     "seven_to_nine":s["oof_directional_counts"]["7_to_9"],
                     "baseline_oof":summary["base"]["oof_accuracy"],
                     "gates":d["gates"],"approved_for_combo":d["pass"]}),flush=True)
        print("COMBO_SELECTION",json.dumps(selection),flush=True)
    else:
        if not x.selection:raise ValueError("must provide selection")
        sel=json.loads(x.selection.read_text())
        if not sel["combo_allowed"]:
            print("STAGE4_NO_COMBO",json.dumps(sel))
            save(x.out/"combination_summary.json",{"skipped":True,"selection":sel})
            return
        metrics,pred=gather(x.incoming,("base","combo"))
        summary=summarize(metrics,("base","combo"))
        decision=evaluate(("base","combo"),summary,pred)
        report={"experiment":"EXP-CMB-009","stage":"gated_combination",
                "selected_operations":sel["selected"],
                "summary":summary,"decision":decision,
                "combo_selected_as_research_candidate":decision["combo"]["pass"],
                "independent_test_used":False}
        save(x.out/"combination_summary.json",report)
        print("STAGE4_COMBO_RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
