#!/usr/bin/env python3
"""EXP-CMB-010 paired 12-run baseline/add480 aggregate; no test-based selection."""
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
ARMS=("base","add480")
SEEDS=(42,43,44)
WRITERS=(41,42,43,44)
LABELS="0123456789ABCDEF"
WATCH=("7_to_9","9_to_7","B_to_8","8_to_B","0_to_D","D_to_0","7_to_A","4_to_F")
def dump(path,o):
    path.write_text(json.dumps(o,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def load(incoming):
    rec,lines={},{}
    for arm in ARMS:
        for seed in SEEDS:
            for writer in WRITERS:
                key=(arm,seed,writer)
                folder=incoming/f"cmb010-{arm}-s{seed}-w{writer}"
                m=json.loads((folder/"manifest.json").read_text())
                assert (m["arm"],m["seed"],m["heldout"])==key
                assert m["training_samples"]==(78720+(480 if arm=="add480" else 0))
                assert m["heldout_samples"]==640 and m["official_val_samples"]==19200
                assert m["frozen"]["batch_size"]==128
                assert not m["heldout_in_fit"] and not m["external_438_used"]
                assert not m["original_test_used"] and not m["professor_test_used"]
                assert m["added_source_sha256"]=="3d1686c67d7bb5c6193384eca1aa9218213c842252baa5b7e172698743b83fc7"
                assert hashlib.sha256((folder/"candidate_model.joblib").read_bytes()).hexdigest()==m["model_sha256"]
                with (folder/"heldout_predictions.csv").open(newline="",encoding="utf-8-sig") as f:
                    rows=list(csv.DictReader(f))
                assert len(rows)==640 and [int(r["index"]) for r in rows]==list(range(640))
                rec[key]=m;lines[key]=rows
    for seed in SEEDS:
        for w in WRITERS:
            m0,m1=rec[("base",seed,w)],rec[("add480",seed,w)]
            assert m0["official_sha256"]==m1["official_sha256"]
            assert m0["old_writers_sha256"]==m1["old_writers_sha256"]
            assert [r["actual"] for r in lines[("base",seed,w)]]==[r["actual"] for r in lines[("add480",seed,w)]]
    return rec,lines
def summarize(rec):
    result={}
    for arm in ARMS:
        rows=[v for k,v in rec.items() if k[0]==arm]
        def avg(data,key):
            return sum(x[data][key] for x in rows)/len(rows)
        result[arm]={"n_models":len(rows),"training_samples_per_model":rows[0]["training_samples"],
            "oof_accuracy":avg("oof","accuracy"),
            "oof_macro_f1":avg("oof","macro_f1"),
            "official_val_accuracy":avg("official_val","accuracy"),
            "official_val_macro_f1":avg("official_val","macro_f1"),
            "writer_accuracy":{str(w):sum(rec[(arm,s,w)]["oof"]["accuracy"] for s in SEEDS)/3 for w in WRITERS},
            "seed_accuracy":{str(s):sum(rec[(arm,s,w)]["oof"]["accuracy"] for w in WRITERS)/4 for s in SEEDS},
            "pairs":{p:sum(x["oof"]["pairs"][p] for x in rows) for p in WATCH},
            "class_errors":{c:sum(x["oof"]["class_errors"][c] for x in rows) for c in LABELS},
            "class_support":{c:sum(x["oof"]["class_support"][c] for x in rows) for c in LABELS},
            "total_oof_correct":sum(x["oof"]["correct"] for x in rows),
            "oof_evaluation_trials":len(rows)*640,
            "n_unique_oof_images":4*640,
            "n_unique_official_val_images":19200}
    return result

def paired(rows):
    corr=reg=0
    class_corr=defaultdict(int);class_reg=defaultdict(int);direction=defaultdict(int)
    for seed in SEEDS:
        for writer in WRITERS:
            x0=rows[("base",seed,writer)]
            x1=rows[("add480",seed,writer)]
            for a,b in zip(x0,x1):
                assert (a["index"],a["actual"])==(b["index"],b["actual"])
                label=a["actual"]
                old=a["pred"]==label; new=b["pred"]==label
                if not old and new:corr+=1;class_corr[label]+=1
                if old and not new:reg+=1;class_reg[label]+=1
                if a["pred"]!=b["pred"]:direction[f"{a['pred']}->{b['pred']}"]+=1
    return {"corrected":corr,"new_errors":reg,"net_correct":corr-reg,
            "corrected_by_actual_class":dict(class_corr),
            "new_errors_by_actual_class":dict(class_reg)}

def decision(s):
    a,b=s["base"],s["add480"]
    tests={"oof_accuracy_non_decrease":b["oof_accuracy"]>=a["oof_accuracy"],
           "official_val_drop_not_more_than_0p3pp":(a["official_val_accuracy"]-b["official_val_accuracy"])<=.003,
           "7_to_9_decrease":b["pairs"]["7_to_9"]<a["pairs"]["7_to_9"],
           "9_to_7_nonincrease":b["pairs"]["9_to_7"]<=a["pairs"]["9_to_7"],
           "B_to_8_nonincrease":b["pairs"]["B_to_8"]<=a["pairs"]["B_to_8"],
           "8_to_B_nonincrease":b["pairs"]["8_to_B"]<=a["pairs"]["8_to_B"],
           "D_to_0_nonincrease":b["pairs"]["D_to_0"]<=a["pairs"]["D_to_0"],
           "0_to_D_nonincrease":b["pairs"]["0_to_D"]<=a["pairs"]["0_to_D"]}
    return {"exploratory_predeclared_checks":tests,"pass_all":all(tests.values()),
            "oof_gain_pp":100*(b["oof_accuracy"]-a["oof_accuracy"]),
            "official_val_gain_pp":100*(b["official_val_accuracy"]-a["official_val_accuracy"]),
            "limitations":["Single unknown-writer cohort; cannot prove distinct from writers41-44",
                           "Existing OOF writers reused repeatedly; development-only, not independent",
                           "Class-targeted data additions can alter prior probabilities",
                           "No untouched independent 16-class writer-test evaluation"],
            "deployment_decision":"No automatic deployment or main merge"}
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--incoming",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    z=a.parse_args()
    z.out.mkdir(parents=True,exist_ok=True)
    rec,pred=load(z.incoming)
    s=summarize(rec);delta=paired(pred);gate=decision(s)
    report={"experiment":"EXP-CMB-010","source_images_by_class":{"7":160,"8":80,"9":160,"B":80},
      "source_sha256":"3d1686c67d7bb5c6193384eca1aa9218213c842252baa5b7e172698743b83fc7",
      "new_cohort_writer_id_known":False,
      "n_fits":24,"arms":s,"paired":delta,"preliminary_gates":gate}
    dump(z.out/"summary.json",report)
    fields=["arm","oof_accuracy","official_val_accuracy","oof_macro_f1","official_val_macro_f1",
            *WATCH,"n_models","training_samples_per_model"]
    with (z.out/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for arm in ARMS:
            row=s[arm]
            w.writerow({"arm":arm,**{k:row[k] for k in fields if k in row},
                        **row["pairs"]})
    with (z.out/"class_error_comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["class","support_total","base_errors","add480_errors",
            "delta_errors","base_error_rate","add480_error_rate"]);w.writeheader()
        for c in LABELS:
            denom=s["base"]["class_support"][c]
            assert denom==s["add480"]["class_support"][c]
            x=s["base"]["class_errors"][c];y=s["add480"]["class_errors"][c]
            w.writerow({"class":c,"support_total":denom,"base_errors":x,"add480_errors":y,
                "delta_errors":y-x,"base_error_rate":x/denom,"add480_error_rate":y/denom})
    print("CMB010_AGGREGATE",json.dumps({"base_oof":s["base"]["oof_accuracy"],
      "new_oof":s["add480"]["oof_accuracy"],
      "base_val":s["base"]["official_val_accuracy"],
      "new_val":s["add480"]["official_val_accuracy"],
      "paired":delta,"gates":gate["exploratory_predeclared_checks"]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
