#!/usr/bin/env python3
"""CMB011: 7-arm writer-heldout paired audit, class-balanced controls and P/R/F1."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
ARMS=("base","add79","addB8","add480","dup79","dupB8","dup480")
COUNT={"base":0,"add79":320,"addB8":160,"add480":480,"dup79":320,"dupB8":160,"dup480":480}
SEEDS=(42,43,44)
WRITERS=(41,42,43,44)
LABELS="0123456789ABCDEF"
PAIRS=((7,9,"7_to_9"),(9,7,"9_to_7"),(11,8,"B_to_8"),(8,11,"8_to_B"),(0,13,"0_to_D"),(13,0,"D_to_0"),(7,10,"7_to_A"),(4,15,"4_to_F"))
SOURCE_SHA="3d1686c67d7bb5c6193384eca1aa9218213c842252baa5b7e172698743b83fc7"
def writecsv(p,rows):
    assert rows
    with p.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def readall(source):
    m,pp={},{}
    for arm in ARMS:
        for seed in SEEDS:
            for writer in WRITERS:
                k=(arm,seed,writer)
                folder=source/f"cmb011-{arm}-s{seed}-w{writer}"
                man=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
                assert (man["arm"],int(man["seed"]),int(man["heldout"]))==k
                assert man["experiment"]=="EXP-CMB-011"
                assert man["training_samples"]==78720+COUNT[arm]
                assert man["added_source_sha256"]==SOURCE_SHA
                assert man["heldout_samples"]==640 and man["official_val_samples"]==19200
                assert man["frozen"]["batch_size"]==128
                assert not man["heldout_in_fit"] and not man["external_438_used"]
                assert not man["original_test_used"] and not man["professor_test_used"]
                assert man["new_cohort_in_fit"]==arm.startswith("add")
                assert man["added_images_count"]==(COUNT[arm] if arm.startswith("add") else 0)
                assert man["duplicated_training_rows_count"]==(COUNT[arm] if arm.startswith("dup") else 0)
                with (folder/"heldout_predictions.csv").open(newline="",encoding="utf-8-sig") as f:
                    rows=list(csv.DictReader(f))
                assert len(rows)==640 and [int(v["index"]) for v in rows]==list(range(640))
                assert all((v["arm"],int(v["seed"]),int(v["heldout"]))==k for v in rows)
                actual=np.asarray([v["actual"] for v in rows])
                pred=np.asarray([v["pred"] for v in rows])
                assert np.isclose((actual==pred).mean(),man["oof"]["accuracy"])
                m[k]=man;pp[k]=(actual,pred)
    for seed in SEEDS:
        for writer in WRITERS:
            original=pp[("base",seed,writer)][0]
            for arm in ARMS[1:]:
                assert np.array_equal(original,pp[(arm,seed,writer)][0])
                assert m[("base",seed,writer)]["official_sha256"]==m[(arm,seed,writer)]["official_sha256"]
                assert m[("base",seed,writer)]["old_writers_sha256"]==m[(arm,seed,writer)]["old_writers_sha256"]
    return m,pp
def compute(m,pp):
    allsum={};classrows=[];foldrows=[]
    for arm in ARMS:
        mats=[]; foldmetrics=[]
        for seed in SEEDS:
            for writer in WRITERS:
                actual,pred=pp[(arm,seed,writer)]
                pr,re,f1,_=precision_recall_fscore_support(actual,pred,labels=list(LABELS),zero_division=0)
                foldmetrics.append([float(pr.mean()),float(re.mean()),float(f1.mean())])
                foldrows.append({"arm":arm,"seed":seed,"heldout_writer":writer,
                                 "accuracy":float((actual==pred).mean()),
                                 "macro_precision":float(pr.mean()),
                                 "macro_recall":float(re.mean()),"macro_f1":float(f1.mean()),
                                 "official_val_accuracy":m[(arm,seed,writer)]["official_val"]["accuracy"],
                                 "official_val_macro_f1":m[(arm,seed,writer)]["official_val"]["macro_f1"],
                                 "fit_seconds":m[(arm,seed,writer)]["fit_seconds"],
                                 "n_iter":m[(arm,seed,writer)]["n_iter"]})
                mats.append(confusion_matrix(actual,pred,labels=list(LABELS)))
        cm=np.sum(mats,axis=0)
        assert cm.sum()==7680
        pre,rec,f,_=precision_recall_fscore_support(
            np.repeat(np.arange(16),cm.sum(axis=1)),
            np.concatenate([np.repeat(j,cm[i,j]) for i in range(16) for j in range(16)]),
            labels=np.arange(16),zero_division=0) if False else (None,None,None,None)
        p_cls=np.diag(cm)/np.maximum(cm.sum(axis=0),1)
        r_cls=np.diag(cm)/np.maximum(cm.sum(axis=1),1)
        f_cls=np.divide(2*p_cls*r_cls,p_cls+r_cls,out=np.zeros_like(p_cls,dtype=float),where=(p_cls+r_cls)>0)
        for i,cl in enumerate(LABELS):
            classrows.append({"arm":arm,"class":cl,"support":int(cm[i].sum()),
                "precision":float(p_cls[i]),"recall":float(r_cls[i]),"f1":float(f_cls[i]),
                "errors":int(cm[i].sum()-cm[i,i])})
        foldarr=np.asarray(foldmetrics)
        av=lambda key:sum(m[(arm,s,w)][key]["accuracy"] for s in SEEDS for w in WRITERS)/12
        avgvalf1=sum(m[(arm,s,w)]["official_val"]["macro_f1"] for s in SEEDS for w in WRITERS)/12
        total_correct=int(np.trace(cm))
        allsum[arm]={"n_models":12,"train_rows":78720+COUNT[arm],
          "new_handmade_rows":COUNT[arm] if arm.startswith("add") else 0,
          "duplicated_official_rows":COUNT[arm] if arm.startswith("dup") else 0,
          "oof_accuracy":float(total_correct/7680),
          "macro_precision":float(foldarr[:,0].mean()),
          "macro_recall":float(foldarr[:,1].mean()),
          "macro_f1":float(foldarr[:,2].mean()),
          "official_val_accuracy":float(av("official_val")),
          "official_val_macro_f1":float(avgvalf1),
          "correct":total_correct,
          "class_errors":{LABELS[i]:int(cm[i].sum()-cm[i,i]) for i in range(16)},
          "class_f1":{LABELS[i]:float(f_cls[i]) for i in range(16)},
          "pairs":{name:int(cm[a,b]) for a,b,name in PAIRS},
          "writer_accuracy":{str(w):float(np.mean([m[(arm,s,w)]["oof"]["accuracy"] for s in SEEDS])) for w in WRITERS},
          "seed_accuracy":{str(s):float(np.mean([m[(arm,s,w)]["oof"]["accuracy"] for w in WRITERS])) for s in SEEDS},
          "confusion_16class":cm.tolist()}
    paired={}
    for arm in ARMS[1:]:
        corrected=regress=0
        byclass={c:{"corrected":0,"new_errors":0} for c in LABELS}
        for seed in SEEDS:
            for w in WRITERS:
                actual,b=pp[("base",seed,w)]
                a2,p=pp[(arm,seed,w)]
                assert np.array_equal(actual,a2)
                fix=(b!=actual)&(p==actual)
                broken=(b==actual)&(p!=actual)
                corrected+=int(fix.sum());regress+=int(broken.sum())
                for c in LABELS:
                    t=actual==c
                    byclass[c]["corrected"]+=int((t&fix).sum())
                    byclass[c]["new_errors"]+=int((t&broken).sum())
        paired[arm]={"corrected":corrected,"new_errors":regress,
                     "net_correct":corrected-regress,"per_class":byclass,
                     "oof_delta_pp":100*(allsum[arm]["oof_accuracy"]-allsum["base"]["oof_accuracy"]),
                     "official_val_delta_pp":100*(allsum[arm]["official_val_accuracy"]-allsum["base"]["official_val_accuracy"])}
    return allsum,paired,foldrows,classrows
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--incoming",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    m,pp=readall(a.incoming)
    summary,paired,folds,classes=compute(m,pp)
    result={"experiment":"EXP-CMB-011",
      "study":"class partition and same-count official-train duplication controls",
      "caveats":["Existing Official Train duplication is sample reweighting, not extra unique data",
                 "Original new writer IDs unknown; 480 grouped as one cohort",
                 "Repeated OOF writer set used for selection in past; not independent test",
                 "No Original Test, Professor Test, external 438 used for training or selection"],
      "n_models":len(m),"n_unique_oof_images":2560,"oof_trials_per_arm":7680,
      "arms":summary,"paired_vs_baseline":paired}
    (a.out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    names=("arm","train_rows","new_handmade_rows","duplicated_official_rows","oof_accuracy","macro_precision","macro_recall","macro_f1","official_val_accuracy","official_val_macro_f1","correct")
    writecsv(a.out/"summary.csv",[{"arm":arm,**{q:summary[arm][q] for q in names if q!="arm"}} for arm in ARMS])
    writecsv(a.out/"per_class.csv",classes)
    writecsv(a.out/"per_fold.csv",folds)
    writecsv(a.out/"paired.csv",[{"arm":arm,**{k:v for k,v in paired[arm].items() if k!="per_class"}} for arm in ARMS[1:]])
    rows=[]
    for arm in ARMS:
        for k,v in summary[arm]["pairs"].items():
            rows.append({"arm":arm,"direction":k,"errors":v})
    writecsv(a.out/"focus_pairs.csv",rows)
    print("CMB011_FINISHED",json.dumps({"n_models":len(m),"arms":{k:{"oof":v["oof_accuracy"],"macro_f1":v["macro_f1"],"val":v["official_val_accuracy"],"pairs":v["pairs"]} for k,v in summary.items()},"paired":{k:{"net_correct":v["net_correct"]} for k,v in paired.items()}},ensure_ascii=False),flush=True)
if __name__=="__main__":
    main()
