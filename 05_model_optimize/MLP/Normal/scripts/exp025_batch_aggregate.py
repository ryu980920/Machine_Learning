#!/usr/bin/env python3
"""EXP025: audit 3-seed OfficialVal/safety gate, then existing Handmade dev set.
No model training, no tuning against Handmade, no Original/Professor Test.
"""
from __future__ import annotations
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="1"
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np,joblib
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from exp006_size_com_comparison import ROOT,load_writers,process,LABELS,WRITERS
from exp011_012_causal_probes import fixed_scale

SEEDS=(42,43,44);BATCHES=(32,64,128,200,256);SCALES=(.8,1.,1.2)
BASE_SHA={42:"e08eab163cc2db86b03338185f1b0b4f1416a074572597f2b692adec85a0d190",43:"f47e4cc2a97af98e18e5bfdca9b2621acac0f9c303bcd1458fe270fb58e9f931",44:"120cdf9f48691d3e75016379fb62a3cbf3f66d57ffcb7c4c59902a9639795413"}

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(p):
 with open(p,encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def save(p,rs):
 assert rs,p
 with open(p,"w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def avg(rows,batch,scale,group,field):
 v=[float(r[field]) for r in rows if int(r["batch"])==batch and float(r["scale"])==scale and r["group"]==group]
 assert len(v)==3,(batch,scale,group,field,len(v))
 return float(np.mean(v))
def main():
 p=argparse.ArgumentParser();p.add_argument("--inputs",type=Path,required=True);p.add_argument("--frozen",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
 a=p.parse_args();out=a.out
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 models={};official=[];train_summary=[]
 for seed in SEEDS:
  folder=a.frozen/f"exp017-fine-s{seed}-slot0"
  model=folder/"slot1_alpha_up.joblib"
  assert model.is_file() and sha(model)==BASE_SHA[seed],model
  models[200,seed]=model
  hist=[r for r in read(folder/"metrics.csv") if r["variant"]=="slot1_alpha_up"]
  assert len(hist)==57,(seed,len(hist))
  for r in hist:
   official.append({"batch":200,"seed":seed,"scale":float(r["scale"]),"group":r["group"],"n":int(r["n"]),
    "accuracy":float(r["accuracy"]),"family_flip":float(r["family_flip"]),"abs_delta_p_letter":float(r["abs_delta_p_letter"])})
  rec=next(r for r in read(folder/"training_records.csv") if r["variant"]=="slot1_alpha_up")
  train_summary.append({"batch":200,"seed":seed,"n_iter":int(rec["n_iter"]),"sha256":sha(model),"origin":"EXP017 frozen"})
 for batch in (32,64,128,256):
  for seed in SEEDS:
   d=a.inputs/f"EXP025-bs{batch}-s{seed}"
   assert d.is_dir(),d
   m=json.loads((d/"manifest.json").read_text())
   model=d/"model.joblib"
   assert model.is_file() and sha(model)==m["model_sha256"]
   assert m["batch_size"]==batch and m["seed"]==seed and m["train_n"]==76800 and m["augmented_n"]==61440
   assert not m["handmade_used_in_training_or_selection"] and not m["original_test_used"] and not m["professor_test_used"]
   models[batch,seed]=model
   for r in read(d/"official_metrics.csv"):
    official.append({"batch":batch,"seed":seed,"scale":float(r["scale"]),"group":r["group"],"n":int(r["n"]),
     "accuracy":float(r["accuracy"]),"family_flip":float(r["family_flip"]),"abs_delta_p_letter":float(r["abs_delta_p_letter"])})
   train_summary.append({"batch":batch,"seed":seed,"n_iter":m["n_iter"],"sha256":m["model_sha256"],"origin":"EXP025"})
 assert len(models)==15
 save(out/"officialval_seed_scale.csv",official);save(out/"training_summary.csv",train_summary)
 gates=[]
 for batch in BATCHES:
  acc=avg(official,batch,1.,"all","accuracy")
  dig=avg(official,batch,1.,"digits","accuracy");let=avg(official,batch,1.,"letters","accuracy")
  bd=max(avg(official,batch,s,"class_"+c,"family_flip") for s in (.8,1.2) for c in "B8D")
  worst_seed=max(float(r["family_flip"]) for r in official if r["batch"]==batch and r["group"] in ("class_B","class_8","class_D") and r["scale"] in (.8,1.2))
  worst_all=max(avg(official,batch,s,"class_"+c,"family_flip") for s in (.8,1.2) for c in LABELS)
  du=avg(official,batch,1.2,"digits","family_flip");ld=avg(official,batch,.8,"letters","family_flip")
  dp=avg(official,batch,1.2,"digits","abs_delta_p_letter");lp=avg(official,batch,.8,"letters","abs_delta_p_letter")
  acc_pass=acc>=.9653645833333333-1e-12
  shortcut=(bd<=.2 and worst_seed<=.3 and worst_all<=.25 and du<=.1 and ld<=.1 and dp<=.12 and lp<=.12 and dig>=.85 and let>=.85)
  gates.append({"batch":batch,"official_acc":acc,"official_acc_pass":acc_pass,"official_digits":dig,"official_letters":let,
   "B8D_worst_mean_flip":bd,"B8D_worst_seed_flip":worst_seed,"max_class_flip":worst_all,
   "digits_up_flip":du,"letters_down_flip":ld,"digits_delta_p":dp,"letters_delta_p":lp,
   "shortcut_pass":shortcut,"official_shortcut_pass":bool(acc_pass and shortcut)})
  print("EXP025_OFFICIAL",batch,round(100*acc,5),"acc",acc_pass,"shortcut",shortcut,flush=True)
 save(out/"official_gate.csv",gates)
 # Explicitly enter fixed-model Handmade audit only AFTER OfficialVal gates were computed.
 hm=load_writers(ROOT/"02_data"/"Hand_made")
 X=np.concatenate([hm[w]["X"] for w in WRITERS],axis=0)
 y=np.concatenate([hm[w]["y"] for w in WRITERS],axis=0)
 writer=np.repeat(np.asarray(WRITERS),640)
 assert X.shape==(2560,784) and np.array_equal(np.bincount(y,minlength=16),np.full(16,160))
 scales={s:process(X if s==1. else fixed_scale(X,s),"com",20) for s in SCALES}
 groups=[];classes=[];cms=[];pairs=[];scalerows=[]
 for batch in BATCHES:
  for seed in SEEDS:
   mdl=joblib.load(models[batch,seed])
   probs={s:np.asarray(mdl.predict_proba(scales[s]),dtype=np.float64) for s in SCALES}
   pred={s:probs[s].argmax(axis=1) for s in SCALES}
   ple={s:probs[s][:,10:].sum(axis=1) for s in SCALES}
   yhat=pred[1.]
   masks=[("all",np.ones(2560,bool)),("digits",y<10),("letters",y>=10)]+[(f"writer_{w}",writer==w) for w in WRITERS]
   for name,mask in masks:
    groups.append({"batch":batch,"seed":seed,"group":name,"n":int(mask.sum()),"correct":int((yhat[mask]==y[mask]).sum()),"accuracy":float(np.mean(yhat[mask]==y[mask]))})
   cm=confusion_matrix(y,yhat,labels=np.arange(16))
   precis,recall,f1,support=precision_recall_fscore_support(y,yhat,labels=np.arange(16),zero_division=0)
   for i,c in enumerate(LABELS):
    classes.append({"batch":batch,"seed":seed,"class":c,"n":int(support[i]),"recall":float(recall[i]),"precision":float(precis[i]),"f1":float(f1[i])})
    for j,d in enumerate(LABELS):cms.append({"batch":batch,"seed":seed,"actual":c,"predicted":d,"count":int(cm[i,j])})
   for c,d in (("B","8"),("8","B"),("D","0"),("0","D"),("7","9"),("9","7"),("F","5"),("5","F")):
    pairs.append({"batch":batch,"seed":seed,"actual":c,"predicted":d,"count":int(cm[LABELS.index(c),LABELS.index(d)])})
   for s in SCALES:
    for c in "B8D":
     mask=y==LABELS.index(c)
     scalerows.append({"batch":batch,"seed":seed,"scale":s,"class":c,"recall":float(np.mean(pred[s][mask]==y[mask])),
      "family_flip":float(np.mean((ple[s][mask]>=.5)!=(ple[1.][mask]>=.5)))})
   print("EXP025_HANDMADE",batch,seed,round(100*float(np.mean(yhat==y)),5),flush=True)
 save(out/"handmade_groups.csv",groups);save(out/"handmade_per_class.csv",classes)
 save(out/"handmade_confusion.csv",cms);save(out/"handmade_pairs.csv",pairs);save(out/"handmade_B8D_scales.csv",scalerows)
 for g in gates:
  batch=g["batch"]
  sub=[r["accuracy"] for r in groups if r["batch"]==batch and r["group"]=="all"]
  assert len(sub)==3
  g["handmade_acc"]=float(np.mean(sub))
  g["handmade_pass"]=g["handmade_acc"]>=.920703125-1e-12
  for k in ("digits","letters"):
   g[f"handmade_{k}"]=float(np.mean([r["accuracy"] for r in groups if r["batch"]==batch and r["group"]==k]))
  g["three_way_pass"]=bool(g["official_shortcut_pass"] and g["handmade_pass"])
  print("EXP025_GATE",batch,g["official_acc"],g["handmade_acc"],g["three_way_pass"],flush=True)
 save(out/"EXP025_FINAL_GATE.csv",gates)
 passed=[g for g in gates if g["batch"]!=200 and g["three_way_pass"]]
 lines=["# EXP025 Batch Size study", "",
  "Study fixed: EXP017 80% Size + 14% Stroke + CoM, (256,128), alpha=0.03, LR=0.001, ReLU/Adam, 3 seeds.",
  "Baseline EXP017 frozen batch=200. New batch=32/64/128/256, fixed sklearn fit stopping rule.",
  "Selection order: OfficialVal >=96.5365%; original EXP017 Shortcut gates; then reused development Handmade2560 >=92.0703%.",
  "Handmade is NOT an independent Test. Original Test and professor Test are not accessed.","",
  "| Batch | OfficialVal | B8D worst Flip | Shortcut Gate | Handmade | Final 3-way Gate |",
  "|---:|---:|---:|---|---:|---|"]
 for g in gates:
  lines.append(f"| {g['batch']} | {g['official_acc']*100:.4f}% | {g['B8D_worst_mean_flip']*100:.2f}% | {'PASS' if g['shortcut_pass'] else 'FAIL'} | {g['handmade_acc']*100:.4f}% | {'PASS' if g['three_way_pass'] else 'FAIL'} |")
 lines+=["","## Decision"]
 if passed:
  winner=sorted(passed,key=lambda g:(-g["official_acc"],g["B8D_worst_mean_flip"]))[0]
  lines.append(f"Batch {winner['batch']} passes all three gates, pending independent verification. EXP017 remains frozen.")
 else:
  lines.append("No new batch size passes all 3 gates; retain EXP017 frozen model.")
 lines+=["","## Caveat","Batch changes update counts per epoch; this is the total effect under the same sklearn fit policy, not equal-update-budget analysis.",
  "Handmade2560 has been repeatedly consulted in prior development; this diagnostic must not be used for continuous hyperparameter tuning.",
  "All new models are held in EXP025 Actions artifacts; GitHub main and EXP017 frozen weights are unchanged."]
 (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 (out/"AUDIT.json").write_text(json.dumps({"total_models":15,"new_models":12,"source_SHA256":"0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879",
  "handmade_SHA256":{str(w):hm[w]["sha256"] for w in WRITERS},"frozen_exp017_SHA256":BASE_SHA,
  "handmade_training_or_tuning":False,"original_test_used":False,"professor_test_used":False},indent=2),encoding="utf-8")
 print("EXP025_COMPLETE",flush=True)
if __name__=="__main__":main()
