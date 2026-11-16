#!/usr/bin/env python3
"""EXP-CMB-004 post hoc 4-way input/weights control for explanation.

For each writer-held-out fold and normalized method:
  A = original held-out image + original frozen Combined Baseline MLP
  B = transformed held-out image + SAME original frozen MLP
  C = SAME transformed held-out image + new MLP trained with normalization
  D = original held-out image + new normalized MLP (OOD diagnostic only)
The critical two comparisons are A->B (pure input intervention) and B->C
(same input, different training). No new training, no Test, no tuning.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

ROOT=Path(__file__).resolve().parents[4]
CMB=ROOT/"05_model_optimize"/"MLP"/"Combined"
sys.path.insert(0,str(CMB/"scripts"))
import exp_cmb002_evidence as core
import exp_cmb004_geometry_normalization as stage

OUT=CMB/"results"/"Geometry_Normalization"
METHODS=stage.METHODS
LABELS=stage.LABELS
WRITERS=stage.WRITERS
RIVAL={"0":"D","D":"0","8":"B","B":"8"}

def read(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def output(path,rows):
    rows=list(rows)
    assert rows
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

def load_new_models(method):
    manifest=json.loads((OUT/method/"run_manifest.json").read_text(encoding="utf-8"))
    result={}
    with zipfile.ZipFile(OUT/method/"trained_models.zip") as z:
        for w in WRITERS:
            fname=f"{method}_heldout_{w}_mlp.joblib"
            data=z.read(fname)
            assert hashlib.sha256(data).hexdigest()==manifest["trained_models_sha256"][fname]
            result[w]=joblib.load(io.BytesIO(data))
    return result

def score(y,p,classes=None):
    classes=np.arange(16) if classes is None else classes
    pr,re,f1,s=precision_recall_fscore_support(y,p,labels=classes,zero_division=0)
    return {"n":len(y),"correct":int(np.sum(y==p)),
            "accuracy":float(np.mean(y==p)),
            "macro_f1":float(np.mean(f1)),
            "macro_precision":float(np.mean(pr)),
            "macro_recall":float(np.mean(re))}

def main():
    manifest,pred,table,hand,labels=core.load_inputs()
    frozen=core.load_models(manifest)
    goal=json.loads((OUT/"AGGREGATE_MANIFEST.json").read_text(encoding="utf-8"))["size_target"]
    assert goal==20
    all_rows=[];focus=[];pergroup=[];bywriter=[];pair_counts=[]
    summary=[]
    for method in METHODS:
        trained=load_new_models(method)
        target=read(OUT/method/"paired_predictions.csv")
        assert len(target)==2560
        method_p={}
        idx=0
        for writer in WRITERS:
            raw=np.asarray(hand[writer],dtype=np.float32).reshape(640,784)
            y=np.asarray(labels[writer],dtype=int)
            transform,scales,_=stage.apply_method(raw,method,goal)
            old=frozen[f"combined_heldout_{writer}_mlp.joblib"]
            new=trained[writer]
            a=old.predict(raw).astype(int)
            b=old.predict(transform).astype(int)
            c=new.predict(transform).astype(int)
            d=new.predict(raw).astype(int)
            for i in range(640):
                r=table[writer,i]
                assert LABELS[a[i]]==r["combined_pred"],(writer,i,"baseline")
                q=target[idx];idx+=1
                assert (q["writer"],q["index_in_writer"],q["actual"],q["prediction"])==(
                    str(writer),str(i),LABELS[y[i]],LABELS[c[i]]),(method,writer,i)
            method_p[writer]={"A":a,"B":b,"C":c,"D":d,"y":y}
            bywriter.extend({"method":method,"writer":writer,"variant":variant,**score(y,p)}
                for variant,p in (("A_baseline_raw",a),("B_baseline_transformed",b),
                                  ("C_retrained_transformed",c),("D_retrained_raw",d)))
            for i in range(640):
                t=LABELS[y[i]]
                all_rows.append({"method":method,"writer":writer,"index_in_writer":i,
                 "true_label":t,"A":LABELS[a[i]],"B":LABELS[b[i]],
                 "C":LABELS[c[i]],"D":LABELS[d[i]],
                 "A_correct":int(a[i]==y[i]),"B_correct":int(b[i]==y[i]),
                 "C_correct":int(c[i]==y[i]),"D_correct":int(d[i]==y[i])})
            for actual,rival in stage.PAIRS:
                subset=y==LABELS.index(actual)
                for variant,p in (("A_baseline_raw",a),("B_baseline_transformed",b),
                                  ("C_retrained_transformed",c),("D_retrained_raw",d)):
                    pair_counts.append({"method":method,"writer":writer,
                     "actual":actual,"rival":rival,"variant":variant,
                     "errors":int(np.sum(p[subset]==LABELS.index(rival)))})
            # True-vs-fixed-rival logits for 4-WAY on every 0/D/8/B image.
            outs={"A":core.logits(old,raw),"B":core.logits(old,transform),
                  "C":core.logits(new,transform),"D":core.logits(new,raw)}
            for i in range(640):
                actual=LABELS[y[i]]
                if actual not in RIVAL:continue
                other=RIVAL[actual]
                a_i,b_i,c_i,d_i=(float(outs[v][i,y[i]]-
                            outs[v][i,LABELS.index(other)]) for v in ("A","B","C","D"))
                focus.append({"method":method,"writer":writer,"index_in_writer":i,
                    "actual":actual,"rival":other,
                    "baseline_correct":int(a[i]==y[i]),
                    "input_only_correct":int(b[i]==y[i]),
                    "retrained_correct":int(c[i]==y[i]),
                    "original_A_margin":a_i,
                    "frozen_B_margin":b_i,
                    "retrained_C_margin":c_i,
                    "OOD_retrained_D_margin":d_i,
                    "input_only_margin_delta_B_minus_A":b_i-a_i,
                    "weights_on_identical_input_margin_delta_C_minus_B":c_i-b_i,
                    "size_ratio":float(scales[i])})
        assert idx==2560
        for variant in ("A","B","C","D"):
            y=np.concatenate([method_p[w]["y"] for w in WRITERS])
            p=np.concatenate([method_p[w][variant] for w in WRITERS])
            s=score(y,p)
            summary.append({"method":method,"variant":{
              "A":"A_baseline_raw","B":"B_baseline_transformed",
              "C":"C_retrained_transformed","D":"D_retrained_raw"}[variant],**s})
            for group,mask in (("digits",y<10),("letters",y>=10)):
                sc=score(y[mask],p[mask],
                    np.arange(10) if group=="digits" else np.arange(10,16))
                pergroup.append({"method":method,"variant":variant,"group":group,**sc})

    output(OUT/"cross_4way_summary.csv",summary)
    output(OUT/"cross_4way_writers.csv",bywriter)
    output(OUT/"cross_4way_digit_letter.csv",pergroup)
    output(OUT/"cross_4way_pair_errors.csv",pair_counts)
    output(OUT/"cross_4way_paired_predictions.csv",all_rows)
    output(OUT/"cross_4way_focus_margins.csv",focus)

    lines=["# EXP-CMB-004 — 학습 효과/입력 변형 효과 분리 대조",
        "",
        "**A** 기존 입력+기존 MLP / **B** 전처리 입력+동일한 기존 MLP",
        "**C** B와 정확히 같은 전처리 입력+해당 방법으로 새로 학습한 MLP / **D** 원본 입력+새 MLP(훈련 분포 밖 대조)",
        "",
        "A→B는 가중치를 고정한 입력 조작의 효과, B→C는 변형 입력을 고정한 재학습 효과를 분리하여 비교한다.",
        "D는 배포용 Pipeline이 아니며 OOD diagnostics로만 읽어야 한다.",
        "",
        "| Method | A old/raw | B old/transformed | C new/transformed | D new/raw (OOD) | Input only B−A | Model adaptation C−B |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        ss=[s for s in summary if s["method"]==method]
        a,b,c,d=[100*s["accuracy"] for s in ss]
        lines.append("| {} | {:.3f}% | {:.3f}% | {:.3f}% | {:.3f}% | {:+.3f}pp | {:+.3f}pp |".format(
            method,a,b,c,d,b-a,c-b))
    lines+=["",
        "### 그룹별 대비와 부작용",
        "",
        "| Method | Group | A old/raw | B old/trans | C new/trans | B−A | C−B |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        for group in ("digits","letters"):
            ss=[s for s in pergroup if s["method"]==method and s["group"]==group]
            a,b,c=[100*s["accuracy"] for s in ss[:3]]
            lines.append("| {} | {} | {:.3f}% | {:.3f}% | {:.3f}% | {:+.3f}pp | {:+.3f}pp |".format(
                method,group,a,b,c,b-a,c-b))
    lines+=["",
        "**Interpretation constraint:** Even B→C compares independently refitted models, not a direct causal identification of one particular weight. OOF authors were repeatedly examined in earlier studies; extrapolations require new writers. Final project Test was not used.",
        ""]
    (OUT/"CROSS_4WAY_SUMMARY.md").write_text("\n".join(lines),encoding="utf-8")
    print((OUT/"CROSS_4WAY_SUMMARY.md").read_text(encoding="utf-8"),flush=True)

if __name__=="__main__":
    main()
