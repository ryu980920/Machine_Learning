#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OfficialVal 5-scale final audit of frozen EXP017 winners; no training/handwriting.

Include exact prior EXP016 80-14 checkpoints for paired scale benchmark.
"""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np,joblib
from exp006_size_com_comparison import ROOT,LABELS,process,load_official,image_geometry
from exp011_012_causal_probes import fixed_scale
from exp017_hparam_coarse import CONF,config_name
SCALES=(.8,.9,1.,1.1,1.2)
SEEDS=(42,43,44)
FOCUS={"B","8","D"}
def path_for(meta,seed,coarse,fine,old):
    if meta is None:
        return Path(old)/f"exp016-s{seed}-r80"/"size80_stroke14.joblib"
    name=meta["variant"]
    if meta["source"]=="coarse":
        found=[i for i,(h,a,lr) in enumerate(CONF) if config_name(h,a,lr)==name]
        assert len(found)==1
        return Path(coarse)/f"exp017-coarse-s{seed}-p{found[0]%3}"/(name+".joblib")
    assert meta["source"]=="fine"
    slot=int(name.split("_",1)[0].replace("slot",""))-1
    assert slot in (0,1,2,3)
    return Path(fine)/f"exp017-fine-s{seed}-slot{slot}"/(name+".joblib")
def write(path,rows):
    assert rows
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run(metadata_path,coarse,fine,old,out):
    doc=json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    assert doc["experiment"]=="EXP017" and doc["stage"]=="joint_final"
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    if doc["selected_frozen_candidate"] is None:
        (out/"SELECTION.json").write_text(json.dumps({"selected":None,"reason":"No candidate passed original predeclared gates"},indent=2))
        print("EXP017_NO_GATE_PASS_UNCHANGED",flush=True);return
    official=load_official(ROOT/"02_data"/"datasets"/"MNIST_EMNIST_official_split_112000.npz")
    y=np.asarray(official["y_val"],dtype=int);X=official["X_val"]
    assert X.shape==(19200,784) and set(np.unique(y))==set(range(16))
    target=int(np.median(image_geometry(official["X_train"])["long_side"]))
    assert target==20
    scaled={s:process(X if s==1. else fixed_scale(X,s),"com",target) for s in SCALES}
    contenders=[("rank1",doc["selected_frozen_candidate"])]
    if doc["runner_up"] is not None:contenders.append(("rank2",doc["runner_up"]))
    contenders.append(("incumbent_80_14",None))
    metrics=[];checkpoints=[]
    for label,meta in contenders:
        for seed in SEEDS:
            path=path_for(meta,seed,coarse,fine,old)
            assert path.exists(),path
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            model=joblib.load(path)
            probabilities={};pred={}
            for s in SCALES:
                probs=np.asarray(model.predict_proba(scaled[s]),dtype=np.float32)
                assert probs.shape==(19200,16)
                probabilities[s]=probs[:,10:].sum(axis=1)
                pred[s]=np.argmax(probs,axis=1)
            base=probabilities[1.]>=.5
            masks={"all":np.ones(len(y),dtype=bool),"digits":y<10,"letters":y>=10}
            masks.update({"class_"+cl:y==i for i,cl in enumerate(LABELS)})
            for scale in SCALES:
                for group,m in masks.items():
                    p=probabilities[scale][m];q=probabilities[1.][m]
                    metrics.append({"candidate":label,
                      "variant":meta["variant"] if meta else "size80_stroke14",
                      "seed":seed,"scale":scale,"group":group,"n":int(np.sum(m)),
                      "accuracy":float(np.mean(pred[scale][m]==y[m])),
                      "family_flip":float(np.mean((p>=.5)!=base[m])),
                      "mean_abs_prob_delta":float(np.mean(np.abs(p-q)))})
            checkpoints.append({"label":label,"seed":seed,"variant":meta["variant"] if meta else "size80_stroke14",
                "source_path":str(path),"sha256":digest,"bytes":path.stat().st_size,
                "fitting_handmade_used":False})
            print("EXP017_FROZEN_AUDIT",label,seed,"val_accuracy",
                 next(r["accuracy"] for r in metrics if r["candidate"]==label and r["seed"]==seed and r["scale"]==1. and r["group"]=="all"),flush=True)
            del model
    write(out/"frozen_models_all_seed_five_scale_metrics.csv",metrics)
    def measure(label,group,sc,field):
        x=[r[field] for r in metrics if r["candidate"]==label and r["group"]==group and abs(r["scale"]-sc)<1e-9]
        assert len(x)==3,(label,group,sc,field,len(x))
        return sum(x)/len(x)
    summary=[]
    for label,meta in contenders:
        focal=max((measure(label,"class_"+cl,sc,"family_flip"),cl,sc)
                for cl in FOCUS for sc in (.8,.9,1.1,1.2))
        allmax=max((measure(label,"class_"+cl,sc,"family_flip"),cl,sc)
                for cl in LABELS for sc in (.8,.9,1.1,1.2))
        digit12=measure(label,"digits",1.2,"family_flip")
        letter08=measure(label,"letters",.8,"family_flip")
        val=measure(label,"all",1.,"accuracy")
        passed=(focal[0]<=.20+1e-10 and allmax[0]<=.25+1e-10 and digit12<=.10+1e-10
                and letter08<=.10+1e-10 and val>=.9431076388888888-1e-10
                and measure(label,"digits",1.2,"mean_abs_prob_delta")<=.12+1e-10
                and measure(label,"letters",.8,"mean_abs_prob_delta")<=.12+1e-10)
        summary.append({"candidate":label,"variant":meta["variant"] if meta else "size80_stroke14",
            "val_accuracy":val,"val_accuracy_08":measure(label,"all",.8,"accuracy"),
            "val_accuracy_12":measure(label,"all",1.2,"accuracy"),
            "digit_12_flip":digit12,"letter_08_flip":letter08,
            "B8D_worst_flip_full5scale":focal[0],"B8D_worst_class":focal[1],
            "B8D_worst_scale":focal[2],"all16_worst_flip_full5scale":allmax[0],
            "all16_worst_class":allmax[1],"all16_worst_scale":allmax[2],
            "passed_five_scale_audit":passed})
    write(out/"frozen_3seed_model_comparison.csv",summary)
    acceptable=[x for x in summary if x["candidate"] in ("rank1","rank2") and x["passed_five_scale_audit"]]
    acceptable.sort(key=lambda x:(-x["val_accuracy"],x["B8D_worst_flip_full5scale"]))
    winner=acceptable[0] if acceptable else None
    select={"experiment":"EXP017","stage":"frozen_officialval_five_scale_audit",
      "new_candidate":winner["variant"] if winner else None,
      "new_candidate_label":winner["candidate"] if winner else None,
      "source":next((meta["source"] for label,meta in contenders if winner and label==winner["candidate"]),None),
      "source_sha256":official["sha256"],"team_handwriting_evaluated":False,
      "original_test_used":False,"professor_test_used":False,
      "checkpoints":checkpoints,"comparison":summary,"predeclared_joint_gate_already_passed":True,
      "five_scale_supplementary_gate_pass":winner is not None}
    (out/"SELECTION.json").write_text(json.dumps(select,ensure_ascii=False,indent=2),encoding="utf-8")
    text=["# EXP유017 동결 모델 OfficialVal 5배율 교차검증","","| 후보 | 1.0 Val | 0.8 Val | 1.2 Val | B·8·D 최악 | 16클래스 최악 | 5배율 Gate |",
         "|---|---:|---:|---:|---:|---:|---|"]
    for r in summary:
        text.append(f"| {r['variant']} | "+ " | ".join(f"{r[k]*100:.2f}%" for k in 
            ("val_accuracy","val_accuracy_08","val_accuracy_12","B8D_worst_flip_full5scale","all16_worst_flip_full5scale"))+
            " | "+("통과" if r["passed_five_scale_audit"] else "탈락")+" |")
    text+=["",f"동결 후보: **{winner['variant'] if winner else '안전 기준 통과 후보 없음'}**.",
           "팀원이 모은 손글씨를 아직 사용하지 않았으며, 교수님 Test/Original Test도 사용하지 않음.",
           "팀 손글씨 실험 전 이 결과와 가중치만 저장하고 선택 기준을 고정해야 한다."]
    (out/"SUMMARY.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print("EXP017_FIVE_SCALE_COMPLETE",winner["variant"] if winner else "NONE",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--metadata",type=Path,required=True)
    p.add_argument("--coarse-models",type=Path,required=True)
    p.add_argument("--fine-models",type=Path,required=True)
    p.add_argument("--old-models",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.metadata,a.coarse_models,a.fine_models,a.old_models,a.out)
