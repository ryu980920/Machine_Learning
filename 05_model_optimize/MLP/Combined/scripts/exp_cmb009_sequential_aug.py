#!/usr/bin/env python3
"""EXP-CMB-009: frozen CMB-008 S6 batch128 plus one train-only geometry augmentation.

Arms: base, warp (balanced upper/lower +/-1px), rotate3, rotate5,
shear005, shear010, and a gated combo constructed AFTER separate comparisons.
No original/professor Test or new 438/480 used to train or select arms.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time,warnings
from pathlib import Path
import cv2
import joblib
import numpy as np
import sklearn
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from exp_cmb008_s5_early_stopping import (
    ROOT, WRITERS, LABELS, V3_SHA, HANDMADE_SHA,
    load_official, load_writers, PixelScaler255, prepareraw, process)
from stroke_normalization import topology

ARMS=("base","warp","rotate3","rotate5","shear005","shear010","combo")
OBSERVE=((7,9,"7_to_9"),(9,7,"9_to_7"),
         (13,0,"D_to_0"),(0,13,"0_to_D"),(11,8,"B_to_8"),(8,11,"8_to_B"))
BASE_SEED=20261118
FRACTION=.20
OFFICIAL=ROOT/"02_data/datasets/MNIST_EMNIST_official_split_112000.npz"
HANDMADE=ROOT/"02_data/Hand_made"
YY,XX=np.mgrid[0:28,0:28].astype(np.float32)

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def train_indices(y):
    """Class-balanced and predetermined before learning; never depends on OOF/Test."""
    selected=[]
    for c in range(16):
        ii=np.flatnonzero(y==c)
        order=np.random.default_rng(BASE_SEED+173*c).permutation(ii)
        selected.extend(order[:int(round(FRACTION*len(ii)))])
    ids=np.asarray(sorted(selected),dtype=np.int32)
    assert len(ids)==len(set(ids.tolist()))
    assert abs(len(ids)/len(y)-FRACTION)<.002
    return ids

def warp_map(kind,sign):
    # Continuous, zero-ended vertical window prevents hard row seams.
    if kind=="upper":
        weights=np.where(YY<=13,np.sin(np.pi*(YY+1)/15),0)
    else:
        weights=np.where(YY>=14,np.sin(np.pi*(28-YY)/15),0)
    return (XX-sign*weights).astype(np.float32),YY.astype(np.float32)

WARP_MAPS={(k,s):warp_map(k,s) for k in ("upper","lower") for s in (-1,1)}

def render(img,arm,index):
    """Same signs/upper-or-lower choice for every label; zero padding; linear sampling."""
    img=np.asarray(img,dtype=np.float32).reshape(28,28)
    bit=int((int(index)*1103515245+12345)&0x7fffffff)
    sign=1 if (bit%2) else -1
    if arm=="warp":
        region="upper" if (bit//2)%2==0 else "lower"
        mx,my=WARP_MAPS[region,sign]
        out=cv2.remap(img,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    elif arm.startswith("rotate"):
        angle=(3 if arm=="rotate3" else 5)*sign
        matrix=cv2.getRotationMatrix2D((13.5,13.5),angle,1.)
        out=cv2.warpAffine(img,matrix,(28,28),flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    elif arm.startswith("shear"):
        slope=(.05 if arm=="shear005" else .10)*sign
        matrix=np.asarray([[1.,slope,-13.5*slope],[0.,1.,0.]],dtype=np.float32)
        out=cv2.warpAffine(img,matrix,(28,28),flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    else:raise ValueError(arm)
    return out

def apply_aug(baseline,y,arm,combo_arms=None):
    if arm=="base":
        return baseline,{"arm":arm,"ratio":0,"attempted":0,"accepted":0,
                         "topology_rejected":0,"mass_rejected":0,"empty_rejected":0}
    operations=tuple(combo_arms if arm=="combo" else [arm])
    if not operations:raise ValueError("combo operations missing")
    ids=train_indices(y)
    orig=np.asarray(baseline[ids],dtype=np.float32).reshape(-1,28,28)
    edited=orig.copy()
    counts={"attempted":len(ids),"accepted":0,"topology_rejected":0,
            "mass_rejected":0,"empty_rejected":0}
    for pos,index in enumerate(ids):
        src=orig[pos]
        proposed=src.copy()
        for op in operations:proposed=render(proposed,op,index)
        if topology(proposed)!=topology(src):
            counts["topology_rejected"]+=1;continue
        mass=float(src.sum())
        if not (.7 <= float(proposed.sum())/max(mass,1.) <= 1.3):
            counts["mass_rejected"]+=1;continue
        if not np.any(proposed>=32):
            counts["empty_rejected"]+=1;continue
        edited[pos]=proposed
        counts["accepted"]+=1
    new=np.asarray(baseline,dtype=np.float32).copy()
    new[ids]=process(edited,"com",20)
    return new,{"arm":arm,"operations":list(operations),
                "ratio":float(len(ids)/len(y)),
                "official_aug_source_ids_sha256":hashlib.sha256(ids.tobytes()).hexdigest(),
                **counts}

def save_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def measure(y,p):
    cm=confusion_matrix(y,p,labels=np.arange(16))
    return {"n":int(len(y)),"correct":int((y==p).sum()),
            "accuracy":float(np.mean(y==p)),
            "macro_f1":float(f1_score(y,p,labels=np.arange(16),average="macro",zero_division=0)),
            "pairs":{key:int(cm[a,b]) for a,b,key in OBSERVE},
            "class_errors":{LABELS[c]:int(cm[c].sum()-cm[c,c]) for c in range(16)},
            "class_support":{LABELS[c]:int(cm[c].sum()) for c in range(16)}}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm",required=True,choices=ARMS)
    p.add_argument("--seed",required=True,type=int,choices=(42,43,44))
    p.add_argument("--heldout",required=True,type=int,choices=WRITERS)
    p.add_argument("--selection",type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    combo=None
    if a.arm=="combo":
        if not a.selection or not a.selection.is_file():raise ValueError("Need selection JSON")
        combo=json.loads(a.selection.read_text())["selected"]
        if len(combo)<2:
            print("COMBO_NOT_JUSTIFIED",json.dumps(combo),flush=True)
            a.out.mkdir(parents=True)
            (a.out/"skip.json").write_text(json.dumps({"reason":"fewer_than_two_gated_improvements","selected":combo}))
            return
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    started=time.perf_counter()
    official=load_official(OFFICIAL)
    handmade=load_writers(HANDMADE)
    assert official["sha256"]==V3_SHA
    assert all(handmade[w]["sha256"]==HANDMADE_SHA[w] for w in WRITERS)
    off_x,w_x,baseline_aug,_=prepareraw(official,handmade,"scale30_stroke28")
    off_x,delta_info=apply_aug(off_x,official["y_train"],a.arm,combo)
    folds=[w for w in WRITERS if w!=a.heldout]
    train_x=np.concatenate([off_x]+[w_x[w] for w in folds])
    train_y=np.concatenate([official["y_train"]]+[handmade[w]["y"] for w in folds])
    assert train_x.shape==(78720,784) and train_y.shape==(78720,)
    ho_x=w_x[a.heldout]
    ho_y=np.asarray(handmade[a.heldout]["y"],dtype=int)
    va_x=process(official["X_val"],"com",20)
    va_y=np.asarray(official["y_val"],dtype=int)
    mlp=MLPClassifier(hidden_layer_sizes=(256,),alpha=.0001,
        learning_rate_init=.001,solver="adam",activation="relu",
        early_stopping=False,batch_size=128,max_iter=200,
        n_iter_no_change=10,tol=.0001,random_state=a.seed)
    model=Pipeline([("scale_0_1",PixelScaler255()),("model",mlp)])
    with warnings.catch_warnings(record=True) as wrns:
        warnings.simplefilter("always")
        tick=time.perf_counter()
        model.fit(train_x,train_y)
        fit_seconds=time.perf_counter()-tick
    ho_pred=np.asarray(model.predict(ho_x),dtype=np.int32)
    va_pred=np.asarray(model.predict(va_x),dtype=np.int32)
    oof=measure(ho_y,ho_pred);val=measure(va_y,va_pred)
    save_csv(a.out/"oof_predictions.csv",
        [{"arm":a.arm,"seed":a.seed,"heldout":a.heldout,
          "index":int(i),"actual":LABELS[int(ho_y[i])],
          "pred":LABELS[int(ho_pred[i])]} for i in range(640)])
    from joblib import dump
    dump(model,a.out/"trained_model.joblib",compress=3)
    meta={"experiment":"EXP-CMB-009","arm":a.arm,"combo_operations":combo,
      "seed":a.seed,"heldout":a.heldout,"model_batch":128,
      "frozen":{"hidden_layer_sizes":[256],"alpha":.0001,
        "learning_rate_init":.001,"solver":"adam","activation":"relu",
        "early_stopping":False,"max_iter":200,"tol":.0001,"n_iter_no_change":10},
      "training_samples":len(train_y),"official_val_samples":len(va_y),
      "heldout_samples":len(ho_y),"official_sha256":official["sha256"],
      "writer_sha256":{str(w):handmade[w]["sha256"] for w in WRITERS},
      "baseline_aug":baseline_aug,"new_train_only_aug":delta_info,
      "original_test_used":False,"professor_test_used":False,
      "external_438_used_in_training_or_selection":False,
      "oof":oof,"official_val":val,"n_iter":int(mlp.n_iter_),
      "fit_seconds":float(fit_seconds),"elapsed_seconds":float(time.perf_counter()-started),
      "warnings":[str(w.message) for w in wrns],
      "sklearn_version":sklearn.__version__,
      "model_sha256":sha(a.out/"trained_model.joblib")}
    (a.out/"manifest.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n")
    print("CMB009_FIT_OK",json.dumps({"arm":a.arm,"seed":a.seed,"heldout":a.heldout,
      "oof":oof["accuracy"],"val":val["accuracy"],"7_to_9":oof["pairs"]["7_to_9"],
      "accepted":delta_info["accepted"],"fit_sec":round(fit_seconds,1)}),flush=True)
if __name__=="__main__":main()
