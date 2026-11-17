#!/usr/bin/env python3
"""CMB008 S6: change Adam minibatch size ONLY (200, 128, 64).

One independent model per (seed 42/43/44, leave-one-writer-out 41/42/43/44,
batch_size 200/128/64). Entire 78,720 training candidate instances used.
early_stopping=False, max_iter=200, n_iter_no_change=10 (sklearn default).
No independent test, Professor Test, or new 480 used.
"""
from __future__ import annotations
import argparse,csv,json,time,warnings,hashlib
import joblib
from pathlib import Path
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from exp_cmb008_s5_early_stopping import (
    ROOT,WRITERS,LABELS,V3_SHA,HANDMADE_SHA,
    load_official,load_writers,PixelScaler255,prepareraw,
    process,fixed_scale
)

OFFICIAL=ROOT/"02_data"/"datasets"/"MNIST_EMNIST_official_split_112000.npz"
HANDMADE=ROOT/"02_data"/"Hand_made"
WATCH=((13,0,"D_to_0"),(11,8,"B_to_8"),(8,11,"8_to_B"),(0,13,"0_to_D"))
FOCUS=(0,8,11,13)

def savecsv(path,rows):
    assert rows
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]))
        wr.writeheader();wr.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--batch-size",choices=(200,128,64),type=int,required=True)
    p.add_argument("--seed",choices=(42,43,44),type=int,required=True)
    p.add_argument("--heldout",choices=WRITERS,type=int,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    assert not a.out.exists(),a.out
    a.out.mkdir(parents=True)
    official=load_official(OFFICIAL)
    writers=load_writers(HANDMADE)
    assert official["sha256"]==V3_SHA
    for w in WRITERS:assert writers[w]["sha256"]==HANDMADE_SHA[w]
    assert len(official["X_train"])==76800 and len(official["X_val"])==19200
    start=time.perf_counter()
    ox,wx,aug,_=prepareraw(official,writers,"scale30_stroke28")
    remain=[w for w in WRITERS if w!=a.heldout]
    X=np.concatenate([ox]+[wx[w] for w in remain])
    y=np.concatenate([official["y_train"]]+[writers[w]["y"] for w in remain])
    assert X.shape==(78720,784) and y.shape==(78720,)
    val_x=process(official["X_val"],"com",20)
    val_y=np.asarray(official["y_val"],dtype=int)
    held_x=wx[a.heldout]
    held_y=np.asarray(writers[a.heldout]["y"],dtype=int)
    assert len(held_y)==640 and held_y.shape==(640,)
    net=MLPClassifier(random_state=a.seed,hidden_layer_sizes=(256,),
        alpha=.0001,learning_rate_init=.001,solver="adam",activation="relu",
        early_stopping=False,batch_size=a.batch_size,max_iter=200,
        n_iter_no_change=10,tol=.0001)
    model=Pipeline([("scale_0_1",PixelScaler255()),("model",net)])
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        tick=time.perf_counter()
        model.fit(X,y)
        elapsed=time.perf_counter()-tick
    # EXP-CMB-008 S6 external audit: preserve precisely this fitted Pipeline.
    # Runs ONLY on isolated experiment branch; does not modify model parameters.
    model_path=a.out/"trained_model.joblib"
    joblib.dump(model,model_path,compress=3)
    model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert net.n_features_in_==784
    assert np.array_equal(net.classes_,np.arange(16))
    yp=np.asarray(model.predict(held_x),dtype=int)
    vp=np.asarray(model.predict(val_x),dtype=int)
    assert len(yp)==640 and len(vp)==19200
    predictions=[{"batch_size":a.batch_size,"seed":a.seed,
      "writer":a.heldout,"index":i,"actual":LABELS[int(held_y[i])],
      "pred":LABELS[int(yp[i])]} for i in range(640)]
    savecsv(a.out/"oof_predictions.csv",predictions)
    focus={}
    for class_id in range(16):
        mask=held_y==class_id
        focus[LABELS[class_id]]={
            "n":int(mask.sum()),
            "errors":int(np.count_nonzero(yp[mask]!=class_id))}
    assert all(v["n"]==40 for v in focus.values())
    pairs={key:int(np.count_nonzero((held_y==aa)&(yp==bb)))
        for aa,bb,key in WATCH}
    # Calculate original scale-sensitivity gate: worst class-specific
    # family-flip on OOF/OfficialVal under 0.8 and 1.2 scale, using CoM.
    # Predict_proba classes 0-9 vs A-F; family flip threshold 0.5.
    sensitivity=[]
    for ds,raw_data,yy,base in (
        ("OOF",writers[a.heldout]["X"],held_y,held_x),
        ("OfficialVal",official["X_val"],val_y,val_x)):
        original_family=(model.predict_proba(base)[:,10:].sum(axis=1)>=.5)
        for scale in (.8,1.2):
            xx=process(fixed_scale(raw_data,scale),"com",20)
            scaled_family=(model.predict_proba(xx)[:,10:].sum(axis=1)>=.5)
            for cl in range(16):
                mask=yy==cl
                sensitivity.append({"dataset":ds,"writer":a.heldout,
                  "seed":a.seed,"scale":scale,"actual":LABELS[cl],
                  "n":int(mask.sum()),
                  "flip_count":int(np.count_nonzero((original_family!=scaled_family)&mask))})
    savecsv(a.out/"class_family_flip.csv",sensitivity)
    meta={"experiment":"EXP-CMB-008","stage":"S6_batch_size",
        "batch_size":a.batch_size,"seed":a.seed,"heldout":a.heldout,
        "frozen":{"hidden_layer_sizes":[256],"alpha":.0001,
          "learning_rate_init":.001,"solver":"adam","activation":"relu",
          "early_stopping":False,"max_iter":200,
          "tol":.0001,"n_iter_no_change":10},
        "exported_model":"trained_model.joblib","exported_model_sha256":model_sha256,
        "n_training":len(y),"n_heldout":len(held_y),
        "n_official_val":len(val_y),
        "official_sha256":official["sha256"],
        "handmade_sha256":{str(w):writers[w]["sha256"] for w in WRITERS},
        "augmentation":aug,"augmentation_seed":20261116,
        "handmade_train_preprocess":"CoM-only",
        "heldout_and_official_val_preprocess":"CoM-only",
        "writer_heldout_not_used_to_train":True,
        "independent_test_used":False,"professor_test_used":False,
        "new_handmade_480_used":False,
        "writer_oof_accuracy":float(np.mean(yp==held_y)),
        "official_val_accuracy":float(np.mean(vp==val_y)),
        "focus_class":focus,"directional_pairs":pairs,
        "n_iter":int(net.n_iter_),"first_loss":float(net.loss_curve_[0]),
        "last_loss":float(net.loss_curve_[-1]),
        "fit_seconds":elapsed,
        "wall_seconds":time.perf_counter()-start,
        "sklearn_version":sklearn.__version__,
        "warnings":[str(w.message) for w in got]}
    (a.out/"manifest.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("BATCH_FOLD_OK",json.dumps({"batch":a.batch_size,"seed":a.seed,
      "heldout":a.heldout,"oof":meta["writer_oof_accuracy"],
      "official":meta["official_val_accuracy"],"pairs":pairs,
      "n_iter":net.n_iter_,"seconds":round(elapsed,1)},ensure_ascii=False),flush=True)
if __name__=="__main__":
    main()
