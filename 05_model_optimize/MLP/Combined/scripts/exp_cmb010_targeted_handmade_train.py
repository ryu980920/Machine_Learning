#!/usr/bin/env python3
"""EXP-CMB-010: Compare baseline Combined Batch128 vs adding new 480 targeted handwritings.

No external438, Original Test, Professor Test or held-out Writer is used to fit.
Treat all 480 as ONE unknown-writer cohort, used only in fit for the add480 arm.
Model/preprocessing/official augment are otherwise the CMB008/CMB009 exact reference.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time,warnings
from pathlib import Path
import joblib,numpy as np,sklearn
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score,confusion_matrix
from exp_cmb008_s5_early_stopping import (
    ROOT,WRITERS,LABELS,V3_SHA,HANDMADE_SHA,
    load_official,load_writers,PixelScaler255,prepareraw,process
)
EXTRA=ROOT/"02_data"/"Hand_made_extra"/"CMB010_7_8_9_B_480.npz"
EXPECTED_SHA="3d1686c67d7bb5c6193384eca1aa9218213c842252baa5b7e172698743b83fc7"
TARGET_COUNTS={7:160,8:80,9:160,11:80}
OFFICIAL=ROOT/"02_data"/"datasets"/"MNIST_EMNIST_official_split_112000.npz"
HANDMADE=ROOT/"02_data"/"Hand_made"
PAIRS=((7,9,"7_to_9"),(9,7,"9_to_7"),(11,8,"B_to_8"),
       (8,11,"8_to_B"),(0,13,"0_to_D"),(13,0,"D_to_0"),
       (7,10,"7_to_A"),(4,15,"4_to_F"))

def sha_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def validate_extra():
    assert sha_file(EXTRA)==EXPECTED_SHA,"New input NPZ unexpectedly changed"
    with np.load(EXTRA,allow_pickle=False) as z:
        required={"X","y","images","labels","label_text","class_names","available_classes","source_image","cell_row_col"}
        assert required.issubset(z.files),(required-set(z.files))
        x=np.asarray(z["X"]);y=np.asarray(z["y"],dtype=np.int32)
        assert x.shape==(480,784) and x.dtype==np.uint8 and y.shape==(480,)
        assert np.array_equal(x,z["images"].reshape(480,784))
        assert np.array_equal(y,z["labels"])
        assert np.array_equal(z["label_text"],np.asarray([LABELS[k] for k in y]))
        assert np.array_equal(z["class_names"],np.asarray(list(LABELS)))
        assert {int(c):int(np.sum(y==c)) for c in sorted(set(y))}==TARGET_COUNTS
        assert len({row.tobytes() for row in x})==480,"Duplicate raw image"
        img=x.reshape(-1,28,28)
        mask=img>=32
        assert np.all(mask.reshape(480,-1).sum(axis=1)>=12),"Missing foreground"
        assert not(np.any(mask[:,0,:]) or np.any(mask[:,-1,:]) or
                   np.any(mask[:,:,0]) or np.any(mask[:,:,-1])),"Foreground clipped at edge"
        assert int(img.max())==206
        assert "writer_id" not in z.files and "writer_ids" not in z.files
    return x,y

def dump_csv(path,records):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)

def metrics(y,p):
    cm=confusion_matrix(y,p,labels=np.arange(16))
    assert cm.sum()==len(y)
    return {"n":int(len(y)),"correct":int(np.sum(y==p)),
        "accuracy":float(np.mean(y==p)),
        "macro_f1":float(f1_score(y,p,average="macro",labels=np.arange(16),zero_division=0)),
        "class_support":{LABELS[c]:int(cm[c].sum()) for c in range(16)},
        "class_errors":{LABELS[c]:int(cm[c].sum()-cm[c,c]) for c in range(16)},
        "pairs":{name:int(cm[a,b]) for a,b,name in PAIRS},
        "digit_correct":int(np.sum(p[y<10]==y[y<10])),
        "letter_correct":int(np.sum(p[y>=10]==y[y>=10]))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",choices=("base","add480"),required=True)
    ap.add_argument("--seed",choices=(42,43,44),type=int,required=True)
    ap.add_argument("--heldout",choices=WRITERS,type=int,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    begin=time.perf_counter()
    off=load_official(OFFICIAL);writers=load_writers(HANDMADE)
    assert off["sha256"]==V3_SHA
    for w in WRITERS:assert writers[w]["sha256"]==HANDMADE_SHA[w]
    new_raw,new_y=validate_extra()
    main_x,wr_x,base_aug,_=prepareraw(off,writers,"scale30_stroke28")
    remain=[w for w in WRITERS if w!=a.heldout]
    fit_x=[main_x]+[wr_x[w] for w in remain]
    fit_y=[np.asarray(off["y_train"],dtype=np.int32)]+[np.asarray(writers[w]["y"],dtype=np.int32) for w in remain]
    if a.arm=="add480":
        # New cohort is never used for writer-held-out evaluation or OfficialVal.
        # Do not change or augment labels; all are unambiguously encoded in NPZ.
        new_x=process(new_raw,"com",20)
        fit_x.append(new_x)
        fit_y.append(new_y)
    fit_x=np.concatenate(fit_x)
    fit_y=np.concatenate(fit_y)
    expected=78720+(480 if a.arm=="add480" else 0)
    assert fit_x.shape==(expected,784) and fit_y.shape==(expected,)
    held_x=wr_x[a.heldout]
    held_y=np.asarray(writers[a.heldout]["y"],dtype=int)
    val_x=process(off["X_val"],"com",20)
    val_y=np.asarray(off["y_val"],dtype=int)
    assert len(held_y)==640 and len(val_y)==19200
    mlp=MLPClassifier(random_state=a.seed,hidden_layer_sizes=(256,),
        alpha=.0001,learning_rate_init=.001,solver="adam",activation="relu",
        early_stopping=False,batch_size=128,max_iter=200,
        n_iter_no_change=10,tol=.0001)
    model=Pipeline([("scale_0_1",PixelScaler255()),("model",mlp)])
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        tick=time.perf_counter()
        model.fit(fit_x,fit_y)
        fit_sec=time.perf_counter()-tick
    assert np.array_equal(mlp.classes_,np.arange(16))
    op=np.asarray(model.predict(held_x),dtype=int)
    vp=np.asarray(model.predict(val_x),dtype=int)
    oof=metrics(held_y,op)
    val=metrics(val_y,vp)
    dump_csv(a.out/"heldout_predictions.csv",
        [{"arm":a.arm,"seed":a.seed,"heldout":a.heldout,
          "index":i,"actual":LABELS[int(held_y[i])],
          "pred":LABELS[int(op[i])]} for i in range(640)])
    # Optional fitted models preserved only in GitHub Actions artifacts, not deployed.
    joblib.dump(model,a.out/"candidate_model.joblib",compress=3)
    m={"experiment":"EXP-CMB-010","arm":a.arm,"seed":a.seed,"heldout":a.heldout,
       "official_sha256":off["sha256"],
       "old_writers_sha256":{str(w):writers[w]["sha256"] for w in WRITERS},
       "added_source_sha256":EXPECTED_SHA,
       "new_cohort_writer_id_known":False,"new_cohort_grouped_as_one":True,
       "added_images_count":480 if a.arm=="add480" else 0,
       "added_by_class":{LABELS[k]:v if a.arm=="add480" else 0 for k,v in TARGET_COUNTS.items()},
       "training_samples":expected,"heldout_samples":len(held_y),
       "official_val_samples":len(val_y),"heldout_in_fit":False,
       "new_cohort_in_fit":a.arm=="add480",
       "new_cohort_tested_as_independent":False,
       "external_438_used":False,"original_test_used":False,"professor_test_used":False,
       "frozen":{"hidden_layer_sizes":[256],"alpha":.0001,"learning_rate_init":.001,
                 "activation":"relu","solver":"adam","early_stopping":False,
                 "batch_size":128,"max_iter":200,"n_iter_no_change":10,"tol":.0001,
                 "official_aug":"scale30_stroke28","inference_preprocess":"CoM"},
       "augmentation":base_aug,"oof":oof,"official_val":val,
       "n_iter":int(mlp.n_iter_),"fit_seconds":float(fit_sec),
       "total_seconds":float(time.perf_counter()-begin),
       "sklearn_version":sklearn.__version__,
       "warnings":[str(w.message) for w in got],
       "model_sha256":sha_file(a.out/"candidate_model.joblib")}
    (a.out/"manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("CMB010_FINISHED",json.dumps({"arm":a.arm,"seed":a.seed,
        "heldout":a.heldout,"fit_count":expected,"oof":oof["accuracy"],
        "val":val["accuracy"],"7_to_9":oof["pairs"]["7_to_9"],
        "B_to_8":oof["pairs"]["B_to_8"],"fit_seconds":round(fit_sec,1)}),flush=True)

if __name__=="__main__":main()
