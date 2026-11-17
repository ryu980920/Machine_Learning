#!/usr/bin/env python3
"""EXP-CMB-008 / 12: faithful per-epoch S5 trace and continuation.

Variants per seed and held-out writer:
  es10       exact S5 ES=True settings; records stop and best checkpoint.
  internal120 ES=True, same train/validation split/optimizer, patience 200,
              continue to 120 epochs (NO Writer/OfficialVal early stopping).
  full120    ES=False, 100% training fold, patience 200, continue to 120 epochs.
Neither held-out Writer nor OfficialVal is used to stop or choose a model.
The 120-epoch limit is for diagnosis, not a newly selected training HP.
"""
from __future__ import annotations
import argparse,csv,json,time,warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from exp_cmb008_s5_early_stopping import (
    ROOT,WRITERS,LABELS,V3_SHA,HANDMADE_SHA,load_official,load_writers,
    PixelScaler255,prepareraw,process
)
PAIRS=((13,0,"D_to_0"),(11,8,"B_to_8"),(8,11,"8_to_B"),(0,13,"0_to_D"))
VARIANTS=("es10","internal120","full120")
OFFICIAL=ROOT/"02_data"/"datasets"/"MNIST_EMNIST_official_split_112000.npz"
HANDMADE=ROOT/"02_data"/"Hand_made"

def save_csv(path, rows):
    rows=list(rows)
    assert rows, path
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

class TracedMLP(MLPClassifier):
    """Hook after each original sklearn 1.8 epoch; never alters its Adam updates."""
    def set_probe(self, X, y):
        self._curve_probe_X=PixelScaler255().transform(X)
        self._curve_probe_y=np.asarray(y,dtype=int)
        self._curve_records=[]
        self._curve_start=time.perf_counter()

    def _update_no_improvement_count(self, early_stopping, X, y, sample_weight):
        # sklearn's original validation score, best coefficients and patience logic.
        super()._update_no_improvement_count(early_stopping, X, y, sample_weight)
        epoch=len(self.loss_curve_)
        pred=np.asarray(self.predict(self._curve_probe_X),dtype=int)
        truth=self._curve_probe_y
        assert len(pred)==len(truth)==640
        row={"epoch":epoch,"train_loss":float(self.loss_curve_[-1]),
             "internal_val_accuracy":float(self.validation_scores_[-1]) if early_stopping else "",
             "internal_val_loss":"",
             "heldout_oof_accuracy":float(np.mean(pred==truth)),
             "n_no_improvement":int(self._no_improvement_count)}
        for a,b,name in PAIRS:
            row[name]=int(np.count_nonzero((truth==a)&(pred==b)))
        if early_stopping and (epoch==1 or epoch%5==0):
            # Internal Validation loss is a diagnostic only, not used in stopping.
            probs=np.clip(self.predict_proba(X),1e-15,1.)
            yy=np.asarray(y)
            actual=np.argmax(yy,axis=1) if yy.ndim==2 else yy.astype(int)
            row["internal_val_loss"]=float(np.mean(-np.log(probs[np.arange(len(actual)),actual])))
        self._curve_records.append(row)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,choices=(42,43,44),required=True)
    ap.add_argument("--heldout",type=int,choices=WRITERS,required=True)
    ap.add_argument("--variant",choices=VARIANTS,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    off=load_official(OFFICIAL)
    writers=load_writers(HANDMADE)
    assert off["sha256"]==V3_SHA
    assert len(off["X_train"])==76800 and len(off["X_val"])==19200
    for w in WRITERS:assert writers[w]["sha256"]==HANDMADE_SHA[w]
    xofficial,xwriters,augmentation,_=prepareraw(off,writers,"scale30_stroke28")
    others=[w for w in WRITERS if w!=a.heldout]
    X=np.concatenate([xofficial]+[xwriters[w] for w in others])
    y=np.concatenate([off["y_train"]]+[writers[w]["y"] for w in others])
    assert X.shape==(78720,784) and y.shape==(78720,)
    assert 0.298<=augmentation["actual_ratio"]<=0.302
    yheld=np.asarray(writers[a.heldout]["y"],dtype=int)
    es=a.variant!="full120"
    patience=10 if a.variant=="es10" else 200
    epochs=200 if a.variant=="es10" else 120
    net=TracedMLP(
        random_state=a.seed,hidden_layer_sizes=(256,),alpha=.0001,
        learning_rate_init=.001,solver="adam",activation="relu",max_iter=epochs,
        early_stopping=es,validation_fraction=.1,n_iter_no_change=patience,tol=.0001
    )
    net.set_probe(xwriters[a.heldout],yheld)
    pipe=Pipeline([("scale_0_1",PixelScaler255()),("model",net)])
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        started=time.perf_counter()
        pipe.fit(X,y)
        train_sec=time.perf_counter()-started
    save_csv(a.out/"epoch_curve.csv",net._curve_records)
    restored=pipe.predict(xwriters[a.heldout]).astype(int)
    save_csv(a.out/"final_heldout_predictions.csv",[
      {"seed":a.seed,"heldout":a.heldout,"index":i,"actual":LABELS[int(yheld[i])],
       "pred":LABELS[int(restored[i])]} for i in range(len(yheld))
    ])
    score=np.asarray(getattr(net,"validation_scores_",[]),dtype=float)
    best_epoch=int(np.argmax(score)+1) if es else None
    # sklearn restores weights from highest validation-score epoch, not the last epoch.
    best_curve=net._curve_records[best_epoch-1] if best_epoch else None
    matches=True
    if best_curve:
        matches=(
          abs(float(np.mean(restored==yheld))-best_curve["heldout_oof_accuracy"])<1e-12
          and all(int(np.sum((yheld==aa)&(restored==bb)))==best_curve[nn]
                  for aa,bb,nn in PAIRS)
        )
        assert matches,"Restored weights differ in OOF accuracy or key error counts from best validation epoch"
    meta={"experiment":"EXP-CMB-008","extension":"S5_Learning_Curve_Diagnostic",
       "variant":a.variant,"seed":a.seed,"heldout":a.heldout,
       "fit_epochs":int(net.n_iter_),"best_internal_val_epoch":best_epoch,
       "best_internal_val_accuracy":float(np.max(score)) if es else None,
       "final_restored_heldout_accuracy":float(np.mean(restored==yheld)),
       "last_epoch_heldout_accuracy":net._curve_records[-1]["heldout_oof_accuracy"],
       "restored_matches_best_epoch_metrics":matches if es else None,
       "start_training_samples":78720,"training_fraction_if_internal_val":.9 if es else 1.,
       "continued_patience":patience,"max_iter":epochs,"tol":.0001,
       "early_stopping":es,"validation_fraction":.1 if es else None,
       "fixed_hidden_layers":[256],"fixed_alpha":.0001,"fixed_lr":.001,
       "random_state":a.seed,"preprocess":"Official Train size30 stroke28, CoM",
       "augmentation_seed":20261116,"augmentation":augmentation,
       "original_official_sha":off["sha256"],
       "handmade_sha":{str(k):writers[k]["sha256"] for k in WRITERS},
       "fit_seconds":train_sec,
       "warnings":[str(w.message) for w in got],
       "sklearn_version":sklearn.__version__,
       "heldout_writer_used_for_training":False,
       "heldout_writer_used_for_early_stop":False,
       "official_val_used_for_early_stop":False,
       "original_test_used":False,"professor_test_used":False,"new_handmade_480_used":False}
    (a.out/"manifest.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FINISHED_TRACE",json.dumps({"variant":a.variant,"seed":a.seed,
      "writer":a.heldout,"n_iter":net.n_iter_,"best_epoch":best_epoch,
      "restored_oof":meta["final_restored_heldout_accuracy"]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
