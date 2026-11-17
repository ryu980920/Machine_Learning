#!/usr/bin/env python3
"""S5 extension: actual train-vs-validation accuracy and loss in THE SAME fit.

The 90% training subset and 10% internal validation come from sklearn 1.8.0's
actual train_test_split call, captured without modifying its returned arrays,
random generator, optimizer updates, early-stopping score, or training order.

Run a single 120-epoch fit with early_stopping=True and patience=200 so that
the same run reaches all diagnostic checkpoints. Monitor the first epoch at
which the normal patience=10 rule would have stopped (counterfactual stop),
without using held-out Writer OOF to choose the checkpoint.

Actual full training accuracy, validation accuracy, internal validation
unregularized cross-entropy and sklearn training objective loss are measured
at epochs 1, 5, 10, 15, ..., 120. OOF accuracy and four key confusion
directions are diagnostic only (never drive stopping or HP selection).
"""
from __future__ import annotations
import argparse,csv,json,time,warnings
from pathlib import Path
import numpy as np
import sklearn
import sklearn.neural_network._multilayer_perceptron as sklearn_mlp
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from exp_cmb008_s5_early_stopping import (
    ROOT,WRITERS,LABELS,V3_SHA,HANDMADE_SHA,
    load_official,load_writers,PixelScaler255,prepareraw
)

OFFICIAL=ROOT/"02_data"/"datasets"/"MNIST_EMNIST_official_split_112000.npz"
HANDMADE=ROOT/"02_data"/"Hand_made"
CONFUSIONS=((13,0,"D_to_0"),(11,8,"B_to_8"),(8,11,"8_to_B"),(0,13,"0_to_D"))
MAX_EPOCH=120
POINTS=frozenset([1]+list(range(5,MAX_EPOCH+1,5)))

def write_csv(p,rows):
    assert rows,p
    with p.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

class TrainValTraceMLP(MLPClassifier):
    """Captures sklearn's *actual* training partition from inside same fit."""
    def set_heldout(self,X,y):
        self.heldout_X=PixelScaler255().transform(X)
        self.heldout_y=np.asarray(y,dtype=int)
        self.epoch_records=[]
        self.counterfactual_patience10_stop=None
        self._internal_split_capture_count=0

    def _fit_stochastic(self,*args,**kwargs):
        original=sklearn_mlp.train_test_split
        def capture_split(*p,**k):
            out=original(*p,**k)
            self._internal_split_capture_count+=1
            self._exact_training_X=out[0]
            self._exact_training_y=self._label_binarizer.inverse_transform(out[2])
            self._exact_validation_X=out[1]
            self._exact_validation_y=self._label_binarizer.inverse_transform(out[3])
            return out
        sklearn_mlp.train_test_split=capture_split
        try:
            return super()._fit_stochastic(*args,**kwargs)
        finally:
            sklearn_mlp.train_test_split=original

    def _update_no_improvement_count(self,early_stopping,X,y,sample_weight):
        # Run sklearn's unchanged validation update and best-weight tracking.
        super()._update_no_improvement_count(early_stopping,X,y,sample_weight)
        assert early_stopping,"This experiment intentionally uses sklearn internal 10% validation."
        epoch=len(self.loss_curve_)
        if self._no_improvement_count>10 and self.counterfactual_patience10_stop is None:
            self.counterfactual_patience10_stop=int(epoch)
        if epoch not in POINTS:
            return
        assert self._internal_split_capture_count==1
        assert len(self._exact_training_X)==70848 and len(X)==7872
        # Full, actual sklearn 90% training partition, NOT a sampled proxy.
        yp_train=self.predict(self._exact_training_X)
        train_accuracy=float(np.mean(yp_train==self._exact_training_y))
        internal_accuracy=float(self.validation_scores_[-1])
        # Validation log-loss: diagnostic, not sklearn's stop criterion.
        # sklearn's training loss includes regularization; these are labelled
        # separately, and their absolute levels should not be conflated.
        probabilities=np.clip(self.predict_proba(X),1e-15,1.)
        yc=np.asarray(y,dtype=int)
        assert probabilities.shape==(7872,16)
        val_logloss=float(np.mean(-np.log(probabilities[np.arange(len(yc)),yc])))
        yp_holdout=self.predict(self.heldout_X)
        holdout_accuracy=float(np.mean(yp_holdout==self.heldout_y))
        row={"epoch":epoch,"training_accuracy":train_accuracy,
            "internal_validation_accuracy":internal_accuracy,
            "training_objective_loss":float(self.loss_curve_[-1]),
            "internal_validation_log_loss":val_logloss,
            "heldout_writer_accuracy_diagnostic_only":holdout_accuracy,
            "no_improvement_epochs":int(self._no_improvement_count)}
        for actual,pred,key in CONFUSIONS:
            row[key]=int(np.count_nonzero((self.heldout_y==actual)&(yp_holdout==pred)))
        self.epoch_records.append(row)

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--seed",type=int,choices=(42,43,44),required=True)
    a.add_argument("--heldout",type=int,choices=WRITERS,required=True)
    a.add_argument("--out",type=Path,required=True)
    args=a.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    official=load_official(OFFICIAL)
    writers=load_writers(HANDMADE)
    assert official["sha256"]==V3_SHA
    for writer in WRITERS:assert writers[writer]["sha256"]==HANDMADE_SHA[writer]
    data,wdata,aug,_=prepareraw(official,writers,"scale30_stroke28")
    rest=[w for w in WRITERS if w!=args.heldout]
    X=np.concatenate([data]+[wdata[w] for w in rest])
    y=np.concatenate([official["y_train"]]+[writers[w]["y"] for w in rest])
    assert X.shape==(78720,784) and len(y)==78720
    net=TrainValTraceMLP(hidden_layer_sizes=(256,),alpha=.0001,
        solver="adam",activation="relu",learning_rate_init=.001,
        early_stopping=True,validation_fraction=.10,n_iter_no_change=200,
        tol=1e-4,max_iter=MAX_EPOCH,random_state=args.seed)
    net.set_heldout(wdata[args.heldout],writers[args.heldout]["y"])
    model=Pipeline([("scale_0_1",PixelScaler255()),("model",net)])
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        start=time.monotonic()
        model.fit(X,y)
        elapsed=time.monotonic()-start
    assert net.n_iter_==120
    assert len(net.epoch_records)==len(POINTS)
    assert [z["epoch"] for z in net.epoch_records]==sorted(POINTS)
    assert net._internal_split_capture_count==1
    assert net.counterfactual_patience10_stop is not None,"No hypothetical stop with patience10 by 120"
    hypothetical_stop=net.counterfactual_patience10_stop
    first_best=int(np.argmax(net.validation_scores_[:hypothetical_stop])+1)
    final_best=int(np.argmax(net.validation_scores_)+1)
    # Actual stopper would have picked the highest-validation-score weights
    # at first_best and restored them, rather than this long-fit final_best.
    print("CURVE_FINISHED",json.dumps({"seed":args.seed,"heldout":args.heldout,
        "counterfactual_es10_stop":hypothetical_stop,
        "counterfactual_es10_best":first_best,
        "best_val_epoch_over_120":final_best,
        "train_accuracy_epoch120":net.epoch_records[-1]["training_accuracy"],
        "val_accuracy_epoch120":net.epoch_records[-1]["internal_validation_accuracy"]}),flush=True)
    write_csv(args.out/"same_run_train_val_curve.csv",net.epoch_records)
    m={"experiment":"EXP-CMB-008_S5","study":"TrueTrainValSameRunLearningCurve",
       "seed":args.seed,"heldout_writer":args.heldout,"total_epochs":MAX_EPOCH,
       "full_training_partition_samples":70848,"internal_validation_samples":7872,
       "actual_training_accuracy_not_subsampled":True,
       "epoch_metrics":sorted(POINTS),
       "counterfactual_es10_stop_epoch":hypothetical_stop,
       "counterfactual_es10_best_val_epoch":first_best,
       "best_validation_epoch_over_120":final_best,
       "model_init_seed":args.seed,"model_hyperparameters":{
          "hidden_layer_sizes":[256],"alpha":.0001,"activation":"relu",
          "solver":"adam","learning_rate_init":.001,
          "early_stopping":True,"validation_fraction":.1,
          "n_iter_no_change":200,"max_iter":120,"tol":.0001},
       "source_sha":{"official":official["sha256"],"handmade":{
          str(w):writers[w]["sha256"] for w in WRITERS}},
       "augmentation":aug,"sklearn_version":sklearn.__version__,
       "warnings":[str(w.message) for w in ws],"elapsed_sec":elapsed,
       "heldout_writer_used_for_fit_or_stopping":False,
       "original_test_used":False,"professor_test_used":False}
    (args.out/"manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
