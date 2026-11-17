#!/usr/bin/env python3
"""Make presentation-ready 16:9 learning curves from verified CMB008 raw CSV.

This does NOT refit models or modify HP. The same-fit 90/10 internal-val
learning curves diagnose overfitting; the ES=False selection comes from the
separate S5 heldout-writer experiment, not these traces.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED_EPOCHS=[1]+list(range(5,121,5))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--s5",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    with a.csv.open(encoding="utf-8-sig",newline="") as handle:
        rows=list(csv.DictReader(handle))
    assert [int(v["epoch"]) for v in rows]==EXPECTED_EPOCHS
    assert all(int(v["n_folds"])==12 for v in rows)
    diag=json.loads(a.summary.read_text(encoding="utf-8"))
    result=json.loads(a.s5.read_text(encoding="utf-8"))
    assert diag["folds"]==12 and diag["full_actual_training_accuracy"] is True
    assert diag["independent_test_used"] is False
    assert result["stage"]=="S5_early_stopping"
    assert result["selected_early_stopping"] is False
    assert result["selected_learning_rate"]=="0.001"
    assert abs(result["scores"]["False"]["oof"]-.9221354166666667)<1e-10
    # Authoritative measurements are the original aggregated CSV, never hardcoded.
    x=[int(v["epoch"]) for v in rows]
    acc_train=[100*float(v["training_accuracy_mean"]) for v in rows]
    acc_val=[100*float(v["internal_validation_accuracy_mean"]) for v in rows]
    loss_train=[float(v["training_objective_loss_mean"]) for v in rows]
    loss_val=[float(v["internal_validation_log_loss_mean"]) for v in rows]
    best=float(diag["mean_es10_best_val_epoch"])
    stop=float(diag["mean_es10_would_stop_epoch"])
    min_i=min(range(len(x)),key=lambda k:loss_val[k])
    assert x[min_i]==10 and round(acc_train[-1],3)==99.969
    assert round(acc_val[-1],3)==96.527
    assert 30<best<stop<50
    a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":16,
        "axes.titlesize":20,"axes.labelsize":17,"xtick.labelsize":14,
        "ytick.labelsize":14,"legend.fontsize":13,"figure.facecolor":"white",
        "savefig.facecolor":"white"})
    BLUE="#176b89"
    ORANGE="#d06c1e"
    SLATE="#4f5965"
    RED="#bd513e"
    def style(ax):
        ax.set_xlim(0,120)
        ax.set_xticks(range(0,121,20))
        ax.grid(alpha=.20)
        ax.set_xlabel("Epoch")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    def annotate_acc(ax):
        ax.plot(x,acc_train,lw=2.7,color=BLUE,label="Training accuracy")
        ax.plot(x,acc_val,lw=2.7,color=ORANGE,label="Internal validation accuracy")
        ax.axvline(best,lw=1.7,color=SLATE,ls=":",label=f"Mean ES best epoch {best:.1f}")
        ax.axvline(stop,lw=1.7,color=SLATE,ls="--",label=f"Mean ES would-stop {stop:.1f}")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(88.5,101.3)
        style(ax)
    def annotate_loss(ax):
        ax.plot(x,loss_train,lw=2.7,color=BLUE,label="Training objective loss")
        ax.plot(x,loss_val,lw=2.7,color=ORANGE,label="Internal validation log-loss")
        ax.axvline(x[min_i],lw=1.7,color=RED,ls="--",label=f"Lowest mean validation loss: epoch {x[min_i]}")
        ax.set_ylabel("Loss (lower is better)")
        ax.set_ylim(0,.42)
        style(ax)
    # Main slide: two aligned panels, directly paste into PowerPoint.
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,9),gridspec_kw={"wspace":.22})
    fig.suptitle("MLP Learning Curves | S5 Early Stopping Diagnosis",fontsize=26,weight="bold",y=.965)
    ax1.set_title("A. Accuracy vs Epoch",weight="bold",pad=20)
    annotate_acc(ax1)
    ax1.legend(loc="lower right",frameon=False,fontsize=12)
    ax2.set_title("B. Loss vs Epoch",weight="bold",pad=20)
    annotate_loss(ax2)
    ax2.legend(loc="upper right",frameon=False,fontsize=12)
    fig.text(.5,.12,
             "12 folds (3 seeds x 4 held-out writers)  |  same-fit 90% Train / 10% Internal Validation",
             ha="center",color=SLATE,fontsize=15)
    fig.text(.5,.086,
             "Diagnostic: patience=200 continued to 120 epochs; vertical ES markers are counterfactual averages.",
             ha="center",color=SLATE,fontsize=12)
    fig.text(.5,.056,
             "Decision: ES=False retained from separate Writer-OOF comparison (not selected from this curve).",
             ha="center",color=SLATE,fontsize=12,style="italic")
    fig.subplots_adjust(top=.84,bottom=.22,left=.075,right=.965)
    fig.savefig(a.out/"S5_presentation_LearningCurve_16x9.png",dpi=200)
    plt.close(fig)
    # Standalone widescreen graphs for slides with large projected figures.
    fig,ax=plt.subplots(figsize=(16,9))
    fig.suptitle("S5 | Training vs Validation Accuracy",weight="bold",fontsize=28,y=.96)
    annotate_acc(ax)
    ax.legend(loc="lower right",frameon=False,fontsize=19)
    fig.text(.5,.105,
             "Same-fit 90% Train / 10% Validation  |  3 seeds x 4 writers  |  dashed ES markers: hypothetical best/stop",
             ha="center",fontsize=15,color=SLATE)
    fig.subplots_adjust(top=.865,bottom=.19,left=.10,right=.95)
    fig.savefig(a.out/"S5_presentation_Accuracy_16x9.png",dpi=200)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(16,9))
    fig.suptitle("S5 | Training vs Validation Loss",weight="bold",fontsize=28,y=.96)
    annotate_loss(ax)
    ax.legend(loc="upper right",frameon=False,fontsize=19)
    fig.text(.5,.10,
             "Validation loss rises after epoch 10; accuracy need not fall at the same time.",
             ha="center",fontsize=16,color=SLATE)
    fig.subplots_adjust(top=.865,bottom=.19,left=.10,right=.95)
    fig.savefig(a.out/"S5_presentation_Loss_16x9.png",dpi=200)
    plt.close(fig)
    print(json.dumps({"figures":3,"best":best,"stop":stop,
        "val_loss_min_epoch":x[min_i],
        "train_accuracy_epoch120":acc_train[-1],
        "val_accuracy_epoch120":acc_val[-1],
        "selected_early_stopping":result["selected_early_stopping"],
        "selected_oof":result["scores"]["False"]["oof"]},ensure_ascii=False),flush=True)
if __name__=="__main__":
    main()
