#!/usr/bin/env python3
"""Generate actual same-fit train-vs-validation Accuracy and Loss learning curves."""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS=(42,43,44)
WRITERS=(41,42,43,44)
POINTS=[1]+list(range(5,121,5))
FIELDS=("training_accuracy","internal_validation_accuracy",
        "training_objective_loss","internal_validation_log_loss",
        "heldout_writer_accuracy_diagnostic_only")
PAIRS=("D_to_0","B_to_8","8_to_B","0_to_D")
def readcsv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def writecsv(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--incoming",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    figdir=a.out/"figures";figdir.mkdir(exist_ok=True)
    curves={};manifests={}
    for seed in SEEDS:
        for writer in WRITERS:
            folder=a.incoming/f"cmb008-trainval-s{seed}-w{writer}"
            assert folder.is_dir(),folder
            m=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
            vals=readcsv(folder/"same_run_train_val_curve.csv")
            assert m["seed"]==seed and m["heldout_writer"]==writer
            assert m["total_epochs"]==120 and m["full_training_partition_samples"]==70848
            assert m["internal_validation_samples"]==7872
            assert m["actual_training_accuracy_not_subsampled"]
            assert not m["original_test_used"] and not m["professor_test_used"]
            assert [int(row["epoch"]) for row in vals]==POINTS
            for row in vals:
                assert 0<=float(row["training_accuracy"])<=1
                assert 0<=float(row["internal_validation_accuracy"])<=1
                assert float(row["training_objective_loss"])>0
                assert float(row["internal_validation_log_loss"])>0
            curves[(seed,writer)]=vals
            manifests[(seed,writer)]=m
    assert len(curves)==12
    source=manifests[(42,41)]["source_sha"]
    assert all(m["source_sha"]==source for m in manifests.values())
    rows=[]
    for i,epoch in enumerate(POINTS):
        observed=[curves[k][i] for k in sorted(curves)]
        z={"epoch":epoch,"n_folds":12}
        for key in FIELDS+PAIRS:
            values=[float(x[key]) for x in observed]
            z[key+"_mean"]=statistics.mean(values)
            z[key+"_std"]=statistics.stdev(values)
            if key in PAIRS:
                z[key+"_total"]=int(sum(values))
                z[key+"_rate"]=sum(values)/480
        z["train_minus_val_accuracy_pp"]=100*(z["training_accuracy_mean"]-z["internal_validation_accuracy_mean"])
        rows.append(z)
    writecsv(a.out/"same_run_train_val_curve_aggregate.csv",rows)
    stops=[x["counterfactual_es10_stop_epoch"] for x in manifests.values()]
    best=[x["counterfactual_es10_best_val_epoch"] for x in manifests.values()]
    stop_mean=statistics.mean(stops);best_mean=statistics.mean(best)
    fold_summary=[{"seed":s,"heldout_writer":w,
      "es10_stop_epoch_same_fit":manifests[(s,w)]["counterfactual_es10_stop_epoch"],
      "es10_best_val_epoch_same_fit":manifests[(s,w)]["counterfactual_es10_best_val_epoch"],
      "best_val_epoch_over_120":manifests[(s,w)]["best_validation_epoch_over_120"]}
      for s in SEEDS for w in WRITERS]
    writecsv(a.out/"per_fold_best_stop.csv",fold_summary)
    val_loss_min=min(rows,key=lambda z:z["internal_validation_log_loss_mean"])
    val_acc_max=max(rows,key=lambda z:z["internal_validation_accuracy_mean"])
    x=[z["epoch"] for z in rows]
    train=[100*z["training_accuracy_mean"] for z in rows]
    val=[100*z["internal_validation_accuracy_mean"] for z in rows]
    tloss=[z["training_objective_loss_mean"] for z in rows]
    vloss=[z["internal_validation_log_loss_mean"] for z in rows]
    oof=[100*z["heldout_writer_accuracy_diagnostic_only_mean"] for z in rows]
    def guide(ax):
        ax.axvline(best_mean,color="#8b5e34",ls=":",lw=1.4,label=f"Mean ES10 best-val epoch {best_mean:.1f}")
        ax.axvline(stop_mean,color="#555555",ls="--",lw=1.4,label=f"Mean ES10 would-stop epoch {stop_mean:.1f}")
        ax.set_xlim(0,120);ax.grid(alpha=.23)
    def export(fig,name):
        fig.tight_layout();fig.savefig(figdir/name,dpi=170,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.8))
    ax.plot(x,train,"o-",ms=3,lw=2,color="#167c86",label="Full training partition Accuracy (70,848)")
    ax.plot(x,val,"o-",ms=3,lw=2,color="#d47620",label="Internal validation Accuracy (7,872)")
    guide(ax);ax.set(xlabel="Epoch",ylabel="Accuracy (%)",title="S5 same-fit Learning Curve — Training vs Validation Accuracy")
    ax.legend(fontsize=8);export(fig,"01_training_vs_validation_accuracy.png")
    fig,ax=plt.subplots(figsize=(11,5.8))
    ax.plot(x,tloss,"o-",ms=3,lw=2,color="#167c86",label="Training Objective Loss")
    ax.plot(x,vloss,"o-",ms=3,lw=2,color="#d47620",label="Validation Cross-Entropy Loss")
    guide(ax);ax.set(xlabel="Epoch",ylabel="Loss (lower is better)",title="S5 same-fit Learning Curve — Training vs Validation Loss")
    ax.legend(fontsize=8);export(fig,"02_training_vs_validation_loss.png")
    fig,ax=plt.subplots(figsize=(11,5.8))
    ax.plot(x,tloss,"o-",ms=3,lw=2,color="#167c86",label="Training objective loss")
    ax.plot(x,vloss,"o-",ms=3,lw=2,color="#d47620",label="Validation cross-entropy")
    ax.axvline(val_loss_min["epoch"],color="#a12f34",ls="-.",lw=1.4,label=f"Lowest mean validation loss: epoch {val_loss_min['epoch']}")
    guide(ax);ax.set(xlim=(5,120),ylim=(0,max(vloss[1:])*1.13),xlabel="Epoch",ylabel="Loss",
             title="Overfitting diagnostic — Validation loss rises")
    ax.legend(fontsize=8);export(fig,"03_overfit_loss_zoom.png")
    fig,ax=plt.subplots(figsize=(10,5.4))
    ax.plot(x,[z["train_minus_val_accuracy_pp"] for z in rows],"o-",ms=3,lw=2,color="#4b6c91",label="Training minus validation accuracy")
    guide(ax);ax.set(xlabel="Epoch",ylabel="Gap (percentage points)",title="Generalization gap by Epoch")
    ax.legend(fontsize=8);export(fig,"04_generalization_accuracy_gap.png")
    fig,ax=plt.subplots(figsize=(11,5.6))
    ax.plot(x,val,"o-",ms=3,lw=2,label="Internal validation Accuracy",color="#d47620")
    ax.plot(x,oof,"o-",ms=3,lw=2,label="Writer-held-out OOF (diagnostic only)",color="#7159a2")
    guide(ax);ax.set(xlabel="Epoch",ylabel="Accuracy (%)",title="Internal Validation vs Writer-held-out OOF")
    ax.legend(fontsize=8);export(fig,"05_validation_vs_oof.png")
    fig,ax=plt.subplots(figsize=(10,5.5))
    for name,label in zip(PAIRS,("D→0","B→8","8→B","0→D")):
        ax.plot(x,[100*z[name+"_rate"] for z in rows],"o-",ms=2,lw=1.6,label=label)
    ax.axhline(5,color="#555555",ls=":",label="Goal: <5%")
    ax.set(xlabel="Epoch",ylabel="Directional error rate (%)",title="Writer-held-out errors: four sensitive directions")
    ax.grid(alpha=.23);ax.legend(fontsize=8);export(fig,"06_writer_directional_errors.png")
    meta={"experiment":"EXP-CMB-008 Same-fit S5 Learning Curve",
      "folds":12,"epochs_measured":POINTS,"training_samples_per_fold":70848,
      "internal_val_samples_per_fold":7872,"full_actual_training_accuracy":True,
      "mean_es10_would_stop_epoch":stop_mean,"mean_es10_best_val_epoch":best_mean,
      "best_aggregate_validation_loss_epoch":val_loss_min["epoch"],
      "best_aggregate_validation_accuracy_epoch":val_acc_max["epoch"],
      "study_diagnostic_only":True,"independent_test_used":False}
    (a.out/"diagnostic_summary.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    def percent(z):return f"{100*z:.3f}%"
    report=["# EXP-CMB-008 · 16 Same-fit Training Accuracy / Validation Accuracy","",
      "이번 그래프의 Training Accuracy와 Validation Accuracy는 동일한 모델·Epoch·실행에서 실제 70,848개 Train과 7,872개 Internal Validation으로 계산했다. 각 Epoch별 12 Fold 평균이며, Training Accuracy는 부분 샘플이 아니라 전체 학습 집합이다.","",
      "## 1. Training vs Validation Accuracy","",
      "![Training and Validation Accuracy](../../results/Hyperparameter_Optimization/S5_SameFit_TrainVal/figures/01_training_vs_validation_accuracy.png)","",
      "## 2. Training vs Validation Loss","",
      "![Training and Validation Loss](../../results/Hyperparameter_Optimization/S5_SameFit_TrainVal/figures/02_training_vs_validation_loss.png)","",
      "![Loss overfit zoom](../../results/Hyperparameter_Optimization/S5_SameFit_TrainVal/figures/03_overfit_loss_zoom.png)","",
      "## 3. Generalization gap","",
      "![Accuracy gap](../../results/Hyperparameter_Optimization/S5_SameFit_TrainVal/figures/04_generalization_accuracy_gap.png)","",
      "## 4. Internal Validation vs Writer-held-out OOF","",
      "![Validation and OOF](../../results/Hyperparameter_Optimization/S5_SameFit_TrainVal/figures/05_validation_vs_oof.png)","",
      "![Four confusion directions](../../results/Hyperparameter_Optimization/S5_SameFit_TrainVal/figures/06_writer_directional_errors.png)","",
      "## 실제 측정값","",
      "| Epoch | Training Accuracy | Internal Validation Accuracy | Training Objective Loss | Validation Cross-Entropy Loss | Train-Val gap |",
      "|---:|---:|---:|---:|---:|---:|"]
    for z in rows:
        report.append(f"| {z['epoch']} | {percent(z['training_accuracy_mean'])} | {percent(z['internal_validation_accuracy_mean'])} | {z['training_objective_loss_mean']:.6f} | {z['internal_validation_log_loss_mean']:.6f} | {z['train_minus_val_accuracy_pp']:+.3f} pp |")
    report+=["","## 동일 실행에서 재구성한 Early Stopping 판단",
      f"- patience=10일 때의 동일 실행 경로에서 계산한 평균 중지 Epoch: **{stop_mean:.2f}**, 평균 복원 Best Validation Epoch: **{best_mean:.2f}**.",
      f"- 관측된 5-Epoch 간격에서 평균 Validation Loss 최저점: **{val_loss_min['epoch']} Epoch**.",
      f"- 평균 Validation Accuracy 최고 관측점: **{val_acc_max['epoch']} Epoch**.",
      "- Early Stopping은 Accuracy 기준으로 체크포인트를 선택하므로, Loss 기준 과적합 시작점과 같은 Epoch일 필요가 없다.",
      "- Training Objective Loss에는 L2 규제가 포함되며 Validation Log-loss는 예측 확률의 Cross-Entropy이므로 절대 수준을 동일 지표처럼 단순 비교하지 않는다.",
      "- 같은 Fold의 같은 학습 경로에서 Train/Val를 추적한 자료여서 서로 다른 실행 결과를 정렬해 발생하는 착시를 줄였다.",
      "- OOF는 오직 진단용이다. OOF의 최고 Epoch를 선택하면 새로운 Writer 데이터에 대한 선택 과적합을 유발할 수 있다.",
      "- 독립 Test, 교수님 Test, 신규 480장 미사용. ES=False는 아직 최종 재검증 전 잠정 선택값이다.",
      "",
      "## 원본", "- [집계 CSV](../../results/Hyperparameter_Optimization/S5_SameFit_TrainVal/same_run_train_val_curve_aggregate.csv)",
      "- [Fold별 Best/Stop Epoch](../../results/Hyperparameter_Optimization/S5_SameFit_TrainVal/per_fold_best_stop.csv)",
      "- [요약 JSON](../../results/Hyperparameter_Optimization/S5_SameFit_TrainVal/diagnostic_summary.json)",""]
    (a.out/"report.md").write_text("\n".join(report),encoding="utf-8")
    print("SAME_FIT_CURVES_DONE",json.dumps({"mean_es10_best":best_mean,
      "mean_es10_stop":stop_mean,"loss_min_epoch":val_loss_min["epoch"],
      "end_training_accuracy":rows[-1]["training_accuracy_mean"],
      "end_internal_val_accuracy":rows[-1]["internal_validation_accuracy_mean"]}),flush=True)

if __name__=="__main__":
    main()
