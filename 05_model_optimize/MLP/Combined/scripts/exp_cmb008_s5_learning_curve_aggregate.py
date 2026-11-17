#!/usr/bin/env python3
"""Aggregate all 36 faithful 3-seed×4-writer×3-arm learning traces.
Produce actual (not simulated) learning curves; never select HP using held-out
writer checkpoint maxima. Verify original S5 ES=True stop and final predictions.
"""
from __future__ import annotations
import argparse,csv,json,statistics,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS=(42,43,44)
WRITERS=(41,42,43,44)
ARMS=("es10","internal120","full120")
PAIRS=("D_to_0","B_to_8","8_to_B","0_to_D")
PAIRS_TEXT=("D→0","B→8","8→B","0→D")
EVAL_EPOCHS=(40,60,80,100,120)
def readcsv(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def writecsv(p,rows):
    assert rows
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ratio(x):return f"{100*x:.3f}%"
def acc_at_arm(rows):
    return statistics.mean(float(x["heldout_oof_accuracy"]) for x in rows)
def build():
    ap=argparse.ArgumentParser()
    ap.add_argument("--incoming",type=Path,required=True)
    ap.add_argument("--original-s5",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"figures").mkdir(exist_ok=True)
    datasets={};metadata={};orig_replays=[]
    for variant in ARMS:
        for seed in SEEDS:
            for w in WRITERS:
                folder=a.incoming/f"cmb008-esdiag-s{seed}-w{w}-{variant}"
                assert folder.is_dir(),folder
                trace=readcsv(folder/"epoch_curve.csv")
                meta=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
                assert meta["variant"]==variant and meta["seed"]==seed and meta["heldout"]==w
                assert meta["heldout_writer_used_for_early_stop"] is False
                assert not meta["original_test_used"] and not meta["professor_test_used"]
                assert meta["fixed_hidden_layers"]==[256] and meta["fixed_alpha"]==.0001 and meta["fixed_lr"]==.001
                assert meta["augmentation"]["requested_ratio"]==.3
                assert len(trace)==meta["fit_epochs"]
                assert [int(t["epoch"]) for t in trace]==list(range(1,len(trace)+1))
                if variant=="es10":assert len(trace)<200
                else:assert len(trace)==120
                datasets[(variant,seed,w)]=trace
                metadata[(variant,seed,w)]=meta
                if variant=="es10":
                    old=a.original_s5/f"cmb008-s5-lr0.001-s{seed}"
                    assert old.is_dir(),old
                    fits=[x for x in readcsv(old/"training_time.csv") if int(x["heldout"])==w]
                    assert len(fits)==1
                    orig_n=int(fits[0]["iterations"])
                    original=[x for x in readcsv(old/"oof_predictions.csv") if int(x["writer"])==w]
                    replay=readcsv(folder/"final_heldout_predictions.csv")
                    assert len(original)==len(replay)==640
                    mismatch=sum(oldr["pred"]!=newr["pred"] for oldr,newr in zip(original,replay))
                    exact=(orig_n==meta["fit_epochs"] and mismatch==0)
                    orig_replays.append({"seed":seed,"heldout":w,"orig_stop":orig_n,
                       "replayed_stop":meta["fit_epochs"],"mismatched_predictions":mismatch,
                       "exact_replay":exact})
    assert len(datasets)==36
    # Replay must match previously published S5 before interpreting new diagnostics.
    assert all(x["exact_replay"] for x in orig_replays),f"Original S5 replication failed: {orig_replays}"
    # The ES=True 120-epoch continuation must exactly match ES10 until its stop:
    # only patience changes; split, initialization, Adam momentum and data are equal.
    replay_checks=[]
    for seed in SEEDS:
        for w in WRITERS:
            early=datasets[("es10",seed,w)]
            long=datasets[("internal120",seed,w)]
            assert len(early)<=len(long)
            disagreements=sum(abs(float(x["train_loss"])-float(y["train_loss"]))>1e-9
                or abs(float(x["internal_val_accuracy"])-float(y["internal_val_accuracy"]))>1e-12
                or abs(float(x["heldout_oof_accuracy"])-float(y["heldout_oof_accuracy"]))>1e-12
                or any(int(x[p])!=int(y[p]) for p in PAIRS)
                for x,y in zip(early,long))
            replay_checks.append({"seed":seed,"writer":w,"pre_stop_disagreements":disagreements})
    assert not any(x["pre_stop_disagreements"] for x in replay_checks),"Extended run diverged from original ES before stopping"
    base_sha=metadata[("es10",42,41)]["original_official_sha"]
    for m in metadata.values():
        assert m["original_official_sha"]==base_sha
        assert m["handmade_sha"]==metadata[("es10",42,41)]["handmade_sha"]
    # Per-epoch pooled 3seed×4writer (7680 held-out predictions). No peek at Test.
    aggregate=[]
    for variant in ARMS:
        upto=max(len(datasets[(variant,s,w)]) for s in SEEDS for w in WRITERS)
        for epoch in range(1,upto+1):
            avail=[datasets[(variant,s,w)][epoch-1] for s in SEEDS for w in WRITERS
                    if epoch<=len(datasets[(variant,s,w)])]
            row={"variant":variant,"epoch":epoch,"n_folds":len(avail),
              "mean_train_loss":statistics.mean(float(z["train_loss"]) for z in avail),
              "mean_oof_accuracy":acc_at_arm(avail),
              "mean_internal_validation_accuracy":(
                statistics.mean(float(z["internal_val_accuracy"]) for z in avail)
                if variant!="full120" else ""),
              "mean_internal_validation_loss":(
                statistics.mean(float(z["internal_val_loss"]) for z in avail if z["internal_val_loss"]!="")
                if variant!="full120" and any(z["internal_val_loss"]!="" for z in avail) else "")}
            for p in PAIRS:
                row[p+"_count"]=sum(int(z[p]) for z in avail) if len(avail)==12 else ""
                row[p+"_rate"]=sum(int(z[p]) for z in avail)/480 if len(avail)==12 else ""
            aggregate.append(row)
    writecsv(a.out/"learning_curve_epoch_aggregate.csv",aggregate)
    fold=[]
    for (variant,seed,w),meta in metadata.items():
        fold.append({"variant":variant,"seed":seed,"heldout":w,
          "stop_epoch":meta["fit_epochs"],"best_internal_validation_epoch":meta["best_internal_val_epoch"] or "",
          "best_internal_validation_accuracy":meta["best_internal_val_accuracy"] if meta["best_internal_val_accuracy"] is not None else "",
          "final_restored_heldout_accuracy":meta["final_restored_heldout_accuracy"],
          "last_epoch_heldout_accuracy":meta["last_epoch_heldout_accuracy"],
          "training_fraction":meta["training_fraction_if_internal_val"],
          "fit_seconds":meta["fit_seconds"]})
    writecsv(a.out/"fold_checkpoint_summary.csv",fold)
    def pooled(variant,epoch):
        ss=[datasets[(variant,s,w)][epoch-1] for s in SEEDS for w in WRITERS]
        return {"epoch":epoch,"accuracy":acc_at_arm(ss),
                "pairs":{p:sum(int(x[p]) for x in ss) for p in PAIRS}}
    checkpoints={}
    for arm in ("internal120","full120"):
        checkpoints[arm]={str(k):pooled(arm,k) for k in EVAL_EPOCHS}
    # es10 values use the already restored checkpoints after a normal sklearn fit.
    restored=[]
    for s in SEEDS:
        for w in WRITERS:
            f=a.incoming/f"cmb008-esdiag-s{s}-w{w}-es10"
            restored.extend(readcsv(f/"final_heldout_predictions.csv"))
    assert len(restored)==7680
    y=np.asarray([r["actual"] for r in restored])
    p=np.asarray([r["pred"] for r in restored])
    old_final={"accuracy":float(np.mean(y==p)),
      "pairs":{key:int(np.sum((y==aa)&(p==bb))) for aa,bb,key in (
         ("D","0","D_to_0"),("B","8","B_to_8"),("8","B","8_to_B"),("0","D","0_to_D"))}}
    assert abs(old_final["accuracy"]-.9177083333333333)<1e-12,old_final
    assert [old_final["pairs"][p] for p in PAIRS]==[77,49,49,14],old_final
    stop=[metadata[("es10",s,w)]["fit_epochs"] for s in SEEDS for w in WRITERS]
    best=[metadata[("es10",s,w)]["best_internal_val_epoch"] for s in SEEDS for w in WRITERS]
    summary={"experiment":"EXP-CMB-008 S5 learning curve","report_status":"evaluated",
      "original_s5_exact_replay":True,"pre_stop_long_continuation_exact":True,
      "counts":{"fold_variants":36,"seeds":3,"writers":4,
                 "internal_validation_samples_per_fold":7872},
      "stop_epochs":stop,"best_epochs":best,"mean_stop_epoch":statistics.mean(stop),
      "mean_best_epoch":statistics.mean(best),"restored_es10":old_final,
      "pre_registered_diagnostic_checkpoints":checkpoints,
      "unchanged_hyperparameters":{"hidden_layer_sizes":[256],"alpha":.0001,
                                  "learning_rate_init":.001,"size_ratio":.3,"stroke_strength":.28},
      "data_source":"training and writer-held-out only","independent_test_used":False,
      "replay_s5_audit":orig_replays,"continuation_audit":replay_checks}
    (a.out/"diagnostic_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # Charts: actual raw measurements, not smoothed or fabricated.
    def series(arm,field,complete=False):
        rows=[x for x in aggregate if x["variant"]==arm and (x["n_folds"]==12 or not complete)]
        x=[z["epoch"] for z in rows if z[field]!=""]
        y=[float(z[field]) for z in rows if z[field]!=""]
        return x,y
    colors={"full120":"#156082","internal120":"#df8425","es10":"#49865b"}
    labels={"full120":"100% train (no ES, 120 epochs)",
            "internal120":"90% train + internal val (continued to 120)",
            "es10":"90% train + ES patience 10 (until stopped)"}
    def common(ax,xlabel="Epoch",ylabel=""):
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig,ax=plt.subplots(figsize=(10,5.4))
    for arm in ARMS:
        xs,ys=series(arm,"mean_train_loss")
        ax.plot(xs,ys,color=colors[arm],label=labels[arm],lw=1.8)
    common(ax,ylabel="Mean training log loss");ax.set_title("Learning curve — Training loss (different train fractions)")
    fig.tight_layout();fig.savefig(a.out/"figures"/"01_training_loss.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.4))
    for arm in ("internal120","es10"):
        xs,ys=series(arm,"mean_internal_validation_accuracy")
        ax.plot(xs,ys,color=colors[arm],label=labels[arm],lw=1.8)
    ax.axvline(statistics.mean(best),color="gray",ls="--",label=f"ES10 mean best val epoch {statistics.mean(best):.1f}")
    ax.axvline(statistics.mean(stop),color="black",ls=":",label=f"ES10 mean stop epoch {statistics.mean(stop):.1f}")
    common(ax,ylabel="Internal validation accuracy")
    ax.set_title("Validation learning curve — best checkpoint vs stop")
    fig.tight_layout();fig.savefig(a.out/"figures"/"02_internal_validation_accuracy.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.4))
    for arm in ARMS:
        xs,ys=series(arm,"mean_oof_accuracy")
        ax.plot(xs,ys,color=colors[arm],label=labels[arm],lw=1.8)
    ax.axhline(old_final["accuracy"],color="gray",ls="--",label=f"ES10 restored OOF {ratio(old_final['accuracy'])}")
    common(ax,ylabel="Held-out writer accuracy (mean across available folds)")
    ax.set_title("Held-out writer learning curve (diagnostic ONLY; never choose epochs using OOF)")
    fig.tight_layout();fig.savefig(a.out/"figures"/"03_heldout_writer_oof.png",dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for ax,p,desc in zip(axs.flatten(),PAIRS,PAIRS_TEXT):
        for arm in ("full120","internal120"):
            xs,ys=series(arm,p+"_rate",complete=True)
            ax.plot(xs,[100*y for y in ys],label=labels[arm],color=colors[arm])
        ax.axhline(5,color="gray",ls="--",alpha=.8,label="Goal <5%")
        ax.set_title(desc);ax.set_ylabel("Directional error (% of actual class)")
        ax.grid(alpha=.2)
    for ax in axs[1]:ax.set_xlabel("Epoch")
    handles,lab=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,lab,loc="lower center",ncol=3,fontsize=8)
    fig.suptitle("Held-out writer critical directional errors (12 folds; diagnostic only)")
    fig.tight_layout(rect=(0,.06,1,.95))
    fig.savefig(a.out/"figures"/"04_directional_errors.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.7))
    n=np.arange(12);ax.scatter(n,best,label="best internal validation",s=70,color=colors["internal120"])
    ax.scatter(n,stop,label="actual ES stop",marker="x",s=85,color="black")
    ax.set_xticks(n,[f"{s}-{w}" for s in SEEDS for w in WRITERS],rotation=45)
    ax.set_xlabel("Seed–heldout writer");ax.set_ylabel("Epoch");ax.grid(alpha=.2)
    ax.legend();ax.set_title("Stop epoch vs restored best-validation checkpoint")
    fig.tight_layout();fig.savefig(a.out/"figures"/"05_best_vs_stop_epoch.png",dpi=170);plt.close(fig)
    # Evidence-first report with no hand-selected OOF-optimal checkpoint deployment.
    rows=["# EXP-CMB-008 · 13 Early Stopping Learning Curve 실측 보고","",
      "> **실측 완료:** 3 Seed × 4 Writer × 3 학습 조건 = 36개 실제 모델. 원본 S5 ES10의 종료 Epoch/보류 Writer 예측을 재현했고, 90% Training 장기 학습은 ES10 종료 전까지 동일한 곡선을 재현했다.",
      "> **제한:** 120 Epoch 연장은 진단용이다. held-out OOF 최적 Epoch를 모델 선택에 사용하지 않는다. Original Test·교수님 Test·신규 손글씨 480장 미사용.","",
      "## 비교 설계와 실험 이유","",
      "| 조건 | Training / Validation | 종료 규칙 | 무엇을 구분하는가? |",
      "|---|---|---|---|",
      "| es10 | Training Fold 90% / 내부 Val 10% | 기존 S5 ES=True, patience 10, max_iter 200 | 실제 종료 위치와 검증 최고점 |",
      "| internal120 | es10과 **동일한** 90% / 내부 Val 10% | 동일 초기값/Adam, patience 200, 120 Epoch 추적 | 종료 후 학습이 지속될 때의 변화 (OOF는 진단만) |",
      "| full120 | Training Fold 100% | ES=False, patience 200, 120 Epoch 추적 | 10% 학습데이터 제외가 결과에 미치는 영향 (검증 분할도 변함) |","",
      "## 실제 종료 및 최고 검증 체크포인트","",
      f"- 기존 ES10 평균 **Stop Epoch {statistics.mean(stop):.2f}**, 평균 **최고 Validation Epoch {statistics.mean(best):.2f}**.",
      f"- 원본 S5 ES10의 restored OOF = **{ratio(old_final['accuracy'])}**. 실제 복원 모델 D→0={old_final['pairs']['D_to_0']}, B→8={old_final['pairs']['B_to_8']}, 8→B={old_final['pairs']['8_to_B']}, 0→D={old_final['pairs']['0_to_D']}건 (3 Seed×4 Writer, 실제 클래스당 480회).",
      "- 모든 Fold의 실제 Stop/Best Epoch와 OOF 복원 성능은 [fold_checkpoint_summary.csv](fold_checkpoint_summary.csv) 보존.",
      "",
      "## 실제 Learning Curve — 원본 데이터에서 생성한 PNG","",
      "![Training loss](figures/01_training_loss.png)","",
      "![Internal validation accuracy](figures/02_internal_validation_accuracy.png)","",
      "![Held-out writer OOF accuracy](figures/03_heldout_writer_oof.png)","",
      "![Directional errors](figures/04_directional_errors.png)","",
      "![Best and stop epochs](figures/05_best_vs_stop_epoch.png)","",
      "## 사전 고정한 Epoch별 진단 체크포인트","",
      "| 조건 | Epoch | OOF 정확도 | D→0 | B→8 | 8→B | 0→D |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ("internal120","full120"):
        for epoch in EVAL_EPOCHS:
            rec=checkpoints[arm][str(epoch)]
            rows.append(f"| {arm} | {epoch} | {ratio(rec['accuracy'])} | "+
              " | ".join(str(rec["pairs"][p]) for p in PAIRS)+" |")
    rows+=["","## 어떤 가설을 검증했는가?","",
       "- **H1 조기 종료:** 위 사전 고정 Epoch 60/80/100/120의 internal120 OOF와 각 오류 방향을 원본 ES10 복원 모델과 비교해, 학습 지속이 실질적인 개선을 보이는지 판단한다. **가장 잘 나온 OOF Epoch를 사후 선택하지 않는다.**",
       "- **H2 내부 검증/Writer 일반화 불일치:** Figure 2의 내부 Validation Accuracy 최고 구간과 Figure 3의 보류 Writer OOF 변화, Figure 4의 D→0을 나란히 비교한다.",
       "- **H3 10% 학습데이터 분리:** 동일 Epoch에서 full120(100% 학습)과 internal120(90% 학습)을 비교한다. 내부 검증 사용/랜덤 미니배치의 차이까지 완벽히 분리한 인과 실험은 아니므로 '데이터 감소만의 효과'로 단정하지 않는다.",
       "- **후속 의사결정:** Early Stopping 설정의 최종 채택/기각은 새 모델을 OOF 최고 Epoch에 맞춰 고르는 대신, Training 내부 validation 기반으로 미리 정한 변경안에서 다시 검증해야 한다.",
       "- **과적합 주의:** 개발용 4 Writer를 S1~S5에서 반복 사용했으므로 추가 OOF도 독립 일반화 측정이 아니며, Original Test 및 교수님 Test는 그대로 봉인했다.",
       "",
       "## 데이터와 재현","",
       "- [원본 Epoch별 관측 CSV](learning_curve_epoch_aggregate.csv) · [진단 요약 JSON](diagnostic_summary.json) · [Fold별 Stop/Best](fold_checkpoint_summary.csv)",
       "- 기존 원본 S5 아티팩트 12 Fold와 모든 재학습 Stop·예측 일치, ES continuation도 Stop 전 구간 loss/Validation/OOF 일치 확인.",
       ""]
    (a.out/"report.md").write_text("\n".join(rows),encoding="utf-8")
    print("CURVES_VERIFIED",json.dumps({"mean_stop":statistics.mean(stop),"mean_best":statistics.mean(best),
      "original_oof":old_final["accuracy"],"checkpoints":checkpoints},ensure_ascii=False),flush=True)
if __name__=="__main__":
    build()
