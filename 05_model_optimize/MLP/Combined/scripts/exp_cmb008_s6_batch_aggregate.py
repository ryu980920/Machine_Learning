#!/usr/bin/env python3
"""S6: audit all 36 paired batch-size fits, evaluate frozen CMB008 gates.

Final deployed/baseline HP are NEVER automatically changed. Outputs a
research-only recommended S6 candidate; user must approve any HP adoption.
"""
from __future__ import annotations
import argparse,csv,json,statistics
from collections import Counter,defaultdict
from pathlib import Path

BATCHES=(200,128,64)
SEEDS=(42,43,44)
WRITERS=(41,42,43,44)
PAIRS=("D_to_0","B_to_8","8_to_B","0_to_D")
FOCUS=("0","8","B","D")
def csvread(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
    assert rows
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]))
        wr.writeheader();wr.writerows(rows)
def acc_str(x):return f"{100*x:.3f}%"
def score_for(batch,artifacts):
    records=[(s,w,artifacts[(batch,s,w)]) for s in SEEDS for w in WRITERS]
    pred=[(s,w,row) for s,w,d in records for row in d["pred"]]
    assert len(pred)==7680
    n_correct=sum(r["actual"]==r["pred"] for _,_,r in pred)
    assert all(Counter(r["actual"] for _,_,r in pred)[c]==480 for c in "0123456789ABCDEF")
    oof=n_correct/7680
    accuracy_seed=[sum(r["actual"]==r["pred"] for z,w,r in pred if z==s)/2560 for s in SEEDS]
    writers={str(w):sum(r["actual"]==r["pred"] for z,z_w,r in pred if z_w==w)/1920 for w in WRITERS}
    val=statistics.mean(float(d["m"]["official_val_accuracy"]) for _,_,d in records)
    pairs={key:sum(r["actual"]==actual and r["pred"]==predicted for _,_,r in pred)
      for key,actual,predicted in
      (("D_to_0","D","0"),("B_to_8","B","8"),("8_to_B","8","B"),("0_to_D","0","D"))}
    classes={cl:sum(r["actual"]==cl and r["pred"]!=cl for _,_,r in pred)/480 for cl in "0123456789ABCDEF"}
    family=defaultdict(lambda:[0,0])
    for _,_,d in records:
        for row in d["flip"]:
            k=(row["dataset"],row["actual"],float(row["scale"]))
            family[k][0]+=int(row["flip_count"])
            family[k][1]+=int(row["n"])
    assert len(family)==2*16*2
    flips={ds:max(count/total for (d,cl,sc),(count,total) in family.items() if d==ds)
           for ds in ("OOF","OfficialVal")}
    return {"batch_size":batch,"oof":oof,"seed_std":statistics.pstdev(accuracy_seed),
       "official_val":val,"writers":writers,"pairs":pairs,
       "classerror":classes,"flip":flips,
       "mean_epochs":statistics.mean(int(d["m"]["n_iter"]) for _,_,d in records),
       "mean_fit_seconds":statistics.mean(float(d["m"]["fit_seconds"]) for _,_,d in records),
       "warnings":sum(bool(d["m"]["warnings"]) for _,_,d in records)}
def gates(base,candidate):
    return {
      "oof_at_least_baseline":candidate["oof"]>=base["oof"]-1e-12,
      "official_val_drop_at_most_0p5pp":candidate["official_val"]>=base["official_val"]-.005-1e-12,
      "writer_drop_at_most_1pp":all(candidate["writers"][w]>=base["writers"][w]-.01-1e-12 for w in base["writers"]),
      "class_0_8_B_D_error_nonincrease":all(candidate["classerror"][cl]<=base["classerror"][cl]+1e-12 for cl in FOCUS),
      "worst_class_family_flip_at_most_plus_5pp":all(candidate["flip"][ds]<=base["flip"][ds]+.05+1e-12 for ds in ("OOF","OfficialVal")),
      "D_to_0_nonincrease":candidate["pairs"]["D_to_0"]<=base["pairs"]["D_to_0"],
      "B_to_8_nonincrease":candidate["pairs"]["B_to_8"]<=base["pairs"]["B_to_8"],
      "8_to_B_nonincrease":candidate["pairs"]["8_to_B"]<=base["pairs"]["8_to_B"],
      "0_to_D_nonincrease":candidate["pairs"]["0_to_D"]<=base["pairs"]["0_to_D"]}
def target(m):return all(m["pairs"][key]<=23 for key in PAIRS)
def compare_predictions(ref,other):
    fixed=induced=0
    for seed in SEEDS:
        for writer in WRITERS:
            aa=ref[(seed,writer)]["pred"]
            bb=other[(seed,writer)]["pred"]
            assert len(aa)==len(bb)==640
            for i in range(640):
                assert aa[i]["actual"]==bb[i]["actual"]
                assert int(aa[i]["index"])==int(bb[i]["index"])==i
                old=aa[i]["actual"]==aa[i]["pred"]
                new=bb[i]["actual"]==bb[i]["pred"]
                fixed+=int(not old and new)
                induced+=int(old and not new)
    return {"fixed":fixed,"induced":induced,"net_gain":fixed-induced}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--incoming",type=Path,required=True)
    p.add_argument("--s5-original",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    artifacts={}
    original=json.loads(a.s5_original.read_text(encoding="utf-8"))
    assert original["stage"]=="S5_early_stopping"
    assert original["selected_early_stopping"] is False
    orig=original["scores"]["False"]
    observed_dirs={k:orig["pairs"][k.replace("_to_","->")] for k in PAIRS}
    assert orig["oof"]==0.9221354166666667
    for batch in BATCHES:
        for seed in SEEDS:
            for writer in WRITERS:
                d=a.incoming/f"cmb008-s6-bs{batch}-s{seed}-w{writer}"
                assert d.is_dir(),d
                m=json.loads((d/"manifest.json").read_text(encoding="utf-8"))
                assert m["stage"]=="S6_batch_size" and m["batch_size"]==batch
                assert m["seed"]==seed and m["heldout"]==writer
                assert m["n_training"]==78720 and m["n_heldout"]==640 and m["n_official_val"]==19200
                assert not m["independent_test_used"] and not m["professor_test_used"]
                assert not m["new_handmade_480_used"]
                assert m["writer_heldout_not_used_to_train"]
                assert m["handmade_train_preprocess"]=="CoM-only"
                assert m["heldout_and_official_val_preprocess"]=="CoM-only"
                assert m["frozen"]=={"hidden_layer_sizes":[256],"alpha":.0001,
                    "learning_rate_init":.001,"solver":"adam","activation":"relu",
                    "early_stopping":False,"max_iter":200,"tol":.0001,"n_iter_no_change":10}
                assert abs(m["augmentation"]["actual_ratio"]-.3)<.002
                preds=csvread(d/"oof_predictions.csv")
                assert len(preds)==640
                assert all(int(r["seed"])==seed and int(r["writer"])==writer and int(r["batch_size"])==batch for r in preds)
                flip=csvread(d/"class_family_flip.csv")
                assert len(flip)==64
                for row in flip:
                    assert row["dataset"] in ("OOF","OfficialVal") and float(row["scale"]) in (.8,1.2)
                    assert 0<=int(row["flip_count"])<=int(row["n"])
                assert all(m["focus_class"][cl]["n"]==40 for cl in "0123456789ABCDEF")
                artifacts[(batch,seed,writer)]={"m":m,"pred":preds,"flip":flip}
    assert len(artifacts)==36
    ref=artifacts[(200,42,41)]["m"]
    assert all(d["m"]["official_sha256"]==ref["official_sha256"] and
      d["m"]["handmade_sha256"]==ref["handmade_sha256"]
      for d in artifacts.values())
    metrics={bs:score_for(bs,artifacts) for bs in BATCHES}
    baseline=metrics[200]
    guard={bs:gates(baseline,metrics[bs]) for bs in BATCHES}
    assert all(guard[200].values())
    safe=[bs for bs in BATCHES if all(guard[bs].values())]
    strict=[bs for bs in safe if target(metrics[bs])]
    pool=strict or safe
    top=max(metrics[bs]["oof"] for bs in pool)
    near=[bs for bs in pool if metrics[bs]["oof"]>=top-.002-1e-12]
    # Existing CMB008 rule: within 0.2 pp, prioritize D->0,
    # then the other 3 confusions, then OOF, seed robustness and Val.
    near.sort(key=lambda bs:(metrics[bs]["pairs"]["D_to_0"],
      sum(metrics[bs]["pairs"][k] for k in PAIRS if k!="D_to_0"),
      -metrics[bs]["oof"],metrics[bs]["seed_std"],
      -metrics[bs]["official_val"],-bs))
    winner=near[0]
    same_oof=abs(baseline["oof"]-orig["oof"])<1e-12
    same_dirs=all(baseline["pairs"][p]==observed_dirs[p] for p in PAIRS)
    original_audit={"original_S5_es_false_oof":orig["oof"],
       "new_rerun_batch200_oof":baseline["oof"],
       "exact_oof_match":same_oof,
       "original_S5_four_confusions":observed_dirs,
       "rerun_batch200_four_confusions":baseline["pairs"],
       "exact_four_confusions_match":same_dirs}
    a.out.mkdir(parents=True,exist_ok=True)
    summary={"experiment":"EXP-CMB-008","stage":"S6_Batch_Size",
       "batch_sizes":list(BATCHES),"n_models":36,"n_seeds":3,"n_writers":4,
       "selected_candidate_research_only":winner,
       "selected_is_not_deployed":True,
       "recommended_condition_requires_user_approval":True,
       "strict_below_5pct_candidates":strict,
       "nonregression_candidates":safe,
       "selection_near_0p2pp":near,
       "same_dataset_SHA_across_all_36":True,
       "official_test_used":False,"professor_test_used":False,
       "new_handmade_480_used":False,"original_baseline_reproduction_audit":original_audit,
       "metrics":{str(bs):metrics[bs] for bs in BATCHES},
       "gates":{str(bs):guard[bs] for bs in BATCHES}}
    (a.out/"S6_results.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    rows=[]
    for bs in BATCHES:
        m=metrics[bs]
        rows.append({"batch_size":bs,"oof_accuracy":m["oof"],
          "official_val_accuracy":m["official_val"],
          "seed_std":m["seed_std"],"mean_epochs":m["mean_epochs"],
          "mean_fit_seconds":m["mean_fit_seconds"],
          "D_to_0":m["pairs"]["D_to_0"],
          "B_to_8":m["pairs"]["B_to_8"],
          "8_to_B":m["pairs"]["8_to_B"],
          "0_to_D":m["pairs"]["0_to_D"],
          "class_0_error":m["classerror"]["0"],
          "class_8_error":m["classerror"]["8"],
          "class_B_error":m["classerror"]["B"],
          "class_D_error":m["classerror"]["D"],
          "oof_max_flip":m["flip"]["OOF"],
          "official_val_max_flip":m["flip"]["OfficialVal"],
          "passes_guards":all(guard[bs].values()),
          "failed_guards":";".join(k for k,v in guard[bs].items() if not v),
          "target_all_four_below_5pct":target(m),
          "warnings":m["warnings"]})
    csvwrite(a.out/"S6_batch_size_comparison.csv",rows)
    paired={str(bs):compare_predictions(
      {(s,w):artifacts[(200,s,w)] for s in SEEDS for w in WRITERS},
      {(s,w):artifacts[(bs,s,w)] for s in SEEDS for w in WRITERS}) for bs in (128,64)}
    (a.out/"S6_paired_corrections.json").write_text(json.dumps(paired,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    notes=["# EXP-CMB-008 · S6 Batch Size 200/128/64 실측 비교",
      "",
      "> **결론은 개발용 추천이며 최종 HP 적용/모델 배포는 하지 않음.** Early Stopping=False는 유지. 3 Seed × 4 held-out Writer × 3 Batch Size = **36개 새 학습**.",
      "",
      "## 목표 및 고정 조건",
      "",
      "- 강의자료 09B 13번 슬라이드의 작은 Batch Size 비교 제안 검증. 바뀌는 항목은 **batch_size** 뿐이다.",
      "- 전처리 Size30% / Stroke28% / CoM, MLP (256,), ReLU, Adam, alpha=1e-4, lr_init=.001, early_stopping=False, max_iter=200. Official Train 76,800 + 다른 Writer 1,920장으로 학습, 보류 Writer 640장과 OfficialVal 19,200장 평가.",
      "- Original Test, 교수님 Test, 신규 480장 사용하지 않음. 원본 SHA·평가 파이프라인 동일.",
      "",
      "## 정량 결과",
      "",
      "| Batch Size | Writer OOF | OfficialVal | D→0 | B→8 | 8→B | 0→D | 가드레일 |",
      "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for bs in BATCHES:
        m=metrics[bs]
        errors=m["pairs"]
        notes.append(f"| {bs} | {acc_str(m['oof'])} | {acc_str(m['official_val'])} | {errors['D_to_0']} | {errors['B_to_8']} | {errors['8_to_B']} | {errors['0_to_D']} | "+("통과" if all(guard[bs].values()) else "실패: "+", ".join(k for k,v in guard[bs].items() if not v))+" |")
    notes+=["","### Writer별 보류 OOF 및 학습 비용",
       "","| Batch | W41 | W42 | W43 | W44 | 평균 학습 Epoch | 평균 Fit Time (초) | 수렴 경고 |",
       "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for bs in BATCHES:
        m=metrics[bs]
        notes.append("| "+str(bs)+" | "+" | ".join(acc_str(m["writers"][str(w)]) for w in WRITERS)+
             f" | {m['mean_epochs']:.1f} | {m['mean_fit_seconds']:.1f} | {m['warnings']} |")
    notes+=["","### 주요 클래스별 전체 오분류율",
       "","| Batch | 0 | 8 | B | D | 최악 Writer OOF | 클래스별 최대 OOF Flip |",
       "|---:|---:|---:|---:|---:|---:|---:|"]
    for bs in BATCHES:
        m=metrics[bs]
        notes.append(f"| {bs} | "+" | ".join(acc_str(m["classerror"][cl]) for cl in FOCUS)+
          " | "+acc_str(min(m["writers"].values()))+" | "+acc_str(m["flip"]["OOF"])+" |")
    notes+=["","## 선정 규칙과 판단",
      "",
      "- **1순위:** D→0 / B→8 / 8→B / 0→D 4방향 각각 <5% (480개에서 23건 이하), 동시에 모든 가드레일 통과.",
      "- **미달 시:** Writer OOF가 기준선보다 낮지 않고, D→0 등 네 방향과 class 0/8/B/D 오류가 증가하지 않으며, Writer별 OOF 최소 1%p/OfficialVal 최대 0.5%p 손실 제한 및 최대 클래스 가족 Flip 증가 5%p 이내의 후보만 선정.",
      "- **가드레일 통과 후보 중** OOF 최고 정확도와 0.2%p 이내인 후보는 D→0을 우선하고 나머지 세 방향 합계로 순서를 매긴다.",
      f"- **검증 결과:** 4방향 엄격 목표 통과 후보: {strict or '없음'}. 안전 가드 통과 후보: {safe}. **개발용 추천 Batch Size: {winner}**.",
      "- 후보의 OOF가 높더라도 한 방향·취약 클래스가 악화되면 전체 정확도만으로 변경하지 않는다.",
      "",
      "## 기준선 재실행 감사",
      "",
      f"- 원본 S5 ES=False OOF: **{acc_str(orig['oof'])}**. 이번 새 Batch=200 OOF: **{acc_str(baseline['oof'])}**. 원본과 전체 정확도 일치 여부: **{same_oof}**.",
      f"- 4방향 오류 완전 일치 여부: **{same_dirs}**. 기존 연구의 분할·환경·실행 차이에 따른 재현성 차이가 나타나면 수치를 숨기지 않는다.",
      "- **이번 S6 후보 비교·선정은 새 36개 실행에 포함된 동일 Batch=200 기준선에 상대 비교했다.** 원본 S5와 수치가 다르더라도 이를 원본 S5를 덮어쓴 결과로 취급하지 않는다.",
      "",
      "## 동일 7,680건 예측에서 교정과 신규 오류",
      "",
      "| 후보 vs Batch=200 | 기존 오답 교정 | 기존 정답이 오답으로 변함 | 순개선 |",
      "|---:|---:|---:|---:|"]
    for bs in (128,64):
        q=paired[str(bs)]
        notes.append(f"| {bs} | {q['fixed']} | {q['induced']} | {q['net_gain']:+d} |")
    notes+=["","## 후속 결정",
      "",
      f"- **추천 조건: Batch Size {winner}** (개발용 분석 결과). **현재 기준선 200의 HP와 모델을 자동으로 바꾸지 않는다.**",
      "- 기존 Early Stopping=False 확정 상태와 무관하게, 이번 실험은 batch_size만 변경했다.",
      "- 동일 보류 Writer 4명을 반복 활용한 개발용 OOF 선택 편향 가능성은 남아 있으며, 독립 Test나 최종 배포 전에 별도 승인 및 검증이 필요하다.",
      "",
      "## 원본 추적",
      "",
      "- [정량 CSV](../../results/Hyperparameter_Optimization/S6_Batch_Size/S6_batch_size_comparison.csv)",
      "- [종합·가드레일 JSON](../../results/Hyperparameter_Optimization/S6_Batch_Size/S6_results.json)",
      "- [오답 교정/유발 JSON](../../results/Hyperparameter_Optimization/S6_Batch_Size/S6_paired_corrections.json)",
      "- [S6 학습 코드](../../scripts/exp_cmb008_s6_batch_size.py)",
      "- [S6 집계 코드](../../scripts/exp_cmb008_s6_batch_aggregate.py)",
      ""]
    (a.out/"S6_Results.md").write_text("\n".join(notes),encoding="utf-8")
    print("BATCH_COMPARE_DONE",json.dumps({
       "recommended_batch":winner,
       "batch200_reproduced_original_oof":same_oof,
       "batch200_reproduced_original_four_dirs":same_dirs,
       "metrics":{str(k):{"oof":v["oof"],"official":v["official_val"],"pairs":v["pairs"],
          "guards":guard[k]} for k,v in metrics.items()}},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
