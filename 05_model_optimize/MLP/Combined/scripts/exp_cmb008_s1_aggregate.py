#!/usr/bin/env python3
"""EXP-CMB-008 S1: compare archived width100 vs new widths on identical writer OOF.
No fitting or Test use. v2: aim four directional error rates <5%, maximize OOF accuracy, guard against overfitting.
"""
import argparse,csv,json,statistics
from collections import Counter,defaultdict
from pathlib import Path

SEEDS=(42,43,44)
WRITERS=(41,42,43,44)
WIDTHS=(64,100,128,256)
WATCH=(("0","D"),("8","B"),("D","0"),("B","8"),("5","F"),("7","9"),("7","A"),("4","F"))
BASE=Path("05_model_optimize/MLP/Combined/results/Balanced_Scale_Stroke/stage_low20to30/raw/scale30_stroke28")
def readcsv(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def pct(v):return f"{100*v:.3f}%"
def read_arm(repo,incoming,width):
    d={}
    for s in SEEDS:
        p=repo/BASE/("seed_"+str(s)) if width==100 else incoming/("cmb008-w"+str(width)+"-s"+str(s))
        assert p.is_dir(),p
        m=json.loads((p/"manifest.json").read_text())
        assert m["arm"]=="scale30_stroke28" and int(m["seed"])==s
        assert m["augmentation_seed"]==20261116 and m["official_train_only_augmentation"]
        assert m["handmade_train_preprocess"]==m["heldout_and_official_val_preprocess"]=="CoM-only"
        assert not (m["original_test_used"] or m["professor_test_used"] or m["new_handmade_480_used"])
        if width==100:
            assert m["experiment"]=="EXP-CMB-005" and m["no_mlp_hyperparameter_search"]
        else:
            assert m["experiment"]=="EXP-CMB-008" and m["stage"]=="S1_width" and m["width"]==width
            assert m["only_changed_parameter"]=="hidden_layer_sizes"
            assert m["fixed_params"]=={"activation":"relu","solver":"adam","alpha":.0001,"learning_rate_init":.001,"max_iter":200,"early_stopping":False}
        oof=readcsv(p/"oof_predictions.csv")
        assert len(oof)==2560
        keys=[(int(x["writer"]),int(x["index"])) for x in oof]
        assert len(set(keys))==2560 and {w for w,i in keys}==set(WRITERS)
        vals=[r for r in readcsv(p/"fold_metrics.csv") if r["dataset"]=="OfficialVal" and r["group"]=="all"]
        assert len(vals)==4 and all(int(x["n"])==19200 for x in vals)
        scales=[x for x in readcsv(p/"per_class_and_scale.csv") if x["dataset"] in ("OOF","OfficialVal") and float(x["scale"]) in (.8,1.2)]
        assert scales
        times=readcsv(p/"training_time.csv")
        assert len(times)==4
        d[s]={"m":m,"oof":oof,"val":vals,"scales":scales,"times":times}
    return d
def summarize(width,d):
    oof=[r for s in SEEDS for r in d[s]["oof"]]
    correct=lambda z:sum(x["actual"]==x["pred"] for x in z)
    n=len(oof);assert n==7680
    seedacc=[correct(d[s]["oof"])/2560 for s in SEEDS]
    val=statistics.mean(float(v["accuracy"]) for s in SEEDS for v in d[s]["val"])
    counts=Counter(x["actual"] for x in oof)
    assert len(counts)==16 and all(n==480 for n in counts.values())
    pairs={a+"->"+b:sum(x["actual"]==a and x["pred"]==b for x in oof) for a,b in WATCH}
    classerror={c:sum(x["actual"]==c and x["pred"]!=c for x in oof)/counts[c] for c in counts}
    writers={str(w):correct([x for x in oof if int(x["writer"])==w])/1920 for w in WRITERS}
    f=defaultdict(lambda:[0.,0])
    for s in SEEDS:
        for r in d[s]["scales"]:
            k=(r["dataset"],r["actual"],r["scale"])
            nn=int(r["n"]); f[k][0]+=nn*float(r["family_flip"]);f[k][1]+=nn
    assert len(f)==64
    flips={ds:max(v[0]/v[1] for k,v in f.items() if k[0]==ds) for ds in ("OOF","OfficialVal")}
    ts=[x for s in SEEDS for x in d[s]["times"]]
    return {"width":width,"oof":correct(oof)/n,"seed_std":statistics.pstdev(seedacc),
       "val":val,"pairs":pairs,"classerror":classerror,"writers":writers,"flip":flips,
       "mean_n_iter":statistics.mean(int(x["iterations"]) for x in ts),
       "mean_fit_seconds":statistics.mean(float(x["fit_seconds"]) for x in ts),
       "warnings":sum(bool(x["warnings"]) for x in ts)}
TARGET_PAIRS=("D->0","B->8","8->B","0->D")
def target_under_5percent(n):
    # Each actual class contributes 160 images/seed x 3 seeds = 480.
    # Strict <5% means at most 23 (NOT 24) errors per direction.
    return all(n["pairs"][p]<24 for p in TARGET_PAIRS)
def check(b,n):
    # Updated user criterion before comparing S1 results, archived as plan 2-B.
    return {
      "oof_accuracy_at_least_baseline":n["oof"]>=b["oof"]-1e-12,
      "official_val_drop_le_0p5pp":n["val"]>=b["val"]-.005-1e-12,
      "writer_drop_le_1pp":all(n["writers"][k]>=b["writers"][k]-.01-1e-12 for k in b["writers"]),
      "class_0_8_B_D_error_no_increase":all(n["classerror"][k]<=b["classerror"][k]+1e-12 for k in ("0","8","B","D")),
      "max_class_family_flip_increase_le_5pp":all(n["flip"][k]<=b["flip"][k]+.05+1e-12 for k in ("OOF","OfficialVal")),
      "D_to_0_no_increase":n["pairs"]["D->0"]<=b["pairs"]["D->0"],
      "B_to_8_no_increase":n["pairs"]["B->8"]<=b["pairs"]["B->8"],
      "8_to_B_no_increase":n["pairs"]["8->B"]<=b["pairs"]["8->B"],
      "0_to_D_no_increase":n["pairs"]["0->D"]<=b["pairs"]["0->D"]}
def paired(b,n):
    fixed=induced=0
    for s in SEEDS:
        x={(int(r["writer"]),int(r["index"])):r for r in b[s]["oof"]}
        y={(int(r["writer"]),int(r["index"])):r for r in n[s]["oof"]}
        assert x.keys()==y.keys()
        for k in x:
            assert x[k]["actual"]==y[k]["actual"]
            old=x[k]["actual"]==x[k]["pred"]
            new=y[k]["actual"]==y[k]["pred"]
            fixed+=not old and new
            induced+=old and not new
    return fixed,induced
def main():
    a=argparse.ArgumentParser();a.add_argument("--repo",type=Path,default=Path("."))
    a.add_argument("--incoming",type=Path,required=True);a.add_argument("--out",type=Path,required=True)
    args=a.parse_args()
    d={w:read_arm(args.repo,args.incoming,w) for w in WIDTHS}
    for w in WIDTHS:
        for s in SEEDS:
            assert d[w][s]["m"]["official_sha256"]==d[100][s]["m"]["official_sha256"]
            assert d[w][s]["m"]["handmade_sha256"]==d[100][s]["m"]["handmade_sha256"]
    m={w:summarize(w,d[w]) for w in WIDTHS}
    assert abs(m[100]["oof"]-.9071614583333334)<1e-9
    assert abs(m[100]["val"]-.961171875)<1e-9
    checks={w:check(m[100],m[w]) for w in WIDTHS}
    target_meets=[w for w in WIDTHS if target_under_5percent(m[w])]
    primary=[w for w in target_meets if all(checks[w].values())]
    # If 5% target is not achieved, fall back to best OOF among candidates that
    # do not worsen any protected error, Writer, OfficialVal or class stability.
    fallback=[w for w in WIDTHS if all(checks[w].values())]
    pool=primary if primary else fallback
    assert 100 in fallback, "baseline must satisfy all non-regression guards"
    top_acc=max(m[w]["oof"] for w in pool)
    near=[w for w in pool if m[w]["oof"]>=top_acc-.002-1e-12]
    # Differences <=0.2pp: prioritize most serious D->0 confusion, then other
    # three directions, seed robustness, OfficialVal, and smaller width.
    near.sort(key=lambda w:(
        m[w]["pairs"]["D->0"],
        sum(m[w]["pairs"][p] for p in ("B->8","8->B","0->D")),
        -m[w]["oof"],m[w]["seed_std"],-m[w]["val"],w))
    nextwidth=near[0]
    status=("TARGET_MET" if primary else
            ("FALLBACK_IMPROVED" if nextwidth!=100 else "BASELINE_RETAINED"))
    result={"status":status,"target_directed_error_rate_strictly_below":.05,
        "denominator_each_direction":480,"max_errors_per_direction":23,
        "priority_direction":"D->0","v2_protocol_updated_during_S1_before_result_comparison":True,
        "zero_error_required":False,
        "target_met_without_guards":target_meets,
        "target_met_and_safeguards_pass":primary,
        "fallback_safeguards_pass":fallback,
        "selected_width_for_S2":nextwidth,"selected_is_final_model":False,
        "selection_reason":(
           "All four error directions <5%, OOF nondecrease, robustness gates passed"
           if primary else "No acceptable model met all four <5% targets; fallback uses OOF accuracy, "
                         "non-increasing key confusion and class error rates plus robustness guards"),
        "original_test_used":False,"professor_test_used":False,
        "baseline_archived_models":12,"new_fits":36,"metrics":m,"gate":checks,
        "paired_vs_baseline":{w:paired(d[100],d[w]) for w in WIDTHS if w!=100}}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"S1_details.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    columns=["width","oof_accuracy","oof_error_rate","official_val_accuracy","oof_std",
             "D_to_0","D_to_0_rate","B_to_8","B_to_8_rate",
             "8_to_B","8_to_B_rate","0_to_D","0_to_D_rate",
             "class_0_error","class_8_error","class_B_error","class_D_error",
             "oof_worst_flip","val_worst_flip","mean_iter","fit_seconds","warnings",
             "target_all_4_under_5pct","passed_safeguards","failed_gates"]
    rows=[]
    for w in WIDTHS:
        x=m[w];g=checks[w]
        rows.append(dict(zip(columns,[w,x["oof"],1-x["oof"],x["val"],x["seed_std"],
          x["pairs"]["D->0"],x["pairs"]["D->0"]/480,
          x["pairs"]["B->8"],x["pairs"]["B->8"]/480,
          x["pairs"]["8->B"],x["pairs"]["8->B"]/480,
          x["pairs"]["0->D"],x["pairs"]["0->D"]/480,
          x["classerror"]["0"],x["classerror"]["8"],x["classerror"]["B"],x["classerror"]["D"],
          x["flip"]["OOF"],x["flip"]["OfficialVal"],x["mean_n_iter"],x["mean_fit_seconds"],
          x["warnings"],target_under_5percent(x),all(g.values()),
          ";".join(k for k,v in g.items() if not v)])))
    with (args.out/"S1_width_comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    md=["# EXP-CMB-008 · 02 S1 은닉층 너비 실측","",
        "> **판정:** "+status+" / 신규 학습 36개, 기준선 기존 12개 재사용.",
        "> **수정된 사용자 목표:** D→0, B→8, 8→B, 0→D 각각 실제 클래스 내 오분류율 **5% 미만**, 동시에 높은 OOF 정확도와 과적합 가드레일. 0건 강제 조건은 폐기.","",
        "## 가설과 실험 변화",
        "크기·획·CoM, Data SHA, Seed, writer split과 나머지 HP 고정. 표현 용량의 효과를 보기 위해 너비 64/128/256을 기준 100과 비교했다.",
        "각 오분류율의 분모는 실제 클래스당 480회(160장/Seed × 3Seed). <5% 기준은 23건 이하.","",
        "| Width | OOF Acc. | OOF 오분류율 | OfficialVal | D→0 건수(%) | B→8 건수(%) | 8→B 건수(%) | 0→D 건수(%) | <5% 목표 | 가드레일 위반 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        def directed(k):
            return str(row[k])+" ("+pct(row[k+"_rate"])+")"
        md.append("| "+str(row["width"])+" | "+pct(row["oof_accuracy"])+" | "+
             pct(row["oof_error_rate"])+" | "+pct(row["official_val_accuracy"])+" | "+
             directed("D_to_0")+" | "+directed("B_to_8")+" | "+directed("8_to_B")+" | "+
             directed("0_to_D")+" | "+("충족" if row["target_all_4_under_5pct"] else "미달")+
             " | "+(row["failed_gates"] or "없음")+" |")
    md += ["","## 선택 결과와 다음 단계",
        "- **목표 4방향 모두 <5% 도달 후보(가드레일 포함):** "+(str(primary) if primary else "없음"),
        "- **선정한 S2 비교 기준 너비:** "+str(nextwidth),
        "- **결정:** "+result["selection_reason"],
        "- **D→0 우선 관찰:** 기존 71건/480회(14.79%)이며 다른 방향 54건/43건/19건과 함께 악화 여부를 확인했다.",
        "- **과적합 감시:** Writer별 Accuracy, 3Seed 변동, OfficialVal 손실, OOF/Val 16클래스 Family Flip, 반복횟수, Loss와 경고를 개별 JSON/CSV로 보존한다.",
        "- **새 모델 최종 확정 아님:** 후속 S2~S6에서 순차적으로 평가한다. 독립 Test는 아직 사용하지 않는다.",
        "- **이력:** 초기 Zero-error 요구는 S1 학습 개시 후, 결과 비교 전 사용자가 수정했고 이 문서에는 변경된 <5% 기준을 적용했다.",""]
    # Report enough evidence to reconstruct the decision rather than only a winner.
    md+=["", "## 작성자·클래스별 정확도와 과적합 위험",
         "다음 표의 네 숫자는 각 실제 클래스의 **전체 오분류율**이며, 위 표의 특정 방향 오분류율과 구별한다.",
         "| Width | 실제 0 오류 | 실제 8 오류 | 실제 B 오류 | 실제 D 오류 | Writer 최저 OOF | Seed Std | OOF 최대 클래스 Flip | OfficialVal 최대 Flip | 수렴 경고 Fold |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for w in WIDTHS:
        x=m[w]
        md.append("| "+str(w)+" | "+" | ".join(pct(x["classerror"][cl]) for cl in ("0","8","B","D"))+
          " | "+pct(min(x["writers"].values()))+" | "+pct(x["seed_std"])+
          " | "+pct(x["flip"]["OOF"])+" | "+pct(x["flip"]["OfficialVal"])+
          " | "+str(x["warnings"])+" |")
    md+=["", "### Writer별 OOF Accuracy (동일 보류 Writer·Seed 집계)",
         "| Width | Writer 41 | Writer 42 | Writer 43 | Writer 44 | 평균 학습 반복 횟수 | 평균 Fold 학습 시간(초) |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
    for w in WIDTHS:
        x=m[w]
        md.append("| "+str(w)+" | "+" | ".join(pct(x["writers"][str(k)]) for k in WRITERS)+
          " | "+f'{x["mean_n_iter"]:.2f}'+" | "+f'{x["mean_fit_seconds"]:.2f}'+" |")
    md+=["", "## 동일 이미지 교정과 신규 오답 (100뉴런 모델 대비)",
         "| 후보 Width | 기존 오답을 정답으로 교정 | 기존 정답을 새로 오분류 | 순개선 |",
         "|---:|---:|---:|---:|"]
    for w in (64,128,256):
        fixed,created=result["paired_vs_baseline"][w]
        md.append("| "+str(w)+" | "+str(fixed)+" | "+str(created)+" | "+
                  ("+" if fixed-created>=0 else "")+str(fixed-created)+" |")
    md+=["", "## 후보 탈락·선정 근거 및 다음 실험에서 검증할 가설",
         "- **64뉴런:** 기준선 대비 OOF 정확도와 OfficialVal이 낮아지고 D→0/B→8 등 오분류가 늘면 탈락한다.",
         "- **128뉴런:** D→0과 8→B가 줄어도 B→8 및 0→D 오류가 늘었다면 특정 클래스 개선을 전체 개선으로 오해하지 않는다.",
         "- **256뉴런:** 정확도·4방향 오분류·Writer별 일반화·0/8/B/D 클래스 오류 가드레일을 통과하면 다음 깊이 실험의 단층 기준으로 삼는다.",
         "- **S2 실험:** 선택한 첫 층 너비 w를 고정하고 (w,), (w, w/2), (w, w) 깊이 후보를 비교한다. 첫 층 동일 조건에서 추가 은닉층 효과를 검증하는 순서다.",
         "- **과적합 주의:** 학습 반복 횟수가 적거나 수렴 경고가 없다는 이유만으로 과적합이 없다고 판단하지 않는다. 반복 사용한 4 Writer의 OOF에서도 탐색 과적합이 가능하며, 별도 보호한 최종 Test 전까지 일반화 성능을 확정하지 않는다.",
         "- **프로토콜 변경 이력:** S1 실행 중 사용자가 기존 0건 규칙을 네 방향 각각 5% 미만 목표로 수정했다. 실측 결과가 확정되기 전 변경한 규칙을 우선 적용하고 과거 규칙도 사전등록 문서에서 확인 가능하다.",
         ""]

    (args.out/"S1_Stage_Results.md").write_text("\n".join(md),encoding="utf-8")
    print("S1_DECISION",json.dumps({"status":result["status"],"provisional":nextwidth,
        "target_pass":primary,"rows":rows},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
