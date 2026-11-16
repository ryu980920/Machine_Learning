#!/usr/bin/env python3
"""EXP-CMB-008 S1: compare archived width100 vs new widths on identical writer OOF.
No fit or access to Original Test; zero 0->D and 8->B is a strict final gate.
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
def check(b,n):
    return {
      "official_validation_no_drop_over_0p5pp":n["val"]>=b["val"]-.005-1e-12,
      "writer_no_drop_over_1pp":all(n["writers"][k]>=b["writers"][k]-.01-1e-12 for k in b["writers"]),
      "B8D_error_no_increase_over_3pp":all(n["classerror"][k]<=b["classerror"][k]+.03+1e-12 for k in ("8","B","D")),
      "max_class_family_flip_no_increase_over_5pp":all(n["flip"][k]<=b["flip"][k]+.05+1e-12 for k in ("OOF","OfficialVal")),
      "zero_0_to_D":n["pairs"]["0->D"]==0,
      "zero_8_to_B":n["pairs"]["8->B"]==0,
      "reverse_D_to_0_no_increase":n["pairs"]["D->0"]<=b["pairs"]["D->0"],
      "reverse_B_to_8_no_increase":n["pairs"]["B->8"]<=b["pairs"]["B->8"]}
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
    full=[w for w in (64,128,256) if all(checks[w].values())]
    # S1 is a provisional choice if neither critical pair reaches zero. A final pass
    # must satisfy zero errors in BOTH directions named by the user.
    safe=[]
    for w in WIDTHS:
        core=all(v for k,v in checks[w].items() if k not in ("zero_0_to_D","zero_8_to_B"))
        nonincrease=all(m[w]["pairs"][k]<=m[100]["pairs"][k] for k in ("0->D","8->B"))
        if w==100 or (core and nonincrease and m[w]["oof"]>=m[100]["oof"]-.002):
            safe.append(w)
    safe.sort(key=lambda w:(-m[w]["oof"],m[w]["pairs"]["0->D"]+m[w]["pairs"]["8->B"],w))
    full.sort(key=lambda w:-m[w]["oof"])
    nextwidth=full[0] if full else safe[0]
    result={"status":"PASSED" if full else "NOT_PASSED","final_gate_passed_widths":full,
        "provisional_next_width":nextwidth,"provisional_is_not_final":not bool(full),
        "baseline_archived_models":12,"new_fits":36,"metrics":m,"gate":checks,
        "paired_vs_baseline":{w:paired(d[100],d[w]) for w in WIDTHS if w!=100}}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"S1_details.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    columns=["width","oof_accuracy","official_val_accuracy","oof_std","0_to_D","8_to_B","D_to_0","B_to_8",
             "class_8_error","class_B_error","class_D_error","oof_worst_flip","val_worst_flip",
             "mean_iter","fit_seconds","warnings","passed_all","failed_gates"]
    rows=[]
    for w in WIDTHS:
        x=m[w];g=checks[w]
        rows.append(dict(zip(columns,[w,x["oof"],x["val"],x["seed_std"],
          x["pairs"]["0->D"],x["pairs"]["8->B"],x["pairs"]["D->0"],x["pairs"]["B->8"],
          x["classerror"]["8"],x["classerror"]["B"],x["classerror"]["D"],
          x["flip"]["OOF"],x["flip"]["OfficialVal"],x["mean_n_iter"],x["mean_fit_seconds"],
          x["warnings"],all(g.values()),";".join(k for k,v in g.items() if not v)])))
    with (args.out/"S1_width_comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    md=["# EXP-CMB-008 · 02 S1 은닉층 너비 실측","",
        "> **엄격 통과 여부:** "+result["status"]+" / 완료된 신규 모델 36개, 과거 기준선 12개 재사용.","",
        "## 변경 근거","기본 은닉층 100의 표현 용량이 부족한지 검증하기 위해 64/128/256을 시험했다. 다른 하이퍼파라미터와 Size30/Stroke28/CoM, Seeds 및 Writer 분할은 동일하다.","",
        "| Width | Hand-made OOF | OfficialVal | 0→D 건수 | 8→B 건수 | D→0 | B→8 | Gate 위반 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        md.append("| "+str(row["width"])+" | "+pct(row["oof_accuracy"])+" | "+pct(row["official_val_accuracy"])+" | "+
             str(row["0_to_D"])+" | "+str(row["8_to_B"])+" | "+str(row["D_to_0"])+" | "+str(row["B_to_8"])+" | "+
             (row["failed_gates"] or "없음")+" |")
    md += ["","## 결론 및 다음 실험",
       "0→D와 8→B 두 방향을 세 Seed 전체에서 모두 **0건**으로 만들고 다른 혼동의 악화를 방지하는 것이 최종 통과 조건이다.",
       "**S1 최종 통과 Width:** "+(str(full) if full else "없음 (최종 승인 불가)"),
       "**S2를 위한 잠정 Width:** "+str(nextwidth),
       "잠정 선택은 최종 통과가 아니며, S2에서도 동일한 Zero-error Gate와 각 작성자/클래스 악화 기준을 다시 적용한다.",
       "모든 Seed/Writer/16클래스 Family Flip과 실패 근거는 S1_details.json 및 S1_width_comparison.csv에 기록했다.",
       "Original Test·교수님 Test·신규 손글씨 480장 미사용.",""]
    (args.out/"S1_Stage_Results.md").write_text("\n".join(md),encoding="utf-8")
    print("S1_DECISION",json.dumps({"status":result["status"],"provisional":nextwidth,
        "pass":full,"rows":rows},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
