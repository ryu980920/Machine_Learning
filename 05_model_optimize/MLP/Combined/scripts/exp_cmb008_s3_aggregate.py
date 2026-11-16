#!/usr/bin/env python3
"""EXP-CMB-008 S3: L2 alpha comparison, frozen (256,) architecture, no model refit or Test access.

Freeze (256,) after S2; alpha 1e-5, 1e-4, 1e-3, 1e-2.
Same 30% Size / 28% Stroke + CoM, seeds, writer folds and default other HP.
Prioritized gate: each actual-class directional D->0/B->8/8->B/0->D rate <5%,
otherwise best OOF accuracy without regression in those errors and guardrails.
"""
import argparse,csv,json,sys
from pathlib import Path
from exp_cmb008_s1_aggregate import SEEDS,WRITERS,readcsv,summarize,check,target_under_5percent,paired,pct

ALPHAS=("1e-5","1e-4","1e-3","1e-2")
WATCH=("D->0","B->8","8->B","0->D")

def artifact(incoming,ref,alpha_key,seed):
    return ((ref/f"cmb008-w256-s{seed}") if alpha_key=="1e-4" else
            (incoming/f"cmb008-s3-a{alpha_key}-s{seed}"))

def read_alpha(incoming,ref,alpha_key):
    result={}
    for seed in SEEDS:
        p=artifact(incoming,ref,alpha_key,seed)
        assert p.is_dir(),p
        m=json.loads((p/"manifest.json").read_text(encoding="utf-8"))
        assert m["experiment"]=="EXP-CMB-008" and m["arm"]=="scale30_stroke28"
        assert int(m["seed"])==seed and m["augmentation_seed"]==20261116
        assert m["handmade_train_preprocess"]=="CoM-only"
        assert m["heldout_and_official_val_preprocess"]=="CoM-only"
        assert m["official_train_only_augmentation"]
        assert not any(m[k] for k in ("original_test_used","professor_test_used","new_handmade_480_used"))
        expected={"activation":"relu","solver":"adam","alpha":float(alpha_key),
                  "learning_rate_init":0.001,"max_iter":200,"early_stopping":False}
        if alpha_key=="1e-4":
            assert m["stage"]=="S1_width" and m["width"]==256
            assert m["only_changed_parameter"]=="hidden_layer_sizes"
        else:
            assert m["stage"]=="S3_alpha"
            assert m["hidden_layer_sizes"]==[256] and m["alpha"]==float(alpha_key)
            assert m["only_changed_parameter"]=="alpha"
            expected["hidden_layer_sizes"]=[256]
        assert m["fixed_params"]==expected
        oof=readcsv(p/"oof_predictions.csv")
        assert len(oof)==2560 and len({(int(x["writer"]),int(x["index"])) for x in oof})==2560
        vals=[x for x in readcsv(p/"fold_metrics.csv") if x["dataset"]=="OfficialVal" and x["group"]=="all"]
        assert len(vals)==4 and all(int(x["n"])==19200 for x in vals)
        scales=[x for x in readcsv(p/"per_class_and_scale.csv") if
                x["dataset"] in ("OOF","OfficialVal") and float(x["scale"]) in (.8,1.2)]
        assert scales
        times=readcsv(p/"training_time.csv")
        assert len(times)==4
        result[seed]={"m":m,"oof":oof,"val":vals,"scales":scales,"times":times}
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--new",type=Path,required=True)
    ap.add_argument("--s1-reference",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    data={alpha_key:read_alpha(a.new,a.s1_reference,alpha_key) for alpha_key in ALPHAS}
    for alpha_key in ALPHAS:
        for seed in SEEDS:
            assert data[alpha_key][seed]["m"]["official_sha256"]==data["1e-4"][seed]["m"]["official_sha256"]
            assert data[alpha_key][seed]["m"]["handmade_sha256"]==data["1e-4"][seed]["m"]["handmade_sha256"]
    measured={alpha_key:summarize(alpha_key,data[alpha_key]) for alpha_key in ALPHAS}
    base=measured["1e-4"]
    assert abs(base["oof"]-.9221354166666667)<1e-9,base["oof"]
    assert abs(base["val"]-.9704427083333333)<1e-9,base["val"]
    checks={alpha_key:check(base,measured[alpha_key]) for alpha_key in ALPHAS}
    all_under=[alpha_key for alpha_key in ALPHAS if target_under_5percent(measured[alpha_key])]
    primary=[alpha_key for alpha_key in all_under if all(checks[alpha_key].values())]
    fallback=[alpha_key for alpha_key in ALPHAS if all(checks[alpha_key].values())]
    assert "1e-4" in fallback
    pool=primary or fallback
    bestacc=max(measured[k]["oof"] for k in pool)
    near=[k for k in pool if measured[k]["oof"]>=bestacc-.002-1e-12]
    near.sort(key=lambda k:(measured[k]["pairs"]["D->0"],
            sum(measured[k]["pairs"][q] for q in WATCH if q!="D->0"),
            -measured[k]["oof"],measured[k]["seed_std"],
            -measured[k]["val"],0 if k=="1e-4" else float(k)))
    winner=near[0]
    status="TARGET_MET" if primary else ("FALLBACK_IMPROVED" if winner!="1e-4" else "BASELINE_RETAINED")
    paired_metrics={alpha_key:paired(data["1e-4"],data[alpha_key]) for alpha_key in ALPHAS if alpha_key!="1e-4"}
    report={"experiment":"EXP-CMB-008","stage":"S3_alpha","status":status,
        "target_rate_each_direction":"strictly < 5%","denominator_per_actual_class":480,
        "strict_target_max_count":23,"priority_confusion":"D->0",
        "selected_alpha_for_S4":winner,"selected_is_final_hp":False,
        "primary_pass":primary,"safe_fallback_candidates":fallback,
        "data_sha_verified_against_s1":True,
        "new_fits":36,"s1_models_reused":12,
        "metrics":measured,"checks":checks,"paired":paired_metrics,
        "original_test_used":False,"professor_test_used":False,
        "selection_reason":"OOF accuracy first under <5% target; if not achievable, no-increase in four directions and protected classes plus Writer/Val/Flip safeguards"}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"S3_details.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    cols=["alpha","oof_accuracy","oof_error_rate","official_val_accuracy","seed_std",
          "D_to_0","D_to_0_rate","B_to_8","B_to_8_rate",
          "8_to_B","8_to_B_rate","0_to_D","0_to_D_rate",
          "class_0_error","class_8_error","class_B_error","class_D_error",
          "oof_worst_flip","official_val_worst_flip","mean_n_iter","mean_fit_seconds",
          "warning_folds","target_4_under_5pct","safeguards_pass","failed_gates"]
    rows=[]
    for alpha_key in ALPHAS:
        x=measured[alpha_key]
        c=checks[alpha_key]
        vals=[alpha_key,x["oof"],1-x["oof"],x["val"],x["seed_std"]]
        for pair in WATCH:
            vals.extend([x["pairs"][pair],x["pairs"][pair]/480])
        vals.extend(x["classerror"][q] for q in ("0","8","B","D"))
        vals.extend([x["flip"]["OOF"],x["flip"]["OfficialVal"],
                     x["mean_n_iter"],x["mean_fit_seconds"],x["warnings"],
                     target_under_5percent(x),all(c.values()),
                     ";".join(k for k,v in c.items() if not v)])
        assert len(vals)==len(cols),(len(vals),len(cols))
        rows.append(dict(zip(cols,vals)))
    with (a.out/"S3_alpha_comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    md=["# EXP-CMB-008 · 06 S3 L2 alpha 최적화 실측","",
        "> 판정: "+status+" / 신규 alpha 3개 × 3Seed × 4Fold=36개, 기준선 alpha=1e-4 기존 12개 재사용.","",
        "## 실험의 인과적 흐름",
        "S2의 (256,128)/(256,256) 두 층 모델은 전체 OOF를 소폭 개선하거나 감소시켰지만, 8→B 등 핵심 혼동을 악화시켰다. 따라서 (256,) 단층을 유지하고 모델 용량을 바꾸지 않은 상태에서 L2 규제 alpha만 비교한다.",
        "1e-5는 기본 1e-4보다 약한 규제, 1e-3과 1e-2는 더 강한 규제다. 다른 HP와 데이터는 고정했다.","",
        "## 정확도·4방향 실제 오분류율 (각 클래스 480회)",
        "| alpha | Hand-made OOF | OOF 오분류율 | OfficialVal | D→0 | B→8 | 8→B | 0→D | 모두 <5% | 탈락 사유 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        def err(n):return str(row[n])+" ("+pct(row[n+"_rate"])+")"
        md.append("| "+str(row["alpha"])+" | "+pct(row["oof_accuracy"])+" | "+pct(row["oof_error_rate"])+
          " | "+pct(row["official_val_accuracy"])+" | "+
          " | ".join(err(k.replace("->","_to_")) for k in WATCH)+" | "+
          ("충족" if row["target_4_under_5pct"] else "미달")+" | "+(row["failed_gates"] or "없음")+" |")
    md+=["","## 과적합·특정 클래스 붕괴 확인",
        "| alpha | 전체 0오류 | 8오류 | B오류 | D오류 | 최저 Writer 정확도 | Seed 표준편차 | OOF 최대 클래스 Flip | Val 최대 Flip | 수렴경고 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ALPHAS:
        z=measured[k]
        md.append("| "+k+" | "+" | ".join(pct(z["classerror"][n]) for n in ("0","8","B","D"))+
          " | "+pct(min(z["writers"].values()))+" | "+pct(z["seed_std"])+
          " | "+pct(z["flip"]["OOF"])+" | "+pct(z["flip"]["OfficialVal"])+
          " | "+str(z["warnings"])+" |")
    md+=["","### 보류 Writer별 OOF 정확도 / 수렴",
        "| alpha | W41 | W42 | W43 | W44 | 평균 학습 반복 | 평균 시간(초) |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ALPHAS:
        z=measured[k]
        md.append("| "+k+" | "+" | ".join(pct(z["writers"][str(w)]) for w in WRITERS)+
          " | "+f'{z["mean_n_iter"]:.2f}'+" | "+f'{z["mean_fit_seconds"]:.2f}'+" |")
    md+=["","### 같은 이미지 교정/새 오답 수 (기준 alpha=1e-4 대비)",
        "| alpha | 기존 오류 교정 | 기존 정답 신규 오분류 | 순개선 |",
        "|---:|---:|---:|---:|"]
    for k in ALPHAS:
        if k=="1e-4":continue
        corr,ind=paired_metrics[k]
        md.append("| "+k+" | "+str(corr)+" | "+str(ind)+" | "+str(corr-ind)+" |")
    md+=["","## 확정·보류와 다음 단계",
        "- **4방향 각 5% 미만 및 가드레일 통과:** "+(str(primary) if primary else "없음"),
        "- **S4 학습률 검증의 기준 alpha:** "+winner,
        "- **선정 이유:** "+report["selection_reason"],
        "- **선정 규칙:** 목표 네 방향 모두 <5%를 우선 확인하고, 미달이면 D→0을 포함한 네 방향과 클래스 0/8/B/D의 오분류 비증가, Writer/OfficialVal/Flip 가드레일, 전체 OOF 최고 정확도를 적용한다. 0.20%p 이내는 D→0을 우선한다.",
        "- **과적합 주의:** Loss curve/n_iter/경고를 남겨도 개발용 OOF 반복 탐색으로 인한 과적합까지 배제할 수는 없다. 독립 Test는 사용하지 않는다.",
        "- **S4 다음 질문:** alpha와 단층 (256,)를 고정한 뒤 Adam learning_rate_init 0.0005,0.001,0.002의 수렴·오류 재배분 효과를 비교한다.",
        "- Test·교수님 Test·신규 480장 미사용.",""]
    (a.out/"S3_Stage_Results.md").write_text("\n".join(md),encoding="utf-8")
    print("S3_DECISION",json.dumps({"status":status,"selected_alpha":winner,"primary":primary,
         "fallback":fallback,"rows":rows},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
