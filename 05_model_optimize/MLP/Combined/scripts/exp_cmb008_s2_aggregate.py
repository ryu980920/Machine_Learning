#!/usr/bin/env python3
"""EXP-CMB-008 S2: second-layer depth comparison, no model refit or Test access.

Frozen first hidden width=256 from S1; evaluate (256,), (256,128), (256,256).
Same 30% Size / 28% Stroke + CoM, seeds, writer folds and default other HP.
Prioritized gate: each actual-class directional D->0/B->8/8->B/0->D rate <5%,
otherwise best OOF accuracy without regression in those errors and guardrails.
"""
import argparse,csv,json,sys
from pathlib import Path
from exp_cmb008_s1_aggregate import SEEDS,WRITERS,readcsv,summarize,check,target_under_5percent,paired,pct

ARCHS=("256","256_128","256_256")
WATCH=("D->0","B->8","8->B","0->D")

def artifact(incoming,ref,arch,seed):
    return ((ref/f"cmb008-w256-s{seed}") if arch=="256" else
            (incoming/f"cmb008-s2-d{arch.split('_')[1]}-s{seed}"))

def read_arch(incoming,ref,arch):
    result={}
    for seed in SEEDS:
        p=artifact(incoming,ref,arch,seed)
        assert p.is_dir(),p
        m=json.loads((p/"manifest.json").read_text(encoding="utf-8"))
        assert m["experiment"]=="EXP-CMB-008" and m["arm"]=="scale30_stroke28"
        assert int(m["seed"])==seed and m["augmentation_seed"]==20261116
        assert m["handmade_train_preprocess"]=="CoM-only"
        assert m["heldout_and_official_val_preprocess"]=="CoM-only"
        assert m["official_train_only_augmentation"]
        assert not any(m[k] for k in ("original_test_used","professor_test_used","new_handmade_480_used"))
        if arch=="256":
            assert m["stage"]=="S1_width" and m["width"]==256
            assert m["only_changed_parameter"]=="hidden_layer_sizes"
        else:
            assert m["stage"]=="S2_depth"
            assert m["hidden_layer_sizes"]==[256,int(arch.split("_")[1])]
            assert m["only_changed_parameter"]=="add_second_hidden_layer"
        assert m["fixed_params"]=={
            "activation":"relu","solver":"adam","alpha":0.0001,
            "learning_rate_init":0.001,"max_iter":200,"early_stopping":False}
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
    data={arch:read_arch(a.new,a.s1_reference,arch) for arch in ARCHS}
    for arch in ARCHS:
        for seed in SEEDS:
            assert data[arch][seed]["m"]["official_sha256"]==data["256"][seed]["m"]["official_sha256"]
            assert data[arch][seed]["m"]["handmade_sha256"]==data["256"][seed]["m"]["handmade_sha256"]
    measured={arch:summarize(arch,data[arch]) for arch in ARCHS}
    base=measured["256"]
    assert abs(base["oof"]-.9221354166666667)<1e-9,base["oof"]
    assert abs(base["val"]-.9704427083333333)<1e-9,base["val"]
    checks={arch:check(base,measured[arch]) for arch in ARCHS}
    all_under=[arch for arch in ARCHS if target_under_5percent(measured[arch])]
    primary=[arch for arch in all_under if all(checks[arch].values())]
    fallback=[arch for arch in ARCHS if all(checks[arch].values())]
    assert "256" in fallback
    pool=primary or fallback
    bestacc=max(measured[k]["oof"] for k in pool)
    near=[k for k in pool if measured[k]["oof"]>=bestacc-.002-1e-12]
    near.sort(key=lambda k:(measured[k]["pairs"]["D->0"],
            sum(measured[k]["pairs"][q] for q in WATCH if q!="D->0"),
            -measured[k]["oof"],measured[k]["seed_std"],
            -measured[k]["val"],0 if k=="256" else int(k.split("_")[1])))
    winner=near[0]
    status="TARGET_MET" if primary else ("FALLBACK_IMPROVED" if winner!="256" else "BASELINE_RETAINED")
    paired_metrics={arch:paired(data["256"],data[arch]) for arch in ARCHS if arch!="256"}
    report={"experiment":"EXP-CMB-008","stage":"S2_depth","status":status,
        "target_rate_each_direction":"strictly < 5%","denominator_per_actual_class":480,
        "strict_target_max_count":23,"priority_confusion":"D->0",
        "selected_arch_for_S3":winner,"selected_is_final_hp":False,
        "primary_pass":primary,"safe_fallback_candidates":fallback,
        "data_sha_verified_against_s1":True,
        "new_fits":24,"s1_models_reused":12,
        "metrics":measured,"checks":checks,"paired":paired_metrics,
        "original_test_used":False,"professor_test_used":False,
        "selection_reason":"OOF accuracy first under <5% target; if not achievable, no-increase in four directions and protected classes plus Writer/Val/Flip safeguards"}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"S2_details.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    cols=["arch","oof_accuracy","oof_error_rate","official_val_accuracy","seed_std",
          "D_to_0","D_to_0_rate","B_to_8","B_to_8_rate",
          "8_to_B","8_to_B_rate","0_to_D","0_to_D_rate",
          "class_0_error","class_8_error","class_B_error","class_D_error",
          "oof_worst_flip","official_val_worst_flip","mean_n_iter","mean_fit_seconds",
          "warning_folds","target_4_under_5pct","safeguards_pass","failed_gates"]
    rows=[]
    for arch in ARCHS:
        x=measured[arch]
        c=checks[arch]
        vals=[arch,x["oof"],1-x["oof"],x["val"],x["seed_std"]]
        for pair in WATCH:
            vals.extend([x["pairs"][pair],x["pairs"][pair]/480])
        vals.extend(x["classerror"][q] for q in ("0","8","B","D"))
        vals.extend([x["flip"]["OOF"],x["flip"]["OfficialVal"],
                     x["mean_n_iter"],x["mean_fit_seconds"],x["warnings"],
                     target_under_5percent(x),all(c.values()),
                     ";".join(k for k,v in c.items() if not v)])
        assert len(vals)==len(cols),(len(vals),len(cols))
        rows.append(dict(zip(cols,vals)))
    with (a.out/"S2_depth_comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    md=["# EXP-CMB-008 · 03 S2 두 번째 은닉층 검증","",
        "> **상태:** "+status+" / (256,) 기준 모델 12개 재사용, (256,128)과 (256,256) 총 신규 24개 학습.",
        "> 동일 Size30/Stroke28/CoM/alpha/LR/Seed/Writer 분할, 변경 변수는 두 번째 은닉층 추가이다.","",
        "## S1 → S2의 인과 흐름",
        "S1은 100→256 첫 층 너비 확장으로 OOF가 90.716%→92.214% 개선되었지만, D→0은 13.96%로 목표 5%에 미달하였다.",
        "따라서 **첫 은닉층 256은 동결**하고, 추가 층이 더 복잡한 형태 특징을 학습하여 D/0 등 오류를 줄이는지 확인한다. 두 번째 층의 128/256 비교는 층 수 및 전체 파라미터 수 증가를 함께 포함하므로 층 수만의 순수 효과라고 해석하지 않는다.","",
        "## 전체 정확도와 주요 혼동쌍 (각 실제 클래스 480회 예측)",
        "| 구조 | OOF Accuracy | OOF 오분류율 | OfficialVal | D→0 | B→8 | 8→B | 0→D | 4개 <5% | 가드레일 위반 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        def err(name):return str(row[name])+" ("+pct(row[name+"_rate"])+")"
        md.append("| ("+row["arch"].replace("_",",")+",) | "+
                  pct(row["oof_accuracy"])+" | "+pct(row["oof_error_rate"])+" | "+
                  pct(row["official_val_accuracy"])+" | "+
                  " | ".join(err(t.replace("->","_to_")) for t in WATCH)+" | "+
                  ("충족" if row["target_4_under_5pct"] else "미달")+
                  " | "+(row["failed_gates"] or "없음")+" |")
    md+=["","## 클래스별 오분류율·작성자별 정확도·과적합 징후",
        "| 구조 | 실제 0 오류 | 8 오류 | B 오류 | D 오류 | Writer 최저 | Seed std | OOF 최대 Flip | OfficialVal 최대 Flip | 경고 Fold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ARCHS:
        x=measured[k]
        md.append("| ("+k.replace("_",",")+",) | "+
             " | ".join(pct(x["classerror"][q]) for q in ("0","8","B","D"))+
             " | "+pct(min(x["writers"].values()))+" | "+pct(x["seed_std"])+
             " | "+pct(x["flip"]["OOF"])+" | "+pct(x["flip"]["OfficialVal"])+
             " | "+str(x["warnings"])+" |")
    md+=["","### 동일 이미지의 교정과 신규 오류 (256 단층 대조)",
        "| 구조 | 기존 오답 교정 | 신규 오답 유발 | 순개선 |",
        "|---|---:|---:|---:|"]
    for k in ("256_128","256_256"):
        fixed,created=paired_metrics[k]
        md.append("| "+k+" | "+str(fixed)+" | "+str(created)+" | "+str(fixed-created)+" |")
    md+=["","## 선택 및 다음 단계",
        "- **4방향 모두 5% 미만 & 안전 Gate 통과:** "+(str(primary) if primary else "없음"),
        "- **S3 L2 규제 검증의 고정 구조:** "+winner,
        "- **결정 원칙:** 1차 5% 목표를 확인하고, 미달이면 기존 256 단층 대비 OOF 정확도·네 방향·4개 취약 클래스·Writer/OfficialVal/Flip이 악화되지 않는 후보만 채택한다.",
        "- **과적합 방지:** 학습 Loss, n_iter, 수렴 경고 및 Writer/Seed 분산을 함께 보되 과적합이 완전히 배제됐다고 단정하지 않는다. OOF 작성자는 탐색 중 반복 사용한 개발 데이터다.",
        "- **후속 S3:** 위 선택 구조를 고정하고 alpha=1e-5,1e-4,1e-3,1e-2의 규제 효과를 비교할 것. 이 단계에서는 아직 alpha를 선택하지 않는다.",
        "- Original Test 및 교수님 Test는 HP 선정에 사용하지 않았다.",""]
    (a.out/"S2_Stage_Results.md").write_text("\n".join(md),encoding="utf-8")
    print("S2_DECISION",json.dumps({"status":status,"selected":winner,"primary":primary,
         "fallback":fallback,"rows":rows},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
