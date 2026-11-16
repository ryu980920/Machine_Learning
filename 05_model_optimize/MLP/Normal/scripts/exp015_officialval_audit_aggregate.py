#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate fixed EXP015 models under Official Validation scale interventions."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
def read(p):
    with Path(p).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def save(p,rows):
    assert rows
    with Path(p).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(x):return sum(x)/len(x)
def run(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    folders=[root/f"exp015-audit-s{s}" for s in (42,43,44)]
    for p in folders:
        meta=json.loads((p/"manifest.json").read_text(encoding="utf-8"))
        assert meta["original_exp006_com_predictions_verified"]
        assert meta["fine_tuning_or_training_performed"] is False
        assert not meta["official_test_used"] and not meta["professor_test_used"]
    rows=[r for p in folders for r in read(p/"officialval_scale_sensitivity.csv")]
    pairs=[r for p in folders for r in read(p/"officialval_pair_errors.csv")]
    save(out/"officialval_scale_all_seed.csv",rows)
    save(out/"officialval_pair_all_seed.csv",pairs)
    grouped=defaultdict(list)
    for r in rows:
        key=(r["model"],r["scale"],r["group"])
        grouped[key].append(r)
    summary=[]
    for (model,scale,group),v in sorted(grouped.items()):
        assert len(v)==3
        s={"model":model,"scale":scale,"group":group,"n_per_seed":int(v[0]["n"]),"seeds":3}
        for col in ("family_flip_rate","mean_abs_delta_p_letter","mean_signed_delta_p_letter","accuracy"):
            s["mean_"+col]=mean([float(r[col]) for r in v])
            s["max_seed_"+col]=max(float(r[col]) for r in v)
        summary.append(s)
    save(out/"officialval_scale_3seed_summary.csv",summary)
    md=["# EXP015 — OfficialVal 크기 개입 교차 진단",
        "",
        "78개 학습 후 기존 사전 Gate를 통과한 9개 후보와 동결된 CoM 원본을 **고정 가중치**로 재평가하였다.",
        "기존 OfficialVal 19,200장의 원본 이미지를 0.8·0.9·1.0·1.1·1.2배 변형한 후 CoM 정렬.",
        "**Hand-made에 대해 사전 등록한 Gate/순위는 변경하지 않는다.**",
        "OfficialVal은 이미 반복 사용된 개발 Validation이지 새로운 독립 Test가 아니다.",
        "",
        "## 숫자 ×1.2 및 문자 ×0.8의 Family 판단 전환율",
        "",
        "| 모델 | OfficialVal 숫자 1.2x flip | OfficialVal 문자 0.8x flip | 원본 OfficialVal Accuracy |",
        "|---|---:|---:|---:|"]
    lookup={(r["model"],r["group"],float(r["scale"])):r for r in summary}
    names=sorted({r["model"] for r in summary})
    for name in names:
        digits=lookup[name,"digits",1.2]
        letters=lookup[name,"letters",.8]
        original=lookup[name,"all",1.]
        md.append(f"| {name} | {100*digits['mean_family_flip_rate']:.2f}% | "
                  f"{100*letters['mean_family_flip_rate']:.2f}% | "
                  f"{100*original['mean_accuracy']:.3f}% |")
    md+=["","## 집중 혼동쌍 B·8·D·0·F·5의 Family Flip","",
        "| 모델 | 문자/숫자 | 0.8x | 1.2x |",
        "|---|---|---:|---:|"]
    for name in names:
        for label in ("8","B","0","D","5","F"):
            c8=lookup[name,"class_"+label,.8]["mean_family_flip_rate"]
            c12=lookup[name,"class_"+label,1.2]["mean_family_flip_rate"]
            md.append(f"| {name} | {label} | {100*c8:.2f}% | {100*c12:.2f}% |")
    md+=["","### 한계 및 결론",
      "- 이미지 배율 조절은 크기뿐 아니라 획 질량, 두께, 보간·topology도 함께 바꾼다.",
      "- Hand-made에서 Size Shortcut이 완화된 후보라도 OfficialVal에서 동일한 강건성을 보이지 않을 수 있다.",
      "- B/8/D/0/F/5 등 특정 문자 쌍에서 높은 Flip이 남는다면 Shortcut 완전 제거를 주장할 수 없다.",
      "- 어떤 모델도 Test를 평가하거나 final 모델에 교체하지 않았다.",
      ""]
    (out/"SUMMARY.md").write_text("\n".join(md),encoding="utf-8")
    (out/"manifest.json").write_text(json.dumps(
        {"experiment":"EXP015_OFFICIALVAL_AUDIT_AGGREGATE",
         "seeds":[42,43,44],"n_fixed_models":len(names),
         "model_fitting_performed":False,"test_used":False,
         "original_preregistered_criteria_modified":False},indent=2),encoding="utf-8")
    print("EXP015_OFFICIALVAL_AUDIT_AGGREGATE_SUCCESS",len(names),len(summary),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.inputs,a.out)
