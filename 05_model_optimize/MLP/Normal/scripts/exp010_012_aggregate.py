#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate genuine, checked EXP010/011/012 results without inferring numbers."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

SEEDS=(42,43,44)
METHODS=("com","size","size_com")
INPUTS=("raw","com","size","size_com")
FOCUS=("B","8","D","0","F","5")

def read_csv(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
def save_csv(path,rows):
    if not rows:raise RuntimeError("No results for "+str(path))
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
def avg(vals):
    if not vals:raise AssertionError("Missing grouped comparison")
    return float(np.mean(vals))
def pct(x):return f"{x*100:.4f}%"
def ffloat(row,key):return float(row[key])
def subgroup(rows,**conditions):
    return [r for r in rows if all(str(r[k])==str(v) for k,v in conditions.items())]

def run(input_root,out):
    input_root,out=Path(input_root),Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    geo=input_root/"exp010-geometry"
    if not geo.is_dir():raise FileNotFoundError(geo)
    geoman=json.loads((geo/"manifest.json").read_text(encoding="utf-8"))
    assert not geoman["official_original_test_used"]
    assert geoman["target_bbox_longside"]==20
    stat=read_csv(geo/"geometry_summary.csv")
    dists=read_csv(geo/"geometry_distance.csv")
    hgeo=read_csv(geo/"handmade_geometry_per_image.csv")
    cross=[];paired=[];errors=[];geometry=[];scale=[];sg=[];manifests={}
    for seed in SEEDS:
        d=input_root/f"exp011-012-seed{seed}"
        if not d.is_dir():raise FileNotFoundError(d)
        manifest=json.loads((d/"manifest.json").read_text(encoding="utf-8"))
        assert manifest["seed"]==seed
        assert manifest["exact_original_prediction_counts_verified"]
        assert not manifest["training_performed"]
        assert not manifest["official_original_test_used"]
        assert not manifest["professor_test_used"]
        assert not manifest["handmade_training_used"]
        assert manifest["train_source_sha256"]==geoman["official_sha256"]
        manifests[str(seed)]=manifest
        cross+=read_csv(d/"cross_metrics.csv")
        paired+=read_csv(d/"paired_effects.csv")
        errors+=read_csv(d/"pair_errors.csv")
        geometry+=read_csv(d/"size_com_geometry_and_outcomes.csv")
        scale+=read_csv(d/"scale_class_response.csv")
        sg+=read_csv(d/"scale_geometry.csv")
    save_csv(out/"all_cross_metrics.csv",cross)
    save_csv(out/"all_paired_effects.csv",paired)
    save_csv(out/"all_pair_errors.csv",errors)
    save_csv(out/"all_size_center_outcomes.csv",geometry)
    save_csv(out/"all_scale_response.csv",scale)
    save_csv(out/"all_scale_geometry.csv",sg)
    summary=[]
    for method in METHODS:
        for dataset in ("OfficialVal","Handmade"):
            for inp in INPUTS:
                for group in ("all","digits","letters",*[f"class_{c}" for c in "0123456789ABCDEF"]):
                    xs=subgroup(cross,trained=method,dataset=dataset,input=inp,group=group)
                    if len(xs)!=3:raise AssertionError((method,dataset,inp,group,len(xs)))
                    summary.append({"trained":method,"dataset":dataset,"input":inp,"group":group,
                        "n_per_seed":int(xs[0]["n"]),
                        "mean_accuracy":avg([ffloat(x,"accuracy") for x in xs]),
                        "seed_42_acc":float(xs[0]["accuracy"]),
                        "seed_43_acc":float(xs[1]["accuracy"]),
                        "seed_44_acc":float(xs[2]["accuracy"])})
    save_csv(out/"exp011_accuracy_matrix.csv",summary)
    comparisons=[]
    for contrast in sorted(set(r["contrast"] for r in paired)):
        for group in ("all","digits","letters",*[f"class_{c}" for c in "0123456789ABCDEF"]):
            xs=subgroup(paired,contrast=contrast,group=group)
            assert len(xs)==3,(contrast,group,len(xs))
            comparisons.append({"contrast":contrast,"group":group,
                "n":int(xs[0]["n"]),"mean_fixed":avg([ffloat(x,"fixed") for x in xs]),
                "mean_new_wrong":avg([ffloat(x,"new_wrong") for x in xs]),
                "mean_net_correct":avg([ffloat(x,"net_correct") for x in xs]),
                "mean_delta_accuracy_pp":avg([ffloat(x,"delta_accuracy_pp") for x in xs])})
    save_csv(out/"exp011_paired_summary.csv",comparisons)

    # Size-only diagonal -> Size+CoM diagonal, class/writer geometric shift
    diagnostic=[]
    for group,sub in [
        ("all",geometry),
        ("digits",[r for r in geometry if r["class"].isdigit()]),
        ("letters",[r for r in geometry if r["class"] in "ABCDEF"]),
        ("recovered",[r for r in geometry if r["size_model_size_correct"]=="0"
             and r["size_com_model_size_com_correct"]=="1"]),
        ("newly_wrong",[r for r in geometry if r["size_model_size_correct"]=="1"
             and r["size_com_model_size_com_correct"]=="0"])
    ]:
        for cl in ("all",*[f"class_{c}" for c in "0123456789ABCDEF"]):
            sample=sub if cl=="all" else [r for r in sub if r["class"]==cl[6:]]
            if not sample:continue
            diagnostic.append({"status":group,"class_or_all":cl,"n_across_seeds":len(sample),
                "mean_com_distance_before":avg([ffloat(x,"com_distance_before") for x in sample]),
                "mean_com_distance_after":avg([ffloat(x,"com_distance_after") for x in sample]),
                "mean_shift_distance":avg([ffloat(x,"size_com_shift_distance") for x in sample]),
                "mean_mass_ratio_after":avg([ffloat(x,"mass_after")/
                    max(ffloat(x,"mass_before"),1e-12) for x in sample]),
                "fraction_nonzero_shift":avg([ffloat(x,"size_com_shift_distance")>0 for x in sample])})
    save_csv(out/"exp011_center_geometry_by_outcome.csv",diagnostic)

    # EXP010 reference geometry and domain differences
    g010=[]
    for cls in ("all","digits","letters",*[f"class_{c}" for c in "0123456789ABCDEF"]):
        for method in ("raw","size","size_com"):
            for feature in ("width","height","long_side","com_offset","mass"):
                a=subgroup(stat,dataset="OfficialTrain",preprocessing=method,
                           group=cls,feature=feature)
                b=subgroup(stat,dataset="Handmade",preprocessing=method,
                           group=cls,feature=feature)
                dx=subgroup(dists,preprocessing=method,segment=cls,
                            feature=feature)
                if not a or not b:raise AssertionError((cls,method,feature))
                g010.append({"group":cls,"preprocessing":method,
                    "feature":feature,"train_median":float(a[0]["median"]),
                    "handmade_median":float(b[0]["median"]),
                    "train_mean":float(a[0]["mean"]),
                    "handmade_mean":float(b[0]["mean"]),
                    "train_handmade_w1":float(dx[0]["train_vs_handmade_w1"]) if dx else ""})
    save_csv(out/"exp010_geometry_core_summary.csv",g010)

    # EXP012 pair margins (pre-registered 6 focus directions), 3 seeds, 3 MLPs
    pairs=[]
    all_classes=[]
    for trained in METHODS:
        for cl in FOCUS:
            for factor in (.8,.9,1.0,1.1,1.2):
                xs=subgroup(scale,trained=trained,true=cl,scale=factor)
                assert len(xs)==480,(trained,cl,factor,len(xs)) # 160/class ×3 seeds
                pairs.append({"trained":trained,"class":cl,
                    "comparison":cl+" vs "+xs[0]["confuser"],
                    "scale":factor,"n_across_seeds":len(xs),
                    "true_recall":avg([int(x["correct"]) for x in xs]),
                    "mean_pair_margin":avg([float(x["true_vs_pair_margin"]) for x in xs]),
                    "mean_p_true":avg([float(x["p_true"]) for x in xs]),
                    "mean_p_confuser":avg([float(x["p_confuser"]) for x in xs]),
                    "n_pred_as_confuser":sum(int(x["pred"]==x["confuser"]) for x in xs)})
        for group in ("digits","letters"):
            for factor in (.8,.9,1.0,1.1,1.2):
                rows=[r for r in scale if r["trained"]==trained and
                    float(r["scale"])==factor and
                    (r["true"] in "0123456789" if group=="digits" else
                     r["true"] in "ABCDEF")]
                assert len(rows)==(1600 if group=="digits" else 960)*3
                all_classes.append({"trained":trained,"group":group,
                    "scale":factor,"n_across_seeds":len(rows),
                    "accuracy":avg([int(r["correct"]) for r in rows])})
    save_csv(out/"exp012_scale_pair_margins.csv",pairs)
    save_csv(out/"exp012_scale_group_accuracy.csv",all_classes)
    topo=[]
    for cl in FOCUS:
        for factor in (.8,.9,1.0,1.1,1.2):
            sample=subgroup(sg,true=cl,scale=factor)
            assert len(sample)==480,(cl,factor,len(sample))
            topo.append({"class":cl,"scale":factor,"n_across_seeds":len(sample),
                "fraction_topology_changed":avg([float(x["topology_changed_for_focus"]) for x in sample]),
                "median_bbox_longside":float(np.median([float(x["bbox_longside_after"]) for x in sample])),
                "mean_ink_mass_ratio":avg([float(x["mass_ratio_after_over_original"]) for x in sample]),
                "mean_com_offset":avg([float(x["com_offset_after"]) for x in sample])})
    save_csv(out/"exp012_geometry_confound_checks.csv",topo)

    # Precompute class mean accuracy at crossover, compare effect vs historic.
    lookup={(r["trained"],r["dataset"],r["input"],r["group"]):float(r["mean_accuracy"])
            for r in summary}
    def acc(method,inp,group,dataset="Handmade"):
        return lookup[method,dataset,inp,group]
    line=["# Normal EXP유010·011·012 — 실제 측정 종합 결과","",
          "**상태: 3가설 검증 완료. EXP유006 원본 9개 체크포인트 해시·예측 정답 수 재검증, 재학습 없음.**",
          "","- 공식 Train 76,800 · Validation 19,200 · Hand-made 2,560 (작성자 41-44).",
          "- EXP유006과 완전히 같은 Combined Size(threshold32, longside20, bilinear), CoM(threshold32, 정수 이동).",
          "- 공식 Original Test, 교수님 Test, 새로운 Hand-made 학습 없음.",
          "- 모든 수치는 동일한 3 Seed(42·43·44)의 성공한 기존 MLP를 사용한 실측이다.",
          "",
          "## EXP유011 원본 모델 vs 평가 입력 교차 비교",
          "",
          "| 학습 모델 | 평가 입력 | Hand-made 숫자 | Hand-made 문자 | 공식 Validation |",
          "|---|---|---:|---:|---:|"]
    for model in METHODS:
        for inp in ("com","size","size_com"):
            line.append(f"| {model} | {inp} | {pct(acc(model,inp,'digits'))} | "
                  f"{pct(acc(model,inp,'letters'))} | "
                  f"{pct(acc(model,inp,'all','OfficialVal'))} |")
    line+=["","## EXP유010 주요 크기 분포","",
           "| 문자/숫자 | 전처리 | 공식 Training BBox 긴 변 중앙값 | Hand-made BBox 긴 변 중앙값 | W1 거리 |",
           "|---|---|---:|---:|---:|"]
    for cl in ("B","8","D","0","F","5"):
        for method in ("raw","size","size_com"):
            r=[x for x in g010 if x["group"]=="class_"+cl and x["preprocessing"]==method
               and x["feature"]=="long_side"][0]
            w=r["train_handmade_w1"]
            line.append(f"| {cl} | {method} | {r['train_median']:.2f}px | "
                        f"{r['handmade_median']:.2f}px | {w:.3f} |")
    line+=["","## EXP유011 중요 결과","",
      "- Size-only vs Size→CoM 대각선 정확도 차이는 **학습도 입력도 동시에 변경**한 값이다.",
      "- 같은 학습 모델의 Size 입력과 Size→CoM 입력 차이가 '입력 위치 효과'를 직접 점검하는 비교다.",
      "- 반대로 동일한 입력에 Size-only 모델과 Size→CoM 모델을 적용한 차이는 모델 학습의 영향이다.",
      "- 서로 다른 train/validation 이미지 분포를 교차 적용하는 경우에는 OOD 효과가 섞일 수 있다.",
      "",
      "## EXP유012 크기 개입 — 중심 정렬 유지","",
      "| 모델 | 실제 클래스 | 0.8x pair margin | 1.0x pair margin | 1.2x pair margin |",
      "|---|---|---:|---:|---:|"]
    for model in ("com","size_com"):
        for cl in FOCUS:
            rr=[r for r in pairs if r["trained"]==model and r["class"]==cl]
            mar={r["scale"]:r["mean_pair_margin"] for r in rr}
            line.append(f"| {model} | {cl} | {mar[.8]:+.4f} | "
                        f"{mar[1.0]:+.4f} | {mar[1.2]:+.4f} |")
    line+=["","- Pair margin = P(true class) − P(confusable class); positive favors true.",
           "- 크기 변화는 획 굵기, 보간, 일부 픽셀 손상까지 바꿀 수 있으므로 단일한 '크기 단서'의 인과 증거로 과도하게 해석하지 않는다.",
           "- 형상 위상 변화율·잉크 질량 변화·CoM 잔여 오프셋은 exp012_geometry_confound_checks.csv에 보존한다.",
           "- 4명 작성자는 반복 사용한 개발 데이터이며 독립 Test 성능을 의미하지 않는다.",
           "",
           "## 파일 안내","",
           "- exp010_geometry_core_summary.csv: 원본·Size·SizeCoM 분포의 실제 중앙값과 W1.",
           "- exp011_accuracy_matrix.csv: Train Model × Eval Input, 3-seed 각 정확도.",
           "- exp011_paired_summary.csv: 같은 이미지의 corrected/new_wrong, 고정 모델 입력 변화.",
           "- exp011_center_geometry_by_outcome.csv: CoM 복구량과 오답 회복/악화의 관계.",
           "- exp012_scale_pair_margins.csv: 클래스쌍 예측 반응 및 올바른 클래스 Margin.",
           "- exp012_geometry_confound_checks.csv: 위상 변화, 획 질량, CoM 통제 검증.",
           "- all_*: 전수 예측 및 원시 계측 결과(Artifact).",
           "",
           "**최종 모델·main 변경 없음.**"]
    (out/"SUMMARY.md").write_text("\n".join(line)+"\n",encoding="utf-8")
    (out/"run_manifest.json").write_text(json.dumps({
        "experiments":["EXP010","EXP011","EXP012"],
        "models":3,"seeds":list(SEEDS),"models_per_seed":list(METHODS),
        "model_run":43785390739,
        "checkpoint_hash_and_baseline_counts_verified":True,
        "test_used":False,"training_used":False,
        "all_class_geometry_n":len(stat),"scale_response_rows":len(scale),
        "source_sha256":geoman["official_sha256"],
    },indent=2,ensure_ascii=False),encoding="utf-8")
    print("EXP010_011_012_AGGREGATION_COMPLETE",flush=True)
    print("\n".join(line),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.input_root,a.out)
