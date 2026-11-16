#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normal Hypothesis B EXP008: ORIGINAL HypA C9 then optional Stroke.

This is deliberately based on the byte-identical HypA EXP004 Stage2 implementation:
  exp_a004_followup.Stage2Preprocessing("C9_BBoxLongside20")
  -> optional StrokeWidthNormalizer(strength=.30/.60) fitted on C9-transformed
     official Training ONLY -> final center_of_mass.
MLP fit/evaluation is performed by the original exp_a004_official_mlp.run()
to preserve all hyperparameters (200 iterations, tol=0, seeds 42/43/44).

No official Original Test, professor Test, or handmade training.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import cv2
from stroke_normalization import StrokeWidthNormalizer, center_of_mass
from exp_a004_followup import Stage2Preprocessing, test_preprocessors
import exp_a004_official_mlp as original

CONDITIONS=("C9_Original", "C9_Stroke030", "C9_Stroke060")
STROKE_STRENGTH={"C9_Original":None,"C9_Stroke030":.30,"C9_Stroke060":.60}

class C9WithStroke:
    def __init__(self, policy):
        if policy not in CONDITIONS:
            raise ValueError(policy)
        self.policy=policy

    def fit(self, X):
        self.c9=Stage2Preprocessing("C9_BBoxLongside20").fit(X)
        # Exactly the original HypA C9 output, including its ORIGINAL CoM.
        pre_c9=self.c9.transform(X)
        self.stroke=None
        self.stroke_strength=STROKE_STRENGTH[self.policy]
        if self.stroke_strength is not None:
            self.stroke=StrokeWidthNormalizer(
                strength=self.stroke_strength, threshold=64,
                fit_quantiles=(.2,.8), preserve_topology=True,
                apply_center=False).fit(pre_c9)
        self.train_count_=len(X)
        return self

    def transform(self, X):
        c9=self.c9.transform(X)
        if self.stroke is None:
            return c9  # EXACT same transformed pixels as original C9 run.
        stroked=self.stroke.transform(c9).reshape(-1,28,28)
        out=np.empty((len(stroked),784),np.float32)
        for i,im in enumerate(stroked):
            out[i]=center_of_mass(im).reshape(-1)
        return out

    def description(self):
        meta={
            "experiment":"Normal-HypB-EXP008",
            "policy":self.policy,
            "hypA_C9_exact_code_source_branch":
                "exp/normal-a-exp004-stroke-normalization-20261116",
            "c9_original":self.c9.description(),
            "order":"HypA C9 Size -> CoM -> optional Stroke -> CoM",
            "official_train_only":True,
            "training_count":self.train_count_,
            "uses_handmade_or_labels_to_fit":False,
            "original_test_used":False,"professor_test_used":False
        }
        if self.stroke is not None:
            meta["stroke"]={
                "strength":self.stroke.strength, "threshold":self.stroke.threshold,
                "fitted_count":self.stroke.n_fitted_,
                "q20":float(self.stroke.q_low_),
                "q80":float(self.stroke.q_high_),
                "median":float(self.stroke.median_),
                "topology_guard":self.stroke.preserve_topology,
            }
        return meta


def self_test():
    test_preprocessors()
    a=np.zeros((40,28,28),np.float32)
    for i in range(40):
        cv2.line(a[i],(4+i%7,5),(4+i%7,23),255,1+i%4)
    source=Stage2Preprocessing("C9_BBoxLongside20").fit(a).transform(a)
    for condition in CONDITIONS:
        p=C9WithStroke(condition).fit(a)
        x=p.transform(a)
        assert x.shape==(40,784) and np.isfinite(x).all()
        assert 0 <= x.min() <= x.max() <= 255
        assert np.array_equal(x,p.transform(a))
        if condition=="C9_Original":
            assert np.array_equal(x,source),"C9 original pixels differ"
    print("EXACT C9 + OPTIONAL STROKE PREPROCESSOR SELF-TEST PASSED",flush=True)


# EXP004 C9 Stage2 original per-seed source, for same-data reproduction checks.
REFERENCE={
  42:{"OfficialVal":18457,"digits":1370,"letters":891,"handmade_all":2261},
  43:{"OfficialVal":18211,"digits":1353,"letters":861,"handmade_all":2214},
  44:{"OfficialVal":18380,"digits":1374,"letters":868,"handmade_all":2242},
}

def read_csv(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def save_csv(path,rows):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def aggregate(root,out):
    root=Path(root)
    out=Path(out)
    if out.exists(): raise RuntimeError("Existing output directory: "+str(out))
    out.mkdir(parents=True)
    folders={c:root/("normal-exp008-"+c) for c in CONDITIONS}
    summary=[]
    by_key={}
    for condition,folder in folders.items():
        manifest=json.loads((folder/"run_manifest.json").read_text(encoding="utf-8"))
        if manifest["official_original_test_used"] or manifest["professor_test_used"]:
            raise AssertionError("Test data used")
        group=read_csv(folder/"group_metrics.csv")
        preds=read_csv(folder/"handmade_predictions.csv")
        if len(preds)!=3*2560:raise AssertionError("Need 2560 predictions per seed")
        for seed in (42,43,44):
            def score(dataset,g):
                matches=[z for z in group if z["condition"]==condition and
                         z["seed"]==str(seed) and z["dataset"]==dataset and
                         z["group"]==g]
                if len(matches)!=1:raise AssertionError(f"Missing {condition} {seed} {dataset} {g}")
                return matches[0]
            v,h,d,l=(score("OfficialVal","all"),score("Handmade","all"),
                       score("Handmade","digits"),score("Handmade","letters"))
            summary.append({
                "condition":condition,"seed":seed,
                "official_val_correct":int(v["correct"]),
                "official_val_acc":float(v["accuracy"]),
                "handmade_correct":int(h["correct"]),
                "handmade_acc":float(h["accuracy"]),
                "digits_correct":int(d["correct"]),
                "digits_acc":float(d["accuracy"]),
                "letters_correct":int(l["correct"]),
                "letters_acc":float(l["accuracy"]),
            })
            p={ (int(z["writer"]),int(z["index_in_writer"])):z
                for z in preds if z["seed"]==str(seed) and z["condition"]==condition}
            if len(p)!=2560:raise AssertionError("Duplicate/missing writer row")
            by_key[condition,seed]=p
    save_csv(out/"all_metrics.csv",summary)

    paired=[]
    for seed in (42,43,44):
        a=by_key["C9_Original",seed]
        for candidate in CONDITIONS[1:]:
            b=by_key[candidate,seed]
            if set(a)!=set(b):raise AssertionError("Paired indices differ")
            for group in ("all","digits","letters"):
                keys=[k for k in a if group=="all" or
                     ((a[k]["actual"] in "0123456789")==(group=="digits"))]
                corrected=induced=0
                for k in keys:
                    if a[k]["actual"]!=b[k]["actual"]:
                        raise AssertionError("True label mismatch")
                    gooda=a[k]["actual"]==a[k]["predicted"]
                    goodb=b[k]["actual"]==b[k]["predicted"]
                    corrected+=int((not gooda) and goodb)
                    induced+=int(gooda and (not goodb))
                paired.append({"candidate":candidate,"seed":seed,"group":group,"n":len(keys),
                               "corrected":corrected,"new_wrong":induced,
                               "net_correct":corrected-induced,
                               "delta_accuracy_pp":100*(corrected-induced)/len(keys)})
    save_csv(out/"paired_vs_c9.csv",paired)
    expected=[]
    for x in summary:
        if x["condition"]!="C9_Original":continue
        ref=REFERENCE[x["seed"]]
        expected.append({
            "seed":x["seed"],
            "official_delta_correct":x["official_val_correct"]-ref["OfficialVal"],
            "handmade_delta_correct":x["handmade_correct"]-ref["handmade_all"],
            "digits_delta_correct":x["digits_correct"]-ref["digits"],
            "letters_delta_correct":x["letters_correct"]-ref["letters"],
        })
    save_csv(out/"c9_reference_reproduction.csv",expected)
    all_equal=all(all(z[k]==0 for k in z if k!="seed") for z in expected)
    lines=[
        "# Normal 가설 B EXP유008 — HypA C9 + Stroke 결과",
        "",
        "- C9 baseline uses **exact imported HypA EXP004 Stage2 source** and original 200-iteration MLP settings.",
        "- Stroke is fitted only on C9-processed Official Train; threshold64 and strengths .30/.60.",
        "- All runs: Official Train 76800; Official Validation 19200; Handmade Dev 2560.",
        "- **No Official Original Test, no professor Test, no Handmade fit.**",
        "- C9 original 3-seed reproduces recorded EXP004 values: **"+("YES" if all_equal else "NO; see c9_reference_reproduction.csv")+"**.",
        "",
        "## Seed-level accuracy",
        "",
        "| Method | Seed | Official Val | Handmade | Digits | Letters |",
        "|---|---:|---:|---:|---:|---:|"
    ]
    for x in sorted(summary,key=lambda x:(CONDITIONS.index(x["condition"]),x["seed"])):
        lines.append(f"| {x['condition']} | {x['seed']} | "
                     f"{100*x['official_val_acc']:.4f}% | "
                     f"{100*x['handmade_acc']:.4f}% | "
                     f"{100*x['digits_acc']:.4f}% | "
                     f"{100*x['letters_acc']:.4f}% |")
    lines += ["","## Three-seed mean", "",
              "| Method | Official Val | Handmade | Digits | Letters |",
              "|---|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        arr=[x for x in summary if x["condition"]==c]
        mean=lambda key:100*sum(x[key] for x in arr)/len(arr)
        lines.append(f"| {c} | {mean('official_val_acc'):.4f}% | "
                     f"{mean('handmade_acc'):.4f}% | {mean('digits_acc'):.4f}% | "
                     f"{mean('letters_acc'):.4f}% |")
    lines += ["","## Paired corrections/new errors",
              "",
              "| Method | Seed | Group | Corrected | New errors | Net |",
              "|---|---:|---|---:|---:|---:|"]
    for x in paired:
        lines.append(f"| {x['candidate']} | {x['seed']} | {x['group']} | "
                     f"{x['corrected']} | {x['new_wrong']} | {x['net_correct']} |")
    lines += ["",
             "## Interpretation boundary",
             "All handwritten evaluation uses previously examined writers 41-44; no independent writer test.",
             "The original HypA C9 is **not** the Combined-derived EXP007 size implementation.",
             "Compare against historical EXP007 only with the training budget difference disclosed.",
             "No final Pipeline adoption or main-branch merge is implied by experiment completion."]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("EXP008 SUMMARY COMPLETE; original C9 reproduced =",all_equal,
          json.dumps(expected,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition",choices=CONDITIONS)
    p.add_argument("--out",type=Path)
    p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[4])
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--aggregate",action="store_true")
    p.add_argument("--input-root",type=Path)
    a=p.parse_args()
    if a.self_test:
        self_test()
        return
    if a.out is None:p.error("--out required")
    if a.aggregate:
        if a.input_root is None:p.error("--aggregate requires --input-root")
        aggregate(a.input_root,a.out)
    else:
        if a.condition is None:p.error("condition required")
        if a.out.exists():p.error("Refuse to overwrite existing directory "+str(a.out))
        original.Preprocessing=C9WithStroke
        original.run(a.repo_root,a.out,a.condition,[42,43,44],200)


if __name__=="__main__":
    main()
