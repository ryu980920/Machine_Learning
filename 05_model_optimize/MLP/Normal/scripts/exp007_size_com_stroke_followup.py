#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normal HypB EXP007, controlled Stroke addition after Size -> CoM.

Official train 76,800 only for BOTH MLP and fitted stroke normalizer.
Developer evaluations: Official Validation 19,200; existing Hand-made 2,560.
No official Original Test, professor Test or Hand-made learning.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
from joblib import dump
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from exp006_size_com_comparison import (
    ROOT, WRITERS, LABELS, PAIRS, load_official, load_writers,
    PixelScaler255, image_geometry, process, center_batch,
    score, evaluate_group, save_csv, sha256,
)
from stroke_normalization import StrokeWidthNormalizer

CONDITIONS = ("size_com", "size_com_stroke030", "size_com_stroke060")
ALPHA = {"size_com": None, "size_com_stroke030": 0.30,
         "size_com_stroke060": 0.60}


def train(condition, seed, out):
    out = Path(out)
    if out.exists():
        raise RuntimeError("Refusing to overwrite existing result: " + str(out))
    out.mkdir(parents=True)
    official = load_official(ROOT / "02_data" / "datasets" /
                             "MNIST_EMNIST_official_split_112000.npz")
    handmade = load_writers(ROOT / "02_data" / "Hand_made")
    target = int(np.median(image_geometry(official["X_train"])["long_side"]))
    if target != 20:
        raise AssertionError("Expected official Train BBox long-side median=20")

    raw_hm = np.concatenate([handmade[w]["X"] for w in WRITERS])
    label_hm = np.concatenate([handmade[w]["y"] for w in WRITERS])
    writers = np.repeat(np.array(WRITERS, dtype=int), 640)

    # All conditions share identical Combined-derived Size -> CoM input.
    x_train = process(official["X_train"], "size_com", target)
    x_val = process(official["X_val"], "size_com", target)
    x_hm = process(raw_hm, "size_com", target)
    stroke_fit = None
    alpha = ALPHA[condition]
    if alpha is not None:
        stroke_fit = StrokeWidthNormalizer(strength=alpha, threshold=64,
            fit_quantiles=(0.2, 0.8), preserve_topology=True,
            apply_center=False).fit(x_train)
        def apply_stroke_then_center(x):
            x = stroke_fit.transform(x)
            return center_batch(x, "mass").astype(np.float32)
        x_train = apply_stroke_then_center(x_train)
        x_val = apply_stroke_then_center(x_val)
        x_hm = apply_stroke_then_center(x_hm)

    assert x_train.shape == (76800, 784)
    assert x_val.shape == (19200, 784)
    assert x_hm.shape == (2560, 784)
    assert np.isfinite(x_train).all() and x_train.min() >= 0 and x_train.max() <= 255

    model = Pipeline([("scale_0_1", PixelScaler255()),
                      ("model", MLPClassifier(random_state=seed))])
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, official["y_train"])
    duration = time.monotonic() - start
    val_pred = model.predict(x_val).astype(int)
    hm_pred = model.predict(x_hm).astype(int)
    official_scores = evaluate_group(official["y_val"], val_pred)
    handmade_scores = evaluate_group(label_hm, hm_pred, writers)

    cm = confusion_matrix(label_hm, hm_pred, labels=np.arange(16))
    save_csv(out / "pair_errors.csv",
        [{"actual":a, "predicted":b,
          "count":int(cm[LABELS.index(a),LABELS.index(b)])} for a,b in PAIRS])
    save_csv(out / "confusion_matrix.csv",
        [{"actual":LABELS[i],
          **{LABELS[j]:int(cm[i,j]) for j in range(16)}} for i in range(16)])
    with gzip.open(out / "handmade_predictions.csv.gz", "wt",
                   encoding="utf-8", newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["writer","index","true","pred","correct"])
        for k in range(2560):
            writer.writerow([int(writers[k]), k % 640,
                LABELS[int(label_hm[k])],LABELS[int(hm_pred[k])],
                int(label_hm[k] == hm_pred[k])])
    save_csv(out / "loss_curve.csv",
        [{"iteration":i,"loss":float(v)} for i,v in
         enumerate(model.named_steps["model"].loss_curve_, 1)])
    dump(model, out/"mlp.joblib", compress=3)
    before_geo = image_geometry(process(raw_hm, "size_com", target))
    after_geo = image_geometry(x_hm)
    manifest = {
        "experiment":"Normal-HypB-EXP007", "status":"trained",
        "condition":condition, "seed":seed, "stroke_strength":alpha,
        "method":"Size -> CoM -> Stroke -> CoM" if alpha is not None else "Size -> CoM",
        "target_bbox_longside_px":target, "bbox_threshold":32,
        "stroke_threshold":64, "stroke_fit_only_official_train":True,
        "stroke_q20":None if stroke_fit is None else float(stroke_fit.q_low_),
        "stroke_q80":None if stroke_fit is None else float(stroke_fit.q_high_),
        "stroke_median":None if stroke_fit is None else float(stroke_fit.median_),
        "stroke_topology_guard":alpha is not None,
        "official_train_n":76800, "official_validation_n":19200, "handmade_dev_n":2560,
        "official_original_test_used":False, "professor_test_used":False,
        "handmade_training_used":False,
        "official_sha256":official["sha256"],
        "writer_sha256":{str(w):handmade[w]["sha256"] for w in WRITERS},
        "training_seconds":float(duration),
        "mlp_iterations":int(model.named_steps["model"].n_iter_),
        "fit_warnings":[f"{v.category.__name__}: {v.message}" for v in caught],
        "official_validation":official_scores,
        "handmade":handmade_scores,
        "handmade_center_offset_px_before_stroke":float(before_geo["com_offset"].mean()),
        "handmade_center_offset_px_after_stroke":float(after_geo["com_offset"].mean()),
        "handmade_mass_change":float(x_hm.sum()-process(raw_hm,"size_com",target).sum()),
        "numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,
        "python":platform.python_version(),"model_sha256":sha256(out/"mlp.joblib"),
    }
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2,
                        ensure_ascii=False),encoding="utf-8")
    print("EXP007 COMPLETE",json.dumps({
        "condition":condition,"seed":seed,
        "official":official_scores["all"],
        "handmade":handmade_scores},ensure_ascii=False),flush=True)


def aggregate(inputs,out):
    inputs=Path(inputs);out=Path(out)
    if out.exists():
        raise RuntimeError("Refusing to overwrite existing output "+str(out))
    out.mkdir(parents=True)
    files=list(inputs.glob("normal-exp007-*/run_manifest.json"))
    if len(files)!=9:
        raise RuntimeError(f"Expected 9 complete runs, got {len(files)}")
    runs={}; metric_rows=[]
    for p in files:
        j=json.loads(p.read_text(encoding="utf-8"))
        key=(j["condition"],int(j["seed"]))
        if key in runs:raise AssertionError("Duplicate run "+str(key))
        if j["official_original_test_used"] or j["professor_test_used"] or j["handmade_training_used"]:
            raise AssertionError("Data boundary violated")
        runs[key]=(p.parent,j)
        for ds in ("official_validation","handmade"):
            for group,s in j[ds].items():
                metric_rows.append({"condition":key[0],"seed":key[1],
                                    "dataset":ds,"group":group,**s})
    if set(runs)!={(c,s) for c in CONDITIONS for s in (42,43,44)}:
        raise AssertionError("Missing method/seed")
    save_csv(out/"metrics_all.csv",metric_rows)

    def predictions(folder):
        with gzip.open(folder/"handmade_predictions.csv.gz","rt",
                       encoding="utf-8",newline="") as f:
            return {(int(r["writer"]),int(r["index"])):r
                    for r in csv.DictReader(f)}
    paired=[]
    for seed in (42,43,44):
        a=predictions(runs["size_com",seed][0])
        for name in CONDITIONS[1:]:
            b=predictions(runs[name,seed][0])
            if set(a)!=set(b) or len(a)!=2560:
                raise AssertionError("Paired predictions missing")
            for grp in ("all","digits","letters"):
                keys=[k for k in a if grp=="all" or
                      ((a[k]["true"] in "0123456789")==(grp=="digits"))]
                for k in keys:
                    if a[k]["true"]!=b[k]["true"]:
                        raise AssertionError("True labels mismatch")
                corrected=sum(a[k]["correct"]=="0" and b[k]["correct"]=="1" for k in keys)
                induced=sum(a[k]["correct"]=="1" and b[k]["correct"]=="0" for k in keys)
                paired.append({"seed":seed,"candidate":name,"group":grp,
                    "n":len(keys),"corrected":corrected,"new_errors":induced,
                    "net_correct":corrected-induced,
                    "delta_accuracy_pp":100*(corrected-induced)/len(keys)})
    save_csv(out/"paired_vs_size_com.csv",paired)

    lines=["# EXP유007 Size -> CoM -> Stroke -> CoM (review)",
       "", "Only development data were evaluated. Official Test / professor Test unused.",
       "All nine models trained on identical 76,800 official Train rows.",
       "Stroke strength was fixed to 0.30 or 0.60 from earlier HypA experiments.",
       "", "| Condition | Seed | Official Val | Hand-made all | Digits | Letters |",
       "|---|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        for seed in (42,43,44):
            d=runs[c,seed][1]
            lines.append(f"| {c} | {seed} | "
                f"{100*d['official_validation']['all']['accuracy']:.3f}% | "
                f"{100*d['handmade']['all']['accuracy']:.3f}% | "
                f"{100*d['handmade']['digits']['accuracy']:.3f}% | "
                f"{100*d['handmade']['letters']['accuracy']:.3f}% |")
    lines += ["","## Paired accuracy changes relative to Size -> CoM",
       "","| Candidate | Seed | Group | Corrected | Newly wrong | Net |",
       "|---|---:|---|---:|---:|---:|"]
    for d in paired:
        lines.append(f"| {d['candidate']} | {d['seed']} | {d['group']} | "
                     f"{d['corrected']} | {d['new_errors']} | {d['net_correct']} |")
    lines += ["","No automatic final model adoption or GitHub main modification.",
      "New writer independent validation required before generalization claims."]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("EXP007 AGGREGATION COMPLETE",len(metric_rows), "metric rows",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition",choices=CONDITIONS)
    p.add_argument("--seed",type=int,choices=(42,43,44))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--aggregate",action="store_true")
    p.add_argument("--inputs",type=Path)
    a=p.parse_args()
    if a.aggregate:
        if not a.inputs:p.error("--aggregate requires --inputs")
        aggregate(a.inputs,a.out)
    else:
        if a.condition is None or a.seed is None:
            p.error("--condition and --seed required")
        train(a.condition,a.seed,a.out)

if __name__=="__main__":
    main()
