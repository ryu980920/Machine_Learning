#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normal Hypothesis B, EXP006: exact Combined Size/CoM comparison.

Uses only OFFICIAL TRAIN data for fitting (76,800 images); evaluates on
OFFICIAL VALIDATION (19,200) and existing 4-author Hand-made developer data
(2,560). Never accesses Original Test or professor Test. No GitHub writes.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
from joblib import dump
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[4]
CMB = ROOT / "05_model_optimize" / "MLP" / "Combined" / "scripts"
NORM = ROOT / "05_model_optimize" / "MLP" / "Normal" / "scripts"
sys.path.insert(0, str(CMB))
sys.path.insert(0, str(NORM))

# Reuse the exact pixel processing used in Combined EXP-CMB-004.
from combined_mlp_writer_cv import load_official, load_writers, PixelScaler255
from exp_cmb004_geometry_normalization import image_geometry, apply_size
from Automatic_Centering_Transformer import center_batch

WRITERS = (41, 42, 43, 44)
LABELS = "0123456789ABCDEF"
METHODS = ("com", "size", "size_com")
PAIRS = (("B","8"), ("8","B"), ("D","0"), ("0","D"),
         ("5","F"), ("F","5"))


def save_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError("Cannot infer fields for empty CSV")
        fieldnames = list(rows[0])
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process(x, method, target):
    if method == "com":
        a = center_batch(x, "mass")
    elif method == "size":
        a, _ = apply_size(x, target)
    elif method == "size_com":
        a, _ = apply_size(x, target)
        a = center_batch(a, "mass")
    else:
        raise ValueError(method)
    a = np.asarray(a, dtype=np.float32).reshape(-1, 784)
    assert np.isfinite(a).all() and 0 <= a.min() <= a.max() <= 255
    return a


def score(y, pred):
    return {
        "n": int(len(y)),
        "correct": int(np.count_nonzero(y == pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(16),
                                   average="macro", zero_division=0)),
    }


def evaluate_group(y, pred, writer_ids=None):
    output = {"all": score(y, pred),
              "digits": score(y[y < 10], pred[y < 10]),
              "letters": score(y[y >= 10], pred[y >= 10])}
    if writer_ids is not None:
        for w in WRITERS:
            ids = writer_ids == w
            output["writer_" + str(w)] = score(y[ids], pred[ids])
    return output


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def train(method, seed, out):
    out = Path(out)
    if out.exists():
        raise RuntimeError("Refusing to overwrite existing results: " + str(out))
    out.mkdir(parents=True)
    data_path = ROOT / "02_data" / "datasets" / "MNIST_EMNIST_official_split_112000.npz"
    official = load_official(data_path)
    hm = load_writers(ROOT / "02_data" / "Hand_made")
    assert official["X_train"].shape == (76800, 784)
    assert official["X_val"].shape == (19200, 784)
    # No information from validation or handwriting enters this target.
    target = int(np.median(image_geometry(official["X_train"])["long_side"]))
    if target != 20:
        raise AssertionError("Combined target no longer reproducible: " + str(target))
    print("OFFICIAL TRAIN ONLY: fitted target long side =", target, flush=True)

    train_x = process(official["X_train"], method, target)
    val_x = process(official["X_val"], method, target)
    hm_raw = np.concatenate([hm[w]["X"] for w in WRITERS], axis=0)
    hm_y = np.concatenate([hm[w]["y"] for w in WRITERS], axis=0)
    writers = np.repeat(np.array(WRITERS), 640)
    hm_x = process(hm_raw, method, target)

    estimator = Pipeline([
        ("scale_0_1", PixelScaler255()),
        ("model", MLPClassifier(random_state=seed)),
    ])
    t0 = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(train_x, official["y_train"])
    fit_seconds = time.monotonic() - t0
    pred_val = estimator.predict(val_x).astype(int)
    pred_hm = estimator.predict(hm_x).astype(int)
    official_metrics = evaluate_group(official["y_val"], pred_val)
    handmade_metrics = evaluate_group(hm_y, pred_hm, writers)

    # The CoM seed-42 control should reproduce the existing EXP004 model.
    if method == "com" and seed == 42:
        if official_metrics["all"]["correct"] != 18608 or handmade_metrics["all"]["correct"] != 1974:
            raise AssertionError("CoM seed=42 reference mismatch: " +
                                 json.dumps({"official": official_metrics["all"],
                                             "handmade": handmade_metrics["all"]}))
        print("HISTORICAL CoM CONTROL: official 18608/19200, handmade 1974/2560", flush=True)

    cm = confusion_matrix(hm_y, pred_hm, labels=np.arange(16))
    save_csv(out / "confusion_matrix.csv",
             [{"actual": LABELS[i], **{LABELS[j]: int(cm[i, j]) for j in range(16)}}
              for i in range(16)])
    save_csv(out / "pair_errors.csv", [
        {"actual": a, "predicted": b, "count": int(cm[LABELS.index(a), LABELS.index(b)])}
        for a, b in PAIRS
    ])
    precision, recall, f1, support = precision_recall_fscore_support(
        hm_y, pred_hm, labels=np.arange(16), zero_division=0)
    save_csv(out / "per_class.csv", [
        {"class": LABELS[k], "recall": float(recall[k]),
         "precision": float(precision[k]), "f1": float(f1[k]), "support": int(support[k])}
        for k in range(16)
    ])
    with gzip.open(out / "handmade_predictions.csv.gz", "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["writer", "index", "true", "pred", "correct"])
        for k in range(2560):
            writer.writerow([int(writers[k]), k % 640, LABELS[int(hm_y[k])],
                             LABELS[int(pred_hm[k])], int(hm_y[k] == pred_hm[k])])

    save_csv(out / "loss_curve.csv",
             [{"iteration": i, "loss": float(loss)} for i, loss in
              enumerate(estimator.named_steps["model"].loss_curve_, 1)])
    model_path = out / "model.joblib"
    dump(estimator, model_path, compress=3)
    before_train = image_geometry(official["X_train"])
    after_train = image_geometry(train_x)
    before_hm = image_geometry(hm_raw)
    after_hm = image_geometry(hm_x)
    geometry = {
        "train_longside_std_before": float(np.std(before_train["long_side"])),
        "train_longside_std_after": float(np.std(after_train["long_side"])),
        "handmade_com_mean_offset_before": float(np.mean(before_hm["com_offset"])),
        "handmade_com_mean_offset_after": float(np.mean(after_hm["com_offset"])),
        "handmade_mass_before": float(np.sum(hm_raw)),
        "handmade_mass_after": float(np.sum(hm_x)),
    }
    result = {
        "experiment": "Normal-HypB-EXP006",
        "status": "trained",
        "method": method, "seed": seed,
        "target_longside_pixels": target,
        "foreground_threshold": 32,
        "size": "Combined EXP-CMB-004 apply_size: BBox-centered scipy affine_transform, order=1",
        "centering": "Combined EXP-CMB-004 center_batch: CoM to (13.5,13.5), integer shift",
        "pipeline": "process(raw) -> PixelScaler255 -> MLPClassifier(random_state=seed)",
        "model_hyperparameters": {"random_state": seed, "other": "sklearn MLPClassifier defaults"},
        "n_training": 76800, "n_official_validation": 19200, "n_handmade_dev": 2560,
        "official_test_used": False, "professor_test_used": False,
        "handmade_used_in_fit": False, "target_fitted_on_train_only": True,
        "official_sha256": official["sha256"],
        "writer_sha256": {str(w): hm[w]["sha256"] for w in WRITERS},
        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "python": platform.python_version(), "fit_seconds": float(fit_seconds),
        "mlp_iterations": int(estimator.named_steps["model"].n_iter_),
        "fit_warnings": [f"{v.category.__name__}: {v.message}" for v in caught],
        "official_validation": official_metrics,
        "handmade": handmade_metrics,
        "geometry": geometry,
        "model_sha256": sha256(model_path),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPLETE", json.dumps({"method": method, "seed": seed,
        "official": official_metrics["all"], "handmade": handmade_metrics},
        ensure_ascii=False), flush=True)


def aggregate(inputs, out):
    inputs = Path(inputs)
    out = Path(out)
    if out.exists():
        raise RuntimeError("Refusing to overwrite: " + str(out))
    out.mkdir(parents=True)
    files = sorted(inputs.glob("normal-exp006-*/run_manifest.json"))
    if len(files) != 9:
        raise RuntimeError(f"Expected 9 run manifests, got {len(files)}")
    observed = {}
    summaries = []
    for p in files:
        data = json.loads(p.read_text(encoding="utf-8"))
        method, seed = data["method"], int(data["seed"])
        if (method, seed) in observed:
            raise ValueError("Duplicate run: " + str((method,seed)))
        observed[method, seed] = p.parent
        if data["official_test_used"] or data["professor_test_used"] or data["handmade_used_in_fit"]:
            raise AssertionError("Data boundary violated")
        for group, metrics in data["handmade"].items():
            summaries.append({"method": method, "seed": seed, "dataset": "handmade",
                "group": group, **metrics})
        for group, metrics in data["official_validation"].items():
            summaries.append({"method": method, "seed": seed, "dataset": "official_val",
                "group": group, **metrics})
    if set(observed) != {(m,seed) for m in METHODS for seed in (42,43,44)}:
        raise AssertionError("Method-seed mismatch")
    save_csv(out / "all_metrics.csv", summaries)

    def read_predictions(path):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            return {(int(r["writer"]), int(r["index"])): r
                    for r in csv.DictReader(f)}
    paired = []
    for seed in (42,43,44):
        base = read_predictions(observed["com",seed] / "handmade_predictions.csv.gz")
        for compared in ("size", "size_com"):
            other = read_predictions(observed[compared,seed] / "handmade_predictions.csv.gz")
            if set(base) != set(other) or len(base) != 2560:
                raise AssertionError("Paired prediction index mismatch")
            groups = {"all": [], "digits": [], "letters": []}
            for key in sorted(base):
                a, b = base[key], other[key]
                if a["true"] != b["true"]:
                    raise AssertionError("Label mismatch")
                grp = "digits" if a["true"] in "0123456789" else "letters"
                groups["all"].append((a,b))
                groups[grp].append((a,b))
            for grp, rows in groups.items():
                repaired = sum(int(a["correct"]) == 0 and int(b["correct"]) == 1 for a,b in rows)
                induced = sum(int(a["correct"]) == 1 and int(b["correct"]) == 0 for a,b in rows)
                paired.append({"seed":seed,"control":"com","candidate":compared,
                    "group":grp,"n":len(rows),"corrected":repaired,"new_errors":induced,
                    "net_correct":repaired-induced,
                    "delta_accuracy_pp":100*(repaired-induced)/len(rows)})
    save_csv(out / "paired_vs_com.csv", paired)

    lines = [
      "# EXP유006 — Normal Size/CoM comparison (review only)",
      "",
      "All values below are measurements from GitHub Actions; no Test data were used.",
      "Methods are CoM-only, BBox Size-only, and BBox Size -> CoM.",
      "Each model was freshly fitted on the SAME 76,800 official Train images,",
      "using sklearn MLPClassifier defaults and seeds 42,43,44.",
      "Target BBox long side 20px was fitted on official Training only.",
      "",
      "## Per-seed accuracy",
      "",
      "| Method | Seed | Official Val | Hand-made all | Digits | Letters |",
      "|---|---:|---:|---:|---:|---:|"
    ]
    for method in METHODS:
        for seed in (42,43,44):
            r=json.loads((observed[method,seed]/"run_manifest.json").read_text(encoding="utf-8"))
            o=r["official_validation"]["all"]["accuracy"]
            h=r["handmade"]
            lines.append(f"| {method} | {seed} | {100*o:.3f}% | "
                         f"{100*h['all']['accuracy']:.3f}% | "
                         f"{100*h['digits']['accuracy']:.3f}% | "
                         f"{100*h['letters']['accuracy']:.3f}% |")
    lines += [
      "", "## Paired Size+CoM vs CoM-only", "",
      "| Seed | Group | Corrections | Newly wrong | Net |",
      "|---|---|---:|---:|---:|"
    ]
    for r in paired:
        if r["candidate"] == "size_com":
            lines.append(f"| {r['seed']} | {r['group']} | {r['corrected']} | "
                         f"{r['new_errors']} | {r['net_correct']} |")
    lines += ["", "This is developer-data evidence, not an independent new-writer Test.",
              "Official Original Test and professor Test were never opened.",
              "Stroke normalization is OUT OF SCOPE. No model or pipeline is adopted.",
              "Review official Val, digits/letters and writer-level tradeoffs before next stage."]
    (out / "SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("EXP006 AGGREGATION COMPLETE", len(summaries), "summary rows", flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--seed", type=int, choices=(42,43,44))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--inputs", type=Path)
    a=p.parse_args()
    if a.aggregate:
        if not a.inputs:
            p.error("--aggregate requires --inputs")
        aggregate(a.inputs, a.out)
    else:
        if a.method is None or a.seed is None:
            p.error("--method and --seed are required")
        train(a.method, a.seed, a.out)

if __name__ == "__main__":
    main()
