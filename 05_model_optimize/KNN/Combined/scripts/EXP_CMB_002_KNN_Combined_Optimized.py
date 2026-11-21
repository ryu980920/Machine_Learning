#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-CMB-002: frozen optimized KNN Normal versus +3-writer Combined, 4-writer OOF.

Run from repository root:
 python 05_model_optimize/KNN/Combined/scripts/EXP_CMB_002_KNN_Combined_Optimized.py

Official Test is NEVER loaded. Every held-out writer is excluded from that fold's
training rows, but the frozen Normal configuration (including weak classes) was
selected with the same 4-writer development set in earlier EXP-013/014. Thus the
OOF result is *development OOF*, NOT an independent unbiased generalization test.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[4]
NORMAL = ROOT / "05_model_optimize" / "KNN" / "Normal" / "scripts"
COMBINED = ROOT / "05_model_optimize" / "KNN" / "Combined" / "scripts"
sys.path.insert(0, str(NORMAL))
sys.path.insert(0, str(COMBINED))
sys.path.insert(0, str(ROOT / "03_experiments" / "model_optimization"))

from common import LABELS, fixed_split
from EXP004_KNN_Normal_CoM import BatchedCoM
from final_knn_normal import assemble_training, prepare_features, EXPECTED_VALIDATION_CORRECT
from EXP_CMB_001_KNN_Combined_Baseline import (
    WRITERS, load_writer, summary, save_csv, save_cm,
)

OUT = ROOT / "05_model_optimize" / "KNN" / "Combined" / "results" / "EXP_CMB_002_optimized_4writer_oof"
EXPECTED_NORMAL_HANDMADE = 2381  # EXP-014 targeted_10 / EXP-015 k3_distance


def fresh_knn():
    return KNeighborsClassifier(n_neighbors=3, weights="distance", metric="euclidean",
                                algorithm="brute", n_jobs=-1)


def paired_counts(y, before, after):
    return {
        "corrected": int(((before != y) & (after == y)).sum()),
        "new_errors": int(((before == y) & (after != y)).sum()),
        "unchanged_correct": int(((before == y) & (after == y)).sum()),
        "unchanged_wrong": int(((before != y) & (after != y)).sum()),
    }


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    xtr, ytr, xva, yva, sha = fixed_split()
    if (len(ytr), len(yva)) != (76800, 19200):
        raise RuntimeError("Unexpected official development split")
    print("Official Training=76800; Official Validation=19200; Official Test NOT loaded", flush=True)
    handmade = {w: load_writer(w) for w in WRITERS}
    com = BatchedCoM()
    t0 = time.perf_counter()

    x_train_opt, y_train_opt = assemble_training(xtr, ytr, com)
    del xtr, ytr
    if x_train_opt.shape != (107520, 1108):
        raise RuntimeError("Frozen Normal training feature count mismatch")
    print("[Preprocessing] Official Validation and four writer sets...", flush=True)
    fv = prepare_features(xva, com, "official Validation")
    del xva
    fw = {w: prepare_features(handmade[w][0], com, f"handmade writer {w}") for w in WRITERS}

    print("[Optimized Normal] Official Training + official augmentation only...", flush=True)
    normal = fresh_knn()
    normal.fit(x_train_opt, y_train_opt)
    normal_off_pred = normal.predict(fv).astype(np.int64)
    normal_off = summary(yva, normal_off_pred)
    if normal_off["correct"] != EXPECTED_VALIDATION_CORRECT:
        raise RuntimeError(f"Frozen Normal Validation mismatch: {normal_off['correct']} vs 18851")
    normal_pred = {w: normal.predict(fw[w]).astype(np.int64) for w in WRITERS}
    truth = np.concatenate([handmade[w][1] for w in WRITERS])
    n_oof = np.concatenate([normal_pred[w] for w in WRITERS])
    n_result = summary(truth, n_oof)
    if n_result["correct"] != EXPECTED_NORMAL_HANDMADE:
        raise RuntimeError(f"Frozen Normal handwriting mismatch: {n_result['correct']} vs 2381")
    print(f"Normal EXACTLY REPRODUCED: Official={normal_off['accuracy']:.6f}; "
          f"Hand-made={n_result['accuracy']:.6f}", flush=True)
    del normal

    combined_pred = {}
    fold_rows = []
    official_rows = [{"variant": "optimized_normal", "heldout_writer": "all",
                      "n": normal_off["n"], "correct": normal_off["correct"],
                      "accuracy": normal_off["accuracy"], "macro_f1": normal_off["macro_f1"]}]

    for heldout in WRITERS:
        other = [w for w in WRITERS if w != heldout]
        extra_x = np.concatenate([fw[w] for w in other], axis=0)
        extra_y = np.concatenate([handmade[w][1] for w in other], axis=0)
        if extra_x.shape != (1920, 1108) or len(extra_y) != 1920:
            raise RuntimeError("Wrong 3-writer training rows")
        fit_x = np.concatenate([x_train_opt, extra_x], axis=0)
        fit_y = np.concatenate([y_train_opt, extra_y], axis=0)
        if fit_x.shape != (109440, 1108) or heldout in other:
            raise RuntimeError("Fit shape or held-out writer leakage")
        del extra_x, extra_y
        print(f"\n[Fold heldout={heldout}] Train writers={other}; "
              f"107520 official-derived + 1920 writer rows = 109440", flush=True)
        model = fresh_knn()
        t = time.perf_counter()
        model.fit(fit_x, fit_y)
        fit_seconds = time.perf_counter() - t
        del fit_x, fit_y
        t = time.perf_counter()
        pred_hm = model.predict(fw[heldout]).astype(np.int64)
        hand_predict_seconds = time.perf_counter() - t
        t = time.perf_counter()
        pred_off = model.predict(fv).astype(np.int64)
        off_predict_seconds = time.perf_counter() - t
        del model
        combined_pred[heldout] = pred_hm
        cr = summary(handmade[heldout][1], pred_hm)
        nr = summary(handmade[heldout][1], normal_pred[heldout])
        ov = summary(yva, pred_off)
        fold_rows.append({
            "heldout_writer": heldout, "train_writers": ",".join(map(str, other)),
            "official_derived_train": 107520, "handmade_train": 1920,
            "combined_train_n": 109440, "heldout_n": 640,
            "normal_correct": nr["correct"], "normal_accuracy": nr["accuracy"],
            "normal_macro_f1": nr["macro_f1"], "combined_correct": cr["correct"],
            "combined_accuracy": cr["accuracy"], "combined_macro_f1": cr["macro_f1"],
            "combined_official_accuracy": ov["accuracy"],
            "combined_official_macro_f1": ov["macro_f1"],
            "fit_s": round(fit_seconds, 3),
            "handmade_predict_s": round(hand_predict_seconds, 3),
            "official_predict_s": round(off_predict_seconds, 3),
        })
        official_rows.append({"variant": "combined", "heldout_writer": heldout,
                              "n": ov["n"], "correct": ov["correct"],
                              "accuracy": ov["accuracy"], "macro_f1": ov["macro_f1"]})
        save_csv(OUT / "fold_results_partial.csv", fold_rows)
        save_csv(OUT / "official_validation_partial.csv", official_rows)
        print(f"   Hand-made Normal={nr['accuracy']:.6f} / Combined={cr['accuracy']:.6f}; "
              f"Official Validation={ov['accuracy']:.6f}", flush=True)

    c_oof = np.concatenate([combined_pred[w] for w in WRITERS])
    c_result = summary(truth, c_oof)
    paired = paired_counts(truth, n_oof, c_oof)
    preds = []
    for offset, w in enumerate(WRITERS):
        for i in range(640):
            j = offset * 640 + i
            preds.append({"writer": w, "index": i, "true": LABELS[int(truth[j])],
                          "normal_pred": LABELS[int(n_oof[j])],
                          "combined_pred": LABELS[int(c_oof[j])],
                          "normal_correct": int(n_oof[j] == truth[j]),
                          "combined_correct": int(c_oof[j] == truth[j])})
    save_csv(OUT / "fold_results.csv", fold_rows)
    save_csv(OUT / "official_validation.csv", official_rows)
    save_csv(OUT / "oof_predictions.csv", preds)
    save_cm(OUT / "normal_oof_confusion_matrix.csv", truth, n_oof)
    save_cm(OUT / "combined_oof_confusion_matrix.csv", truth, c_oof)
    save_csv(OUT / "per_class_oof.csv", [
        {"class": label, "support": n_result["per_class"][label]["support"],
         "normal_recall": n_result["per_class"][label]["recall"],
         "normal_f1": n_result["per_class"][label]["f1"],
         "combined_recall": c_result["per_class"][label]["recall"],
         "combined_f1": c_result["per_class"][label]["f1"]}
        for label in LABELS])
    report = {
        "experiment": "EXP-CMB-002: optimized Normal vs raw 3-writer Combined, development 4-writer OOF",
        "official_dataset_sha256": sha,
        "handmade_sha256": {str(w): handmade[w][2] for w in WRITERS},
        "official_train_source": 76800, "official_derived_train": 107520,
        "handmade_train_per_fold": 1920, "combined_train_per_fold": 109440,
        "official_val": 19200, "heldout_per_fold": 640,
        "frozen_model": {"k": 3, "weights": "distance", "metric": "euclidean",
                         "preprocessing": "CoM -> Deskew", "features": "Pixel 20% + HOG 80%",
                         "base_aug": "30% size+stroke, strength 0.44",
                         "additional_aug": "D/B/5/3 7680 official-training-derived rows"},
        "normal_oof": n_result, "combined_oof": c_result,
        "normal_official_validation": normal_off,
        "official_validation_per_fold": official_rows,
        "folds": fold_rows, "paired_handmade": paired,
        "official_test_used": False,
        "warning": "All 4 Hand-made writers were used to choose the frozen Normal feature and targeted classes in earlier EXP-013/014. Despite per-fold writer exclusion from FIT, this is development OOF and cannot be claimed independent/unbiased writer generalization.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "sklearn": sklearn.__version__},
        "elapsed_seconds": round(time.perf_counter()-t0, 3),
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n4-WRITER DEVELOPMENT OOF SUMMARY", flush=True)
    print(f"Optimized Normal   Accuracy={n_result['accuracy']:.6f}, "
          f"MacroF1={n_result['macro_f1']:.6f} ({n_result['correct']}/2560)", flush=True)
    print(f"Optimized Combined Accuracy={c_result['accuracy']:.6f}, "
          f"MacroF1={c_result['macro_f1']:.6f} ({c_result['correct']}/2560)", flush=True)
    print(f"Change={100*(c_result['accuracy']-n_result['accuracy']):+.4f} percentage points", flush=True)
    print(f"Paired={paired}", flush=True)
    print(f"Saved: {OUT}", flush=True)
    print("No official Test was loaded. OOF is development-only due to prior Hand-made model selection.", flush=True)


if __name__ == "__main__":
    run()
