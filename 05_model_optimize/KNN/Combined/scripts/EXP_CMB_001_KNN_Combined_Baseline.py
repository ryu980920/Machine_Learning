#!/usr/bin/env python3
"""EXP-CMB-001 KNN: Normal versus Combined with 4-writer held-out OOF.

Run from repository root:
 python 05_model_optimize/KNN/Combined/scripts/EXP_CMB_001_KNN_Combined_Baseline.py

Only official training/development validation and writer-separated handwriting are used.
The official Original Test and professor's tests are NEVER read.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "03_experiments" / "model_optimization"))
from common import fixed_split, pipeline, LABELS
from knn import build_baseline

WRITERS = (41, 42, 43, 44)
EXPECTED_HANDMADE_SHA = {
    41: "0a829172a72787c770139c73de34cc8b751181829ff6fe08ec260247b9be5a2f",
    42: "f364fb453d05b4fe2a705c2b8a80683b404608c7c257d664e3ec7ac86932d4c5",
    43: "10db20997c727c1bc1f1299f8e89626fadd3ec9618468d6759c86b0d2ddc4a66",
    44: "0e266a36e7b29edfcca0514836b5c383d13939948b4e335d70f61bb40bdc084d",
}
OUT = ROOT / "05_model_optimize" / "KNN" / "Combined" / "results" / "EXP_CMB_001_baseline_4writer_oof"


def save_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ids = np.arange(16)
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    return {
        "n": int(y_true.size), "correct": int((y_true == y_pred).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "per_class": {label: {"precision": float(p[i]), "recall": float(r[i]),
                              "f1": float(f[i]), "support": int(n[i])}
                      for i, label in enumerate(LABELS)},
    }


def save_cm(path: Path, truth: np.ndarray, predictions: np.ndarray) -> None:
    cm = confusion_matrix(truth, predictions, labels=np.arange(16))
    save_csv(path, [{"true\\pred": LABELS[i], **{LABELS[j]: int(cm[i, j]) for j in range(16)}}
                    for i in range(16)])


def load_writer(writer: int) -> tuple[np.ndarray, np.ndarray, str]:
    path = ROOT / "02_data" / "Hand_made" / f"ml26_{writer}.npz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_HANDMADE_SHA[writer]:
        raise ValueError(f"Writer {writer} SHA-256 mismatch: {digest}")
    with np.load(path, allow_pickle=False) as d:
        if set(d.files) != {"img", "label", "label_id"}:
            raise ValueError(f"Unexpected NPZ keys: {path}")
        images = d["img"]
        y = d["label_id"].astype(np.int64)
        text_labels = d["label"]
    if images.shape != (640, 28, 28) or images.dtype != np.uint8 or y.shape != (640,):
        raise ValueError(f"Writer {writer}: unexpected image or label shapes")
    if not np.array_equal(y, np.array([LABELS.index(str(v)) for v in text_labels])):
        raise ValueError(f"Writer {writer}: inconsistent label encoding")
    if not np.all(np.bincount(y, minlength=16) == 40):
        raise ValueError(f"Writer {writer}: class counts differ from 40")
    return images.reshape(640, 784).astype(np.float32), y, digest


def fresh_model():
    knn, scaled = build_baseline()
    if (knn.n_neighbors, knn.weights, knn.p, scaled) != (5, "uniform", 2, True):
        raise ValueError("Baseline definition changed: expected k=5, uniform, Euclidean, pixel/255")
    return pipeline(knn, scaled)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    x_off_train, y_off_train, x_off_val, y_off_val, data_sha = fixed_split()
    handmade = {w: load_writer(w) for w in WRITERS}
    assert len(x_off_train) == 76800 and len(x_off_val) == 19200
    print(f"Official Train={len(y_off_train)}; Validation={len(y_off_val)}; Hand-made=2560", flush=True)
    print("Official Test is not loaded.", flush=True)

    start = time.perf_counter()
    normal = fresh_model()
    print("[Normal] Fitting official Training only...", flush=True)
    normal.fit(x_off_train, y_off_train)
    normal_fit = time.perf_counter() - start
    normal_off_pred = normal.predict(x_off_val).astype(np.int64)
    normal_official = summary(y_off_val, normal_off_pred)
    print(f"[Normal] Official Validation={normal_official['accuracy']:.6f}; "
          f"MacroF1={normal_official['macro_f1']:.6f}", flush=True)
    if normal_official["correct"] != 18488:
        raise RuntimeError("Normal official baseline did not reproduce EXP-001 (18488/19200). "
                           "Check environment/data and stop before Combined interpretation.")

    normal_preds = {w: normal.predict(handmade[w][0]).astype(np.int64) for w in WRITERS}
    combined_preds = {}
    fold_rows = []
    official_rows = [{"model": "normal", "heldout_writer": "all", **{k: normal_official[k] for k in ("n", "correct", "accuracy", "macro_f1")}}]

    for heldout in WRITERS:
        fit_writers = [w for w in WRITERS if w != heldout]
        x_extra = np.concatenate([handmade[w][0] for w in fit_writers], axis=0)
        y_extra = np.concatenate([handmade[w][1] for w in fit_writers], axis=0)
        x_train = np.concatenate([x_off_train, x_extra], axis=0)
        y_train = np.concatenate([y_off_train, y_extra], axis=0)
        if len(x_train) != 78720 or len(y_extra) != 1920:
            raise RuntimeError("Combined training size mismatch")
        if heldout in fit_writers:
            raise RuntimeError("Hand-made writer leakage")
        print(f"\n[Fold heldout={heldout}] Train writers={fit_writers}; rows={len(y_train)}", flush=True)
        model = fresh_model()
        t = time.perf_counter()
        model.fit(x_train, y_train)
        fit_s = time.perf_counter() - t
        t = time.perf_counter()
        pred = model.predict(handmade[heldout][0]).astype(np.int64)
        handwritten_s = time.perf_counter() - t
        t = time.perf_counter()
        off_pred = model.predict(x_off_val).astype(np.int64)
        official_s = time.perf_counter() - t
        combined_preds[heldout] = pred
        hand_result = summary(handmade[heldout][1], pred)
        normal_result = summary(handmade[heldout][1], normal_preds[heldout])
        off_result = summary(y_off_val, off_pred)
        fold_rows.append({
            "heldout_writer": heldout, "train_writers": ",".join(map(str, fit_writers)),
            "official_train_n": 76800, "handmade_train_n": 1920, "train_n": 78720,
            "heldout_n": 640, "normal_accuracy": normal_result["accuracy"],
            "normal_macro_f1": normal_result["macro_f1"],
            "combined_accuracy": hand_result["accuracy"],
            "combined_macro_f1": hand_result["macro_f1"],
            "combined_correct": hand_result["correct"],
            "combined_official_val_accuracy": off_result["accuracy"],
            "combined_official_val_macro_f1": off_result["macro_f1"],
            "fit_s": round(fit_s, 3), "handmade_predict_s": round(handwritten_s, 3),
            "official_val_predict_s": round(official_s, 3),
        })
        official_rows.append({"model": "combined", "heldout_writer": heldout,
                              **{k: off_result[k] for k in ("n", "correct", "accuracy", "macro_f1")}})
        print(f"   Hand-made Normal={normal_result['accuracy']:.6f} / "
              f"Combined={hand_result['accuracy']:.6f}, "
              f"Official Validation={off_result['accuracy']:.6f}", flush=True)
        save_csv(OUT / "fold_results_partial.csv", fold_rows)
        save_csv(OUT / "official_validation_partial.csv", official_rows)

    truth = np.concatenate([handmade[w][1] for w in WRITERS])
    normal_oof = np.concatenate([normal_preds[w] for w in WRITERS])
    combined_oof = np.concatenate([combined_preds[w] for w in WRITERS])
    nres, cres = summary(truth, normal_oof), summary(truth, combined_oof)
    paired = {
        "corrected": int(((normal_oof != truth) & (combined_oof == truth)).sum()),
        "new_errors": int(((normal_oof == truth) & (combined_oof != truth)).sum()),
        "unchanged_correct": int(((normal_oof == truth) & (combined_oof == truth)).sum()),
        "unchanged_wrong": int(((normal_oof != truth) & (combined_oof != truth)).sum()),
    }
    predictions = []
    for offset, w in enumerate(WRITERS):
        y = handmade[w][1]
        for i in range(640):
            idx = offset * 640 + i
            predictions.append({"writer": w, "index": i, "true": LABELS[int(y[i])],
                                "normal_pred": LABELS[int(normal_oof[idx])],
                                "combined_pred": LABELS[int(combined_oof[idx])],
                                "normal_correct": int(normal_oof[idx] == y[i]),
                                "combined_correct": int(combined_oof[idx] == y[i])})

    save_csv(OUT / "fold_results.csv", fold_rows)
    save_csv(OUT / "official_validation.csv", official_rows)
    save_csv(OUT / "oof_predictions.csv", predictions)
    save_cm(OUT / "normal_oof_confusion_matrix.csv", truth, normal_oof)
    save_cm(OUT / "combined_oof_confusion_matrix.csv", truth, combined_oof)
    save_csv(OUT / "per_class_oof.csv", [
        {"class": label, "normal_recall": nres["per_class"][label]["recall"],
         "normal_f1": nres["per_class"][label]["f1"],
         "combined_recall": cres["per_class"][label]["recall"],
         "combined_f1": cres["per_class"][label]["f1"],
         "support": nres["per_class"][label]["support"]}
        for label in LABELS
    ])
    report = {
        "experiment": "EXP-CMB-001 KNN Baseline / 4-Writer OOF",
        "official_train": 76800, "official_validation": 19200,
        "combined_train_per_fold": 78720, "handmade_train_per_fold": 1920,
        "heldout_per_fold": 640, "handmade_total": 2560,
        "data_sha256": data_sha,
        "handmade_sha256": {str(w): handmade[w][2] for w in WRITERS},
        "model": "KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)",
        "scaling": "pixel / 255",
        "official_test_used": False,
        "normal_oof": nres, "combined_oof": cres,
        "paired_handmade": paired,
        "folds": fold_rows, "official_validation_by_model": official_rows,
        "normal_fit_seconds": normal_fit,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "sklearn": sklearn.__version__},
        "note": "Writer-held-out development OOF; no official Test and no heldout-writer training leakage",
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n4-WRITER OOF SUMMARY", flush=True)
    print(f"Normal   Accuracy={nres['accuracy']:.6f}, Macro F1={nres['macro_f1']:.6f} "
          f"({nres['correct']}/2560)", flush=True)
    print(f"Combined Accuracy={cres['accuracy']:.6f}, Macro F1={cres['macro_f1']:.6f} "
          f"({cres['correct']}/2560)", flush=True)
    print(f"Change={100*(cres['accuracy'] - nres['accuracy']):+.4f} percentage points", flush=True)
    print(f"Paired={paired}", flush=True)
    print(f"Saved: {OUT}", flush=True)


if __name__ == "__main__":
    run()
