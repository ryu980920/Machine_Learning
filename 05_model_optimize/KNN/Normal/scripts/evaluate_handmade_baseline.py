#!/usr/bin/env python3
"""KNN Normal baseline: evaluate the four standardized development handwriting sets.

Run from repository root:
python 05_model_optimize/KNN/Normal/scripts/evaluate_handmade_baseline.py

The official Original Test is never loaded. Hand-made is evaluation-only.
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
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "03_experiments" / "model_optimization"))
from common import fixed_split, pipeline, scores, LABELS
from knn import build_baseline

HANDMADE = ROOT / "02_data" / "Hand_made"
OUTPUT = ROOT / "05_model_optimize" / "KNN" / "Normal" / "results" / "EXP002_handmade_baseline"

EXPECTED_SHA256 = {
    41: "0a829172a72787c770139c73de34cc8b751181829ff6fe08ec260247b9be5a2f",
    42: "f364fb453d05b4fe2a705c2b8a80683b404608c7c257d664e3ec7ac86932d4c5",
    43: "10db20997c727c1bc1f1299f8e89626fadd3ec9618468d6759c86b0d2ddc4a66",
    44: "0e266a36e7b29edfcca0514836b5c383d13939948b4e335d70f61bb40bdc084d",
}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    x_train, y_train, _, _, dataset_sha = fixed_split()
    estimator, use_scaling = build_baseline()
    model = pipeline(estimator, use_scaling)

    print("KNN Normal baseline training:", len(y_train), flush=True)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    y_all, pred_all = [], []
    writer_rows, prediction_rows = [], []
    predict_seconds = 0.0

    for writer_id in (41, 42, 43, 44):
        filepath = HANDMADE / f"ml26_{writer_id}.npz"
        digest = hashlib.sha256(filepath.read_bytes()).hexdigest()
        if digest != EXPECTED_SHA256[writer_id]:
            raise ValueError(f"Hand-made SHA256 mismatch for {writer_id}: {digest}")

        with np.load(filepath, allow_pickle=False) as data:
            if set(data.files) != {"img", "label", "label_id"}:
                raise ValueError(f"Unexpected keys: {filepath}: {data.files}")
            images = data["img"]
            label_ids = data["label_id"].astype(np.int64)
            labels = data["label"]

        if images.shape != (640, 28, 28) or images.dtype != np.uint8:
            raise ValueError(f"Unexpected image shape/dtype: {filepath}")
        if label_ids.shape != (640,):
            raise ValueError(f"Unexpected label shape: {filepath}")
        if not np.array_equal(
            label_ids, np.asarray([LABELS.index(str(c)) for c in labels], dtype=np.int64)
        ):
            raise ValueError(f"Incorrect label mapping: {filepath}")
        counts = np.bincount(label_ids, minlength=16)
        if not np.all(counts == 40):
            raise ValueError(f"Unexpected per-class counts: {filepath}")

        x = images.reshape(640, 784).astype(np.float32)
        start = time.perf_counter()
        predictions = model.predict(x).astype(np.int64)
        predict_seconds += time.perf_counter() - start

        correct = int(np.sum(predictions == label_ids))
        acc = accuracy_score(label_ids, predictions)
        macro = f1_score(label_ids, predictions, labels=np.arange(16),
                         average="macro", zero_division=0)
        writer_rows.append({
            "writer": writer_id, "n": 640, "correct": correct,
            "accuracy": float(acc), "macro_f1": float(macro)
        })
        print(f"Writer {writer_id}: {correct}/640, accuracy={acc:.6f}, macro_f1={macro:.6f}", flush=True)

        for i, (true, pred) in enumerate(zip(label_ids, predictions)):
            prediction_rows.append({
                "writer": writer_id, "index": i,
                "true": LABELS[int(true)], "predicted": LABELS[int(pred)],
                "correct": int(true == pred)
            })
        y_all.append(label_ids)
        pred_all.append(predictions)

    y_true = np.concatenate(y_all)
    y_pred = np.concatenate(pred_all)
    result = scores(y_true, y_pred)
    result.update({
        "model": "KNN Normal Baseline", "handmade_n": len(y_true),
        "official_train_n": len(y_train), "official_dataset_sha256": dataset_sha,
        "handmade_sha256": {str(k): v for k, v in EXPECTED_SHA256.items()},
        "fit_seconds": fit_seconds, "handmade_predict_seconds": predict_seconds,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__, "numpy": np.__version__,
        "hyperparameters": {str(k): repr(v) for k, v in estimator.get_params().items()},
        "scaling": "pixel/255 in pipeline",
        "evaluation_role": "repeated-use development Hand-made; not independent final test",
        "writers": writer_rows
    })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(OUTPUT / "by_writer.csv", writer_rows)
    write_csv(OUTPUT / "predictions.csv", prediction_rows)
    cm = np.array(result["confusion_matrix"])
    write_csv(OUTPUT / "confusion_matrix.csv", [
        {"true/pred": LABELS[i], **{LABELS[j]: int(cm[i, j]) for j in range(16)}}
        for i in range(16)
    ])
    errors = [
        {"true": LABELS[i], "predicted": LABELS[j], "count": int(cm[i, j])}
        for i in range(16) for j in range(16)
        if i != j and cm[i, j]
    ]
    errors.sort(key=lambda row: (-row["count"], row["true"], row["predicted"]))
    write_csv(OUTPUT / "top_confusions.csv", errors)
    print(f"\nTotal: accuracy={result['accuracy']:.6f}, macro_f1={result['macro_f1']:.6f}")
    print("Top 10 confusions:")
    for row in errors[:10]:
        print(f"  {row['true']} -> {row['predicted']}: {row['count']}")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
