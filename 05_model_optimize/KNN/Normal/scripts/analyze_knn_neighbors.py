#!/usr/bin/env python3
"""EXP-003: Visualize Hand-made KNN misclassifications and their 5 nearest training neighbors.

Run from repository root:
  python 05_model_optimize/KNN/Normal/scripts/analyze_knn_neighbors.py

This script does NOT overwrite EXP-001/EXP-002, tune hyperparameters, or load official Test.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "03_experiments" / "model_optimization"))
from common import LABELS, fixed_split, pipeline
from knn import build_baseline

HANDMADE = ROOT / "02_data" / "Hand_made"
BASELINE = ROOT / "05_model_optimize" / "KNN" / "Normal" / "results" / "EXP002_handmade_baseline"
OUTPUT = ROOT / "05_model_optimize" / "KNN" / "Normal" / "results" / "EXP003_error_neighbors"

EXPECTED_SHA256 = {
    41: "0a829172a72787c770139c73de34cc8b751181829ff6fe08ec260247b9be5a2f",
    42: "f364fb453d05b4fe2a705c2b8a80683b404608c7c257d664e3ec7ac86932d4c5",
    43: "10db20997c727c1bc1f1299f8e89626fadd3ec9618468d6759c86b0d2ddc4a66",
    44: "0e266a36e7b29edfcca0514836b5c383d13939948b4e335d70f61bb40bdc084d",
}
TARGET_PAIRS = [("0", "D"), ("9", "4"), ("7", "1"), ("8", "B")]
CASES_PER_PAIR = 4


def load_handmade():
    images, labels = {}, {}
    for writer in (41, 42, 43, 44):
        path = HANDMADE / f"ml26_{writer}.npz"
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != EXPECTED_SHA256[writer]:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual_sha}")
        with np.load(path, allow_pickle=False) as obj:
            data = obj["img"]
            y = obj["label_id"]
            strings = obj["label"]
        if data.shape != (640, 28, 28) or data.dtype != np.uint8:
            raise ValueError(f"Unexpected images: {path} {data.shape} {data.dtype}")
        if len(y) != 640 or np.bincount(y.astype(int), minlength=16).tolist() != [40] * 16:
            raise ValueError(f"Unexpected label distribution: {path}")
        if not np.array_equal(y.astype(int), np.array([LABELS.index(str(s)) for s in strings])):
            raise ValueError(f"Label mismatch: {path}")
        images[writer], labels[writer] = data, y.astype(int)
    return images, labels


def select_cases(records, pair):
    selected = []
    pool = [r for r in records if (r["true"], r["predicted"]) == pair]
    for writer in (41, 42, 43, 44):
        options = [r for r in pool if r["writer"] == writer]
        if options:
            selected.append(options[0])
    for item in pool:
        if len(selected) == CASES_PER_PAIR:
            break
        if item not in selected:
            selected.append(item)
    if len(selected) < CASES_PER_PAIR:
        raise ValueError(f"Not enough mistakes for pair {pair}: {len(pool)}")
    return selected[:CASES_PER_PAIR]


def center_of_mass(image):
    ink = image.astype(np.float64)
    total = ink.sum()
    if not total:
        return float("nan"), float("nan")
    yy, xx = np.indices(ink.shape)
    return float((xx * ink).sum() / total), float((yy * ink).sum() / total)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty file: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    predictions_path = BASELINE / "predictions.csv"
    metrics_path = BASELINE / "metrics.json"
    with predictions_path.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    exp002 = json.loads(metrics_path.read_text(encoding="utf-8"))
    if len(records) != 2560 or not np.isclose(exp002["accuracy"], 0.58828125):
        raise ValueError("EXP-002 baseline metadata mismatch")
    for row in records:
        row["writer"] = int(row["writer"])
        row["index"] = int(row["index"])
        row["correct"] = int(row["correct"])
    images, labels = load_handmade()
    for row in records:
        true = LABELS[int(labels[row["writer"]][row["index"]])]
        if row["true"] != true or row["correct"] != int(row["true"] == row["predicted"]):
            raise ValueError(f"EXP-002 predictions and handwriting disagree: {row}")

    X_train, y_train, _, _, sha = fixed_split()
    if sha != exp002["official_dataset_sha256"] or len(y_train) != 76800:
        raise ValueError("Official train provenance differs from EXP-002")
    estimator, scaled = build_baseline()
    if not scaled:
        raise ValueError("Expected scaler-enabled KNN baseline")
    model = pipeline(estimator, scaled)
    print("Rebuilding unchanged KNN baseline (76,800 official training samples)...", flush=True)
    model.fit(X_train, y_train)
    scaler = model.named_steps["scale_0_1"]
    knn = model.named_steps["model"]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases_rows, neighbors_rows = [], []
    for target_true, target_pred in TARGET_PAIRS:
        selected = select_cases(records, (target_true, target_pred))
        fig, axes = plt.subplots(CASES_PER_PAIR, 6, figsize=(13.5, 9))
        for case_num, r in enumerate(selected, start=1):
            writer, idx = r["writer"], r["index"]
            img = images[writer][idx]
            x_query = img.reshape(1, -1).astype(np.float32)
            predicted = LABELS[int(model.predict(x_query)[0])]
            if predicted != r["predicted"]:
                raise ValueError(f"EXP-002 prediction not reproduced: writer={writer}, index={idx}, {predicted} != {r['predicted']}")
            distances, indices = knn.kneighbors(scaler.transform(x_query), n_neighbors=5)
            distances, indices = distances[0], indices[0]
            labels_neighbors = [LABELS[int(y_train[i])] for i in indices]
            com_x, com_y = center_of_mass(img)
            case_id = f"{target_true}_to_{target_pred}_W{writer}_{idx:03d}"
            cases_rows.append({
                "case_id": case_id, "writer": writer, "index": idx,
                "true": target_true, "predicted": target_pred,
                "com_x": round(com_x, 3), "com_y": round(com_y, 3),
                "neighbor_labels": "/".join(labels_neighbors),
                "neighbor_votes": "/".join(f"{k}:{v}" for k, v in sorted(Counter(labels_neighbors).items())),
            })
            a = axes[case_num - 1, 0]
            a.imshow(img, cmap="gray", vmin=0, vmax=255)
            a.set_title(f"Hand-made W{writer} #{idx}\nTrue {target_true} / Pred {target_pred}", fontsize=9)
            a.axis("off")
            for rank, (dist, train_idx) in enumerate(zip(distances, indices), start=1):
                label = LABELS[int(y_train[train_idx])]
                a = axes[case_num - 1, rank]
                a.imshow(X_train[train_idx].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
                a.set_title(f"Neighbor {rank}: {label}\nDistance {dist:.3f}", fontsize=9)
                a.axis("off")
                neighbors_rows.append({
                    "case_id": case_id, "rank": rank, "neighbor_train_index": int(train_idx),
                    "neighbor_label": label, "scaled_euclidean_distance": round(float(dist), 6),
                })
        fig.suptitle(f"EXP-003 KNN neighbors | true {target_true} -> predicted {target_pred}", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .96))
        filename = f"neighbors_{target_true}_to_{target_pred}.png"
        fig.savefig(OUTPUT / filename, dpi=170)
        plt.close(fig)
        print(f"Saved {filename}: {len(selected)} hand-made mistakes x 5 nearest neighbors", flush=True)

    write_csv(OUTPUT / "selected_cases.csv", cases_rows)
    write_csv(OUTPUT / "nearest_neighbors.csv", neighbors_rows)
    summary = {
        "experiment": "EXP-003 KNN Normal misclassification + nearest-neighbor inspection",
        "input_baseline": "EXP002_handmade_baseline",
        "baseline_accuracy": exp002["accuracy"],
        "baseline_macro_f1": exp002["macro_f1"],
        "training_n": int(len(y_train)),
        "reviewed_pairs": [f"{a}->{b}" for a, b in TARGET_PAIRS],
        "cases_per_pair": CASES_PER_PAIR,
        "n_cases": len(cases_rows),
        "n_neighbors_per_case": 5,
        "note": "Selected cases illustrate errors. They are not a representative random sample. No causal claim or model improvement is established.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done: {len(cases_rows)} cases; {len(neighbors_rows)} nearest-neighbor comparisons")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
