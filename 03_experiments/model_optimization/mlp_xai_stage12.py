#!/usr/bin/env python3
"""MLP Normal baseline XAI, stages 1-2 ONLY. Never touches official X_test/y_test.

Usage from repository root:
  python 03_experiments/model_optimization/mlp_xai_stage12.py --out xai_stage12_out

Stage 1: repeat published split, refit unchanged MLP baseline; select real
correct/misclassified images from official Validation and current Handmade.
Stage 2: calculate Integrated Gradients of the *logit*, or the logit margin
(predicted minus true) for misclassifications, with black-image baseline.
No image transformations, augmentation, hyperparameter tuning, or Test access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from common import fixed_split, pipeline, LABELS, DEFAULT_DATA
from mlp import build_baseline

WRITERS = (41, 42, 43, 44)
HAND_SHA = {
    41: "0a829172a72787c770139c73de34cc8b751181829ff6fe08ec260247b9be5a2f",
    42: "f364fb453d05b4fe2a705c2b8a80683b404608c7c257d664e3ec7ac86932d4c5",
    43: "10db20997c72787c770139c73de34cc8b751181829ff6fe08ec260247b9be5a2f",
    44: "0e266a36e7b29edfcca0514836b5c383d13939948b4e335d70f61bb40bdc084d",
}
# Correct writer 43 checksum (kept separately to make version explicit).
HAND_SHA[43] = "10db20997c727c1bc1f1299f8e89626fadd3ec9618468d6759c86b0d2ddc4a66"
TARGET_PAIRS = [("9", "4"), ("E", "C"), ("3", "5"), ("7", "2"),
                ("5", "F"), ("4", "F"), ("B", "8"), ("8", "B"),
                ("0", "D"), ("4", "A"), ("A", "4")]
CORRECT_CLASSES = "5F94B8"
PUBLISHED_ACC = 0.9731770833333333
PUBLISHED_HANDMADE_ACC = 0.628515625


def checksum(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_handmade() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    imgs, labels, writers, positions = [], [], [], []
    for w in WRITERS:
        p = ROOT / "02_data" / "Hand_made" / f"ml26_{w}.npz"
        actual_sha = checksum(p)
        if actual_sha != HAND_SHA[w]:
            raise RuntimeError(f"Handmade writer {w} checksum mismatch {actual_sha}")
        with np.load(p, allow_pickle=False) as data:
            im, y, txt = data["img"], data["label_id"], data["label"]
        if im.shape != (640, 28, 28) or im.dtype != np.uint8:
            raise RuntimeError(f"Bad Handmade image shape/dtype for writer {w}")
        if Counter(y.tolist()) != Counter({i: 40 for i in range(16)}):
            raise RuntimeError(f"Bad class balance for writer {w}")
        if any(LABELS[int(i)] != str(c) for i, c in zip(y, txt)):
            raise RuntimeError(f"Label mismatch in writer {w}")
        if w == 42 and np.unique(im).size != 256:
            raise RuntimeError("Writer 42 isn't the updated 256-level grayscale NPZ")
        imgs.append(im.reshape(640, 784).astype(np.float32))
        labels.append(y.astype(np.int64))
        writers.append(np.full(640, w, dtype=np.int32))
        positions.append(np.arange(640, dtype=np.int32))
    return tuple(np.concatenate(v) for v in (imgs, labels, writers, positions))


def official_indices(path: Path, validation_y: np.ndarray) -> np.ndarray:
    # Reconstruct exactly the first split solely from TRAIN metadata.
    with np.load(path, allow_pickle=False) as d:
        y = d["y_train"].astype(np.int64)
        src = d["train_source_index"].astype(np.int64)
        split = d["train_source_split"].astype(np.int64)
    groups = y * 1_000_000_000 + split * 100_000_000 + src
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr, va = next(cv.split(np.zeros((len(y), 1), np.uint8), y, groups))
    if len(tr) != 76800 or len(va) != 19200 or not np.array_equal(y[va], validation_y):
        raise RuntimeError("Original grouped split cannot be reproduced")
    return va


def prior_official_cm() -> np.ndarray:
    p = (ROOT / "03_experiments" / "W02_2026-11-09_11-15" / "results"
         / "EXP-유-003_confusion_long.csv")
    cm = np.zeros((16, 16), dtype=np.int64)
    with p.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["model"] == "MLPClassifier":
                cm[LABELS.index(row["actual_label"]), LABELS.index(row["predicted_label"])] = int(row["count"])
    if cm.sum() != 19200:
        raise RuntimeError("Historical official MLP confusion matrix missing")
    return cm


def prior_handmade() -> dict[tuple[int, int], tuple[str, str]]:
    p = (ROOT / "03_experiments" / "W02_2026-11-09_11-15" / "results"
         / "EXP-유-003_rotation_v4" / "rotation_after_predictions.csv")
    previous = {}
    with p.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["model"] == "MLP":
                previous[int(r["writer"]), int(r["image_idx_in_writer"])] = (r["actual"], r["predicted"])
    if len(previous) != 2560:
        raise RuntimeError("Historical Handmade MLP predictions missing")
    return previous


def get_logits(net, x):
    x = np.asarray(x, dtype=np.float64)
    a = np.maximum(x @ net.coefs_[0] + net.intercepts_[0], 0.0)
    return a @ net.coefs_[1] + net.intercepts_[1]


def integrated_gradients(net, x01: np.ndarray, pred: int, true: int, steps: int = 128):
    """Black baseline and midpoint Riemann sum for 1-hidden-layer ReLU MLP.

    For errors, the objective is logit(pred) - logit(true).
    For correct predictions, the objective is logit(true).
    The input variable is the exact model input x/255, not raw grayscale.
    """
    if len(net.coefs_) != 2 or net.activation != "relu":
        raise RuntimeError("Analytic IG only supports the unchanged 1-layer ReLU MLP")
    x = np.asarray(x01, dtype=np.float64).reshape(784)
    w0, w1 = (np.asarray(a, dtype=np.float64) for a in net.coefs_)
    b0 = np.asarray(net.intercepts_[0], dtype=np.float64)
    out_weight = w1[:, pred].copy()
    if pred != true:
        out_weight -= w1[:, true]
    alpha = (np.arange(steps, dtype=np.float64) + 0.5) / steps
    z = (alpha[:, None] * x[None, :]) @ w0 + b0
    active = (z > 0.0).astype(np.float64)
    grad = (active * out_weight[None, :]) @ w0.T
    attribution = x * grad.mean(axis=0)
    logits_x, logits_0 = get_logits(net, x), get_logits(net, np.zeros_like(x))
    if pred == true:
        delta = logits_x[pred] - logits_0[pred]
        objective = "true_class_logit"
    else:
        delta = ((logits_x[pred] - logits_x[true])
                 - (logits_0[pred] - logits_0[true]))
        objective = "pred_minus_true_logit_margin"
    return attribution.reshape(28, 28), objective, float(delta), float(attribution.sum() - delta)


def diverse_selection(indices: np.ndarray, writers: np.ndarray, count: int, separate: bool):
    if not separate:
        return list(indices[:count])
    chosen = []
    # Favor different writers for each confusion pair.
    buckets = {w: indices[writers[indices] == w].tolist() for w in WRITERS}
    order = sorted(WRITERS, key=lambda w: (-len(buckets[w]), w))
    while len(chosen) < count and any(buckets.values()):
        for w in order:
            if len(chosen) >= count:
                break
            if buckets[w]:
                chosen.append(buckets[w].pop(0))
    return chosen


def select_images(domain: str, x: np.ndarray, y: np.ndarray, pred: np.ndarray,
                  writer: np.ndarray, index: np.ndarray, prob: np.ndarray, net,
                  max_pairs: int, max_correct: int):
    picked, used = [], set()
    for a, b in TARGET_PAIRS:
        matches = np.flatnonzero((y == LABELS.index(a)) & (pred == LABELS.index(b)))
        for k in diverse_selection(matches, writer, max_pairs, domain == "Handmade"):
            if int(k) not in used:
                picked.append((int(k), f"{a}_to_{b}"))
                used.add(int(k))
    for c in CORRECT_CLASSES:
        matches = np.flatnonzero((y == LABELS.index(c)) & (pred == LABELS.index(c)))
        for k in diverse_selection(matches, writer, max_correct, domain == "Handmade"):
            if int(k) not in used:
                picked.append((int(k), f"correct_{c}"))
                used.add(int(k))
    rows, views, raw, attribution = [], [], [], []
    for k, kind in picked:
        actual, prediction = int(y[k]), int(pred[k])
        x01 = x[k].astype(np.float64) / 255.0
        attr, objective, delta, residual = integrated_gradients(net, x01, prediction, actual)
        case = {
            "domain": domain, "case": kind, "writer": int(writer[k]),
            "index": int(index[k]), "true": LABELS[actual], "predicted": LABELS[prediction],
            "correct": int(actual == prediction),
            "p_true": float(prob[k, actual]), "p_predicted": float(prob[k, prediction]),
            "objective": objective, "IG_sum": float(attr.sum()),
            "logit_objective_delta": delta, "IG_completeness_residual": residual,
            "IG_abs_sum": float(np.abs(attr).sum()),
        }
        rows.append(case)
        views.append({"row": case, "img": x[k].reshape(28,28).copy(), "ig": attr.copy()})
        raw.append(x[k].reshape(28, 28).astype(np.uint8))
        attribution.append(attr.astype(np.float32))
    return rows, views, raw, attribution


def save_plots(views: list[dict], out: Path, batch: int = 6):
    plot_dir = out / "review_grids"
    plot_dir.mkdir(exist_ok=True)
    for start in range(0, len(views), batch):
        page = views[start:start+batch]
        fig, axes = plt.subplots(len(page), 3, figsize=(10.7, 2.7 * len(page)), squeeze=False)
        for j, case in enumerate(page):
            r, img, att = case["row"], case["img"], case["ig"]
            maxabs = max(float(np.max(np.abs(att))), 1e-8)
            axes[j,0].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[j,1].imshow(att, cmap="RdBu_r", vmin=-maxabs, vmax=maxabs)
            axes[j,2].imshow(np.abs(att), cmap="magma", vmin=0, vmax=maxabs)
            writer_desc = f"writer {r['writer']} #" if r["writer"] >= 0 else "train index #"
            axes[j,0].set_title(f"{r['domain']}: {r['true']} -> {r['predicted']}  ({writer_desc}{r['index']})", fontsize=9)
            axes[j,1].set_title("Signed IG (red = supports objective)", fontsize=9)
            axes[j,2].set_title("Absolute IG", fontsize=9)
            for ax in axes[j]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(plot_dir / f"cases_{start//batch+1:02d}.png", dpi=135)
        plt.close(fig)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("mlp_xai_stage12_results"))
    parser.add_argument("--max-per-pair", type=int, default=4)
    parser.add_argument("--max-correct-per-class", type=int, default=2)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("Loading official TRAIN split (never reading X_test/y_test)", flush=True)
    xtr, ytr, xva, yva, official_sha = fixed_split(DEFAULT_DATA)
    vaidx = official_indices(DEFAULT_DATA, yva)
    print("Loading current Handmade image arrays", flush=True)
    xhm, yhm, whm, ihm = read_handmade()
    estimator, is_scaled = build_baseline()
    if not is_scaled or estimator.get_params()["random_state"] != 42:
        raise RuntimeError("Unexpected changed baseline definition")
    model = pipeline(estimator, is_scaled)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        print("Fitting unchanged original MLP Baseline; no Handmade in fit()", flush=True)
        model.fit(xtr, ytr)
    pv = model.predict(xva).astype(np.int64)
    ph = model.predict(xhm).astype(np.int64)
    historical = prior_handmade()
    hand_disagreements = 0
    for k, (w, idx) in enumerate(zip(whm, ihm)):
        a, b = historical[int(w), int(idx)]
        if a != LABELS[int(yhm[k])]:
            raise RuntimeError(f"Handmade label mismatch at {w}, {idx}")
        hand_disagreements += b != LABELS[int(ph[k])]
    cm = confusion_matrix(yva, pv, labels=np.arange(16))
    cm_historical_l1 = int(np.abs(cm - prior_official_cm()).sum())
    av, ah = float(accuracy_score(yva, pv)), float(accuracy_score(yhm, ph))
    print(f"Official accuracy={av:.8f}, historical={PUBLISHED_ACC:.8f}, CM_L1={cm_historical_l1}", flush=True)
    print(f"Handmade accuracy={ah:.8f}, historical={PUBLISHED_HANDMADE_ACC:.8f}, predictions differing={hand_disagreements}", flush=True)
    net = model.named_steps["model"]
    if net.classes_.tolist() != list(range(16)):
        raise RuntimeError("Unexpected MLP class order")

    pvproba, phproba = model.predict_proba(xva), model.predict_proba(xhm)
    official_writer = np.full(len(yva), -1, dtype=np.int32)
    print("Selecting correct/errors and calculating integrated gradients", flush=True)
    o_rows, o_views, o_raw, o_attrs = select_images(
        "OfficialVal", xva, yva, pv, official_writer, vaidx, pvproba, net,
        args.max_per_pair, args.max_correct_per_class)
    h_rows, h_views, h_raw, h_attrs = select_images(
        "Handmade", xhm, yhm, ph, whm, ihm, phproba, net,
        args.max_per_pair, args.max_correct_per_class)
    cases = o_rows + h_rows
    write_csv(out / "selected_cases.csv", cases)
    write_csv(out / "handmade_all_predictions.csv", [
        {"writer": int(w), "index": int(idx), "true": LABELS[int(a)],
         "predicted": LABELS[int(b)], "correct": int(a==b)}
        for w,idx,a,b in zip(whm, ihm, yhm, ph)])
    np.savez_compressed(out / "selected_pixels_and_IG.npz",
        pixels=np.stack(o_raw+h_raw), IG=np.stack(o_attrs+h_attrs))
    save_plots(o_views+h_views, out)

    metadata = {
        "scope": "Stage1 extract + Stage2 Integrated Gradients ONLY; not Stage3",
        "test_data_used": False,
        "handmade_used_for_training": False,
        "data_sha256": official_sha, "handmade_sha256": HAND_SHA,
        "official_split": "StratifiedGroupKFold(5, shuffle=True, random_state=42), first fold",
        "training_n": int(len(ytr)), "validation_n": int(len(yva)),
        "handmade_n": int(len(yhm)), "official_accuracy": av,
        "historical_official_accuracy": PUBLISHED_ACC,
        "official_confusion_matrix_cell_L1_difference_from_historical": cm_historical_l1,
        "official_accuracy_exact_match": bool(av == PUBLISHED_ACC),
        "handmade_accuracy": ah, "historical_handmade_accuracy": PUBLISHED_HANDMADE_ACC,
        "handmade_prediction_mismatch_count": int(hand_disagreements),
        "historical_predictions_exact_match": bool(cm_historical_l1 == 0 and hand_disagreements == 0),
        "official_macro_f1": float(f1_score(yva, pv, average="macro")),
        "handmade_macro_f1": float(f1_score(yhm, ph, average="macro")),
        "selected_n": len(cases), "selected_official_n": len(o_rows),
        "selected_handmade_n": len(h_rows),
        "XAI_method": "128-step midpoint Integrated Gradients; black baseline; class logit or predicted-vs-true logit margin",
        "IG_completeness_residual_max_abs": float(max(abs(r["IG_completeness_residual"]) for r in cases)),
        "model_params": {k: repr(v) for k,v in net.get_params().items()},
        "warnings": [f"{w.category.__name__}: {w.message}" for w in ws],
        "versions": {"python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn.__version__,
                     "matplotlib":matplotlib.__version__},
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    description = [
        "# Normal MLP baseline: Stage 1+2 XAI report",
        "",
        "## Reproduction check",
        f"- Official: {len(ytr):,} train / {len(yva):,} Validation; accuracy {av:.8%} (published {PUBLISHED_ACC:.8%}).",
        f"- Official CM cell L1 difference from historical: **{cm_historical_l1}**.",
        f"- Handmade: {len(yhm):,} development-only images; accuracy {ah:.8%} (published {PUBLISHED_HANDMADE_ACC:.8%}).",
        f"- Handmade prediction disagreements versus published CSV: **{hand_disagreements} / {len(yhm)}**.",
        f"- Selected images with actual pixels and IG: {len(cases)}, Official {len(o_rows)}, Handmade {len(h_rows)}.",
        f"- Max Integrated Gradients completeness residual: {metadata['IG_completeness_residual_max_abs']:.4g}.",
        "",
        "## Read the outputs",
        "- `selected_cases.csv`: exact image provenance and objective/probabilities.",
        "- `review_grids/*.png`: original 28x28 images + signed and absolute pixel attributions.",
        "- `selected_pixels_and_IG.npz`: matching images and attribution maps, same order as selected_cases.csv.",
        "- `handmade_all_predictions.csv`: fresh predictions for 2,560 images.",
        "- `run_metadata.json`: checksums, accuracy, reproduction deviations, software versions.",
        "",
        "## Interpretation limits",
        "Attribution shows contributions to logits (or predicted-versus-true margin), **not proof** that",
        "the model relies on shape versus brightness/position. Causal confirmation via controlled",
        "image transforms is deliberately deferred to Stage 3 after the user's approval.",
        "Official X_test/y_test were never read. No hand-written images were used in fit().",
        "If historical predictions differ, interpret all attributions as **the retrained baseline**,",
        "not as the historically saved model's exact weights.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(description)+"\n", encoding="utf-8")
    print(f"DONE: {out} ({len(cases)} selected), only stages 1 and 2", flush=True)


if __name__ == "__main__":
    main()
