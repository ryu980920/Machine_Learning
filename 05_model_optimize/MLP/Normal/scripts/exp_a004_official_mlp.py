#!/usr/bin/env python3
"""Hypothesis A EXP004: MLP ONLY, official-training-fitted morphology normalizers.

All thresholds/quantiles are fitted on the fixed official TRAIN subset.
Handmade is evaluated, never used to fit preprocessing or MLP weights.
Official Original Test (X_test/y_test) and professor Test are never opened.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from stroke_normalization import StrokeWidthNormalizer, grayscale, center_of_mass, topology
from width_normalization import GlyphWidthNormalizer

SHA = "0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879"
CLASSES = "0123456789ABCDEF"
CONDITIONS = ("C0_CoM", "C1_Stroke015", "C2_Stroke030",
              "C3_GlyphWidth015", "C4_Stroke015_GlyphWidth015")
PAIRS = (("B", "8"), ("8", "B"), ("D", "0"), ("0", "D"), ("5", "F"), ("F", "5"))

def checksum(path):
    with open(path, "rb") as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def save_csv(path, rows):
    if not rows:
        raise ValueError("Empty output " + str(path))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

class Preprocessing:
    """Training fitted, no labels, same transform for train/validation/handmade."""
    def __init__(self, policy):
        if policy not in CONDITIONS:
            raise ValueError(policy)
        self.policy = policy
        self.stroke = None
        self.width = None

    def fit(self, X):
        self.n_fit = len(X)
        if self.policy in ("C1_Stroke015", "C2_Stroke030", "C4_Stroke015_GlyphWidth015"):
            strength = .30 if self.policy == "C2_Stroke030" else .15
            self.stroke = StrokeWidthNormalizer(strength=strength, preserve_topology=True).fit(X)
        if self.policy in ("C3_GlyphWidth015", "C4_Stroke015_GlyphWidth015"):
            x = self.stroke.transform(X) if self.stroke else X
            self.width = GlyphWidthNormalizer(strength=.15, preserve_topology=True).fit(x)
        return self

    def transform(self, X):
        a = grayscale(X)
        if self.stroke:
            a = self.stroke.transform(a).reshape(-1, 28, 28)
        if self.width:
            a = self.width.transform(a).reshape(-1, 28, 28)
        result = np.empty((len(a), 784), np.float32)
        for i, img in enumerate(a):
            result[i] = center_of_mass(img).reshape(-1)
        return result

    def description(self):
        result = {"condition": self.policy, "training_fit_count": self.n_fit,
                  "fit_data": "OfficialTrainOnly", "uses_labels": False,
                  "uses_writer": False, "com_last": True, "topology_guard": True}
        if self.stroke:
            t = self.stroke
            result["stroke"] = {"strength": t.strength, "threshold": t.threshold,
                                "median": float(t.median_), "q20": float(t.q_low_),
                                "q80": float(t.q_high_), "n": t.n_fitted_}
        if self.width:
            t = self.width
            result["glyph_width"] = {"strength": t.strength, "threshold": t.threshold,
                                     "median": float(t.median_), "q20": float(t.q_low_),
                                     "q80": float(t.q_high_), "n": t.n_fitted_}
        return result

def load_inputs(root):
    official = root / "02_data/datasets/MNIST_EMNIST_official_split_112000.npz"
    if not official.exists():
        raise FileNotFoundError(official)
    s = checksum(official)
    if s != SHA:
        raise RuntimeError(f"Official dataset SHA differs: {s}")
    sys.path.insert(0, str(root / "03_experiments/model_optimization"))
    sys.path.insert(0, str(root / "05_model_optimize/MLP/Normal/scripts"))
    from common import fixed_split
    from Pixcel_Attribution import read_handmade
    train_x, train_y, val_x, val_y, s2 = fixed_split(official)
    hand_x, hand_y, writers, positions = read_handmade()
    assert s2 == SHA and train_x.shape == (76800, 784)
    assert val_x.shape == (19200, 784) and hand_x.shape == (2560, 784)
    assert set(writers.tolist()) == {41, 42, 43, 44}
    return train_x, train_y, val_x, val_y, hand_x, hand_y, writers, positions

def add_metrics(rows, condition, seed, dataset, y, pred, writers=None):
    groups = {"all": np.ones(len(y), dtype=bool),
              "digits": y < 10, "letters": y >= 10}
    for i, cl in enumerate(CLASSES):
        groups[f"class_{cl}"] = y == i
    if writers is not None:
        for w in (41, 42, 43, 44):
            groups[f"writer_{w}"] = writers == w
            groups[f"writer_{w}_digits"] = (writers == w) & (y < 10)
            groups[f"writer_{w}_letters"] = (writers == w) & (y >= 10)
    for name, mask in groups.items():
        if not mask.any():
            continue
        a, b = y[mask], pred[mask]
        rows.append({"condition": condition, "seed": seed, "dataset": dataset,
                     "group": name, "n": len(a), "correct": int((a == b).sum()),
                     "accuracy": float(accuracy_score(a, b)),
                     "macro_f1": float(f1_score(a, b, average="macro", zero_division=0))})

def make_real_images(out, raw, after, labels):
    """Only real Handmade input images; no synthetic evidence or decorative chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = []
    for cl in "B8D05F":
        hits = np.flatnonzero(labels == CLASSES.index(cl))
        selected.extend(hits[[0, min(80, len(hits) - 1)]].tolist())
    fig, axes = plt.subplots(len(selected), 2, figsize=(4.6, len(selected)*1.5))
    for i, j in enumerate(selected):
        axes[i, 0].imshow(raw[j].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        axes[i, 1].imshow(after[j].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(CLASSES[int(labels[j])], rotation=0, labelpad=12)
    axes[0, 0].set_title("Actual input")
    axes[0, 1].set_title("Actual transformed")
    fig.tight_layout()
    fig.savefig(out / "real_handmade_examples.png", dpi=140)
    plt.close(fig)

def run(root, out, condition, seeds, max_iter):
    out.mkdir(parents=True, exist_ok=True)
    train_x, train_y, val_x, val_y, hand_x, hand_y, writers, positions = load_inputs(root)
    assert not set(seeds).difference({42, 43, 44}), "Unexpected seed"
    print("SOURCE SHA MATCHED", SHA, "train", len(train_x), "val", len(val_x),
          "handmade", len(hand_x), flush=True)

    pipeline = Preprocessing(condition).fit(train_x)
    desc = pipeline.description()
    print("TRAIN ONLY FIT", json.dumps(desc), flush=True)
    pre_train = pipeline.transform(train_x)
    pre_val = pipeline.transform(val_x)
    pre_hand = pipeline.transform(hand_x)
    assert np.isfinite(pre_train).all() and np.isfinite(pre_val).all()
    assert np.isfinite(pre_hand).all()
    assert (0 <= pre_train.min() <= pre_train.max() <= 255)
    assert pre_train.shape == train_x.shape
    assert pre_val.shape == val_x.shape and pre_hand.shape == hand_x.shape

    # Experimental original-image evidence: exact samples, not recreated graphs.
    make_real_images(out, hand_x, pre_hand, hand_y)
    count_topology = sum(topology(x.reshape(28,28)) != topology(z.reshape(28,28))
                         for x, z in zip(hand_x, pre_hand))
    mass_before = hand_x.sum(axis=1, dtype=np.float64)
    mass_after = pre_hand.sum(axis=1, dtype=np.float64)
    desc["handmade_topology_changes_after_full_transform"] = int(count_topology)
    desc["handmade_ink_loss_any"] = int(np.count_nonzero(mass_after < mass_before - .01))
    desc["reference_eval_note"] = "Handmade is development diagnostic only; neither preprocessing nor model fitted on it."

    pre_train /= np.float32(255)
    pre_val /= np.float32(255)
    pre_hand /= np.float32(255)
    all_metrics, all_pairs, predictions, fits = [], [], [], []
    for seed in seeds:
        print("START MLP", condition, seed, "max_iter", max_iter, flush=True)
        net = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam",
                            alpha=0.0001, learning_rate_init=0.001,
                            max_iter=max_iter, tol=0.0,
                            n_iter_no_change=max_iter + 1,
                            batch_size="auto", shuffle=True, random_state=seed)
        t0 = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            net.fit(pre_train, train_y)
        v = net.predict(pre_val).astype(np.int64)
        h = net.predict(pre_hand).astype(np.int64)
        add_metrics(all_metrics, condition, seed, "OfficialVal", val_y, v)
        add_metrics(all_metrics, condition, seed, "Handmade", hand_y, h, writers)
        for a, b in PAIRS:
            all_pairs.append({"condition": condition, "seed": seed, "true": a,
                              "pred": b, "count": int(np.sum((hand_y == CLASSES.index(a)) &
                                                           (h == CLASSES.index(b))))})
        for i in range(len(h)):
            predictions.append({"condition": condition, "seed": seed,
                                "writer": int(writers[i]), "index_in_writer": int(positions[i]),
                                "actual": CLASSES[int(hand_y[i])], "predicted": CLASSES[int(h[i])]})
        model_path = out / f"{condition}_seed{seed}.joblib"
        joblib.dump(net, model_path, compress=2)
        fits.append({"condition": condition, "seed": seed, "iteration": int(net.n_iter_),
                     "training_loss": float(net.loss_), "fit_seconds": round(time.monotonic()-t0, 1),
                     "checkpoint_sha256": checksum(model_path)})
        save_csv(out / "group_metrics.csv", all_metrics)
        save_csv(out / "pair_errors.csv", all_pairs)
        save_csv(out / "handmade_predictions.csv", predictions)
        save_csv(out / "fit_metrics.csv", fits)
        def acc(data, group):
            return next(r["accuracy"] for r in all_metrics if
                        r["seed"] == seed and r["dataset"] == data and r["group"] == group)
        letter = acc("Handmade", "letters")
        digit = acc("Handmade", "digits")
        official = acc("OfficialVal", "all")
        passed = letter >= 0.8 and digit >= 0.85 and official >= 0.973
        print(f"RESULT {condition} seed {seed}: letter={letter:.5f}, digit={digit:.5f}, official={official:.5f}, PASS={passed}",flush=True)
        (out / "run_manifest.json").write_text(
            json.dumps({"source_sha256": SHA, "condition": condition,
                        "preprocessing": desc, "model": "MLPClassifier(100 ReLU hidden nodes)",
                        "max_iter": max_iter, "seeds_planned": list(seeds),
                        "seed_results": fits, "official_original_test_used": False,
                        "professor_test_used": False}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        del net, v, h
        gc.collect()
    print("COMPLETED", condition, "models", len(fits), flush=True)

def self_test():
    import cv2
    x = np.zeros((40, 28, 28), np.float32)
    for i in range(len(x)):
        cv2.line(x[i], (5 + i % 5, 5), (5 + i % 5, 23), 255, 1 + i % 5)
    for cond in CONDITIONS:
        p = Preprocessing(cond).fit(x)
        z = p.transform(x)
        assert z.shape == (40, 784) and np.isfinite(z).all()
        assert 0 <= z.min() <= z.max() <= 255
        assert np.array_equal(z, p.transform(x))
    print("SELF TEST PASS (synthetic only; does not claim real accuracy)", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--self-test", action="store_true")
    opt = parser.parse_args()
    if opt.self_test:
        self_test()
    else:
        if opt.repo_root is None or opt.out_dir is None or opt.condition is None:
            parser.error("repo-root, out-dir, condition required")
        run(opt.repo_root, opt.out_dir, opt.condition, opt.seeds, opt.max_iter)
