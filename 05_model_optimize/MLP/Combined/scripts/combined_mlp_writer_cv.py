#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-CMB-001: 16-class Normal vs Combined MLP, writer-wise 4-fold OOF evaluation.

Reads official v3 *training* arrays only (never X_test/y_test); reads current
four standardized Grayscale Hand-made NPZs (already aligned). No GitHub writes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from joblib import dump
from pixel_transform import PixelScaler255

LABELS = "0123456789ABCDEF"
WRITERS = (41, 42, 43, 44)
V3_SHA = "0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879"
HANDMADE_SHA = {
    41: "0a829172a72787c770139c73de34cc8b751181829ff6fe08ec260247b9be5a2f",
    42: "f364fb453d05b4fe2a705c2b8a80683b404608c7c257d664e3ec7ac86932d4c5",
    43: "10db20997c727c1bc1f1299f8e89626fadd3ec9618468d6759c86b0d2ddc4a66",
    44: "0e266a36e7b29edfcca0514836b5c383d13939948b4e335d70f61bb40bdc084d",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_sha(path: Path, expected: str) -> str:
    found = sha256(path)
    if found != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: got {found}; expected {expected}")
    return found


def load_official(path: Path):
    digest = require_sha(path, V3_SHA)
    with np.load(path, allow_pickle=False) as z:
        # The official 16,000 Test images/labels MUST NOT be accessed here.
        X = z["X_train"]
        y = z["y_train"].astype(np.int64)
        src = z["train_source_index"].astype(np.int64)
        split = z["train_source_split"].astype(np.int64)
        aug = z["train_is_augmented"]
    if X.shape != (96000, 28, 28) or y.shape != (96000,) or int(aug.sum()) != 5349:
        raise ValueError("Official v3 array shape or augmentation count differs from GitHub experiment 2")
    if not np.array_equal(np.unique(y), np.arange(16)):
        raise ValueError("Official training labels are not 0..15")
    groups = y * 1_000_000_000 + split * 100_000_000 + src
    tr, va = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups))
    if (len(tr), len(va)) != (76800, 19200):
        raise ValueError(f"Unexpected train/val split: {len(tr)}, {len(va)}")
    if np.intersect1d(groups[tr], groups[va]).size:
        raise ValueError("Source-group leakage across official train/validation")
    if not np.array_equal(np.sort(np.r_[tr, va]), np.arange(96000)):
        raise ValueError("Official train/val do not partition all 96,000 samples")
    return {
        "train_idx": tr.astype(np.int32),
        "val_idx": va.astype(np.int32),
        "train_group_ids": groups[tr].astype(np.int64),
        "val_group_ids": groups[va].astype(np.int64),
        "train_class_counts": np.bincount(y[tr], minlength=16).tolist(),
        "val_class_counts": np.bincount(y[va], minlength=16).tolist(),
        "X_train": X[tr].reshape(-1, 784).astype(np.float32),
        "y_train": y[tr],
        "X_val": X[va].reshape(-1, 784).astype(np.float32),
        "y_val": y[va],
        "sha256": digest,
    }


def load_writers(directory: Path):
    result = {}
    for writer in WRITERS:
        path = directory / f"ml26_{writer}.npz"
        digest = require_sha(path, HANDMADE_SHA[writer])
        with np.load(path, allow_pickle=False) as z:
            if not {"img", "label", "label_id"}.issubset(z.files):
                raise ValueError(f"Required handmade NPZ keys missing in {path}")
            img = z["img"]
            label = z["label"].astype(str)
            y = z["label_id"].astype(np.int64)
        if img.shape != (640, 28, 28) or img.dtype != np.uint8 or y.shape != (640,):
            raise ValueError(f"Unexpected shape/dtype for writer {writer}: {img.shape}, {img.dtype}")
        if not np.array_equal(np.bincount(y, minlength=16), np.full(16, 40)):
            raise ValueError(f"Writer {writer}: expected 40 examples of each class")
        if not np.array_equal(label, np.array(list(LABELS))[y]):
            raise ValueError(f"Writer {writer}: label/label_id mismatch")
        # These are the *current standardized NPZs*. No extra rotation,
        # inversion, binarization, or x255 should be applied.
        result[writer] = {"X": img.reshape(-1, 784).astype(np.float32),
                          "y": y, "sha256": digest}
    return result


def new_mlp():
    # Exact model family and settings in the GitHub W02 baseline:
    # MLPClassifier(random_state=42), PixelScaler255, no augmentation.
    return Pipeline([
        ("scale_0_1", PixelScaler255()),
        ("model", MLPClassifier(random_state=42)),
    ])


def metrics(y_true, y_pred):
    return {"n": len(y_true),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=np.arange(16), zero_division=0))}


def save_table(path, header, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_cm(path, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(16))
    save_table(path, ["Actual/Predicted", *LABELS], ([LABELS[i], *cm[i]] for i in range(16)))
    return cm


def save_per_class(path, y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(16), zero_division=0)
    save_table(path, ["class", "precision", "recall", "f1", "support"],
               ([LABELS[i], precision[i], recall[i], f1[i], support[i]] for i in range(16)))


def make_plots(out, y, normal, combined, losses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm_pairs = [
        ("normal", confusion_matrix(y, normal, labels=np.arange(16))),
        ("combined", confusion_matrix(y, combined, labels=np.arange(16))),
    ]
    for name, cm in cm_pairs:
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xticks=np.arange(16), yticks=np.arange(16),
               xticklabels=list(LABELS), yticklabels=list(LABELS),
               xlabel="Predicted", ylabel="Actual", title=f"{name.title()} — OOF Hand-made Confusion Matrix")
        for i in range(16):
            for j in range(16):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > cm.max() * 0.6 else "black")
        fig.tight_layout()
        fig.savefig(out / f"{name}_OOF_confusion_matrix.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, curve in losses.items():
        ax.plot(np.arange(1, len(curve)+1), curve, label=key)
    ax.set(xlabel="MLP iteration", ylabel="Training loss", title="Normal and Combined Fold Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "MLP_loss_curves.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official", type=Path, required=True, help="Current official v3 112000 NPZ")
    p.add_argument("--handmade-dir", type=Path, required=True, help="Directory containing current ml26_41.npz ... ml26_44.npz")
    p.add_argument("--output", type=Path, required=True, help="NEW directory for this run (must not already exist)")
    p.add_argument("--check-only", action="store_true", help="Verify source files and split, but do NOT train")
    args = p.parse_args()
    if args.output.exists() and not args.check_only:
        p.error("Output directory already exists: choose a NEW output path to preserve older experiments")

    print("Loading and SHA-verifying official v3 Train and four current handmade files", flush=True)
    official = load_official(args.official)
    writers = load_writers(args.handmade_dir)
    print("Verified: Official train 76,800 / validation 19,200; 4 writers x 640, 40 per class", flush=True)
    args.output.mkdir(parents=True, exist_ok=not args.check_only)
    out = args.output
    sha_info = {"official_v3": official["sha256"],
                "handmade": {str(w): writers[w]["sha256"] for w in WRITERS}}
    (out / "input_hashes.json").write_text(json.dumps(sha_info, indent=2), encoding="utf-8")
    np.savez_compressed(out / "official_split_indices.npz",
                        train_idx=official["train_idx"], val_idx=official["val_idx"],
                        train_group_ids=official["train_group_ids"],
                        val_group_ids=official["val_group_ids"])
    # These are row indices in the original official X_train[0:96000].
    # Fold train rows in Combined start with X_train[train_idx], followed by
    # the 640 original rows of each training writer in ascending writer order.
    fold_manifest = []
    for held in WRITERS:
        train_writers = [w for w in WRITERS if w != held]
        fold_manifest.append({
            "held_out_writer": held,
            "combined_official_training_row_indices": "official_split_indices.npz:train_idx",
            "combined_handmade_training": {str(w): {"original_row_indices": [0, 639], "n": 640}
                                           for w in train_writers},
            "handmade_validation": {"writer": held, "original_row_indices": [0, 639], "n": 640},
            "official_development_validation": "official_split_indices.npz:val_idx",
            "note": "Original Hand-made NPZ row order. 0-based inclusive indices. Validation writer excluded from Combined training.",
        })
    manifest = {
        "experiment_id": "EXP-CMB-001", "model": "MLPClassifier(random_state=42)",
        "mlp_hyperparameters": MLPClassifier(random_state=42).get_params(deep=True),
        "scaling": "PixelScaler255: X/255.0", "official_train_split": {
            "algorithm": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), first fold",
            "train_n": len(official["train_idx"]),
            "validation_n": len(official["val_idx"]),
            "train_class_counts_0_to_15": official["train_class_counts"],
            "validation_class_counts_0_to_15": official["val_class_counts"],
            "group_overlap": 0,
        },
        "handmade_cv": "leave-one-writer-out / 4 folds",
        "folds": fold_manifest,
        "data_sha256": sha_info,
        "official_test_used": False,
        "professor_test_used": False,
    }
    (out / "training_validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save_table(out / "writer_fold_assignment.csv",
               ["held_out_writer", "train_writer_1", "train_writer_2", "train_writer_3",
                "official_train_n", "handmade_train_n", "validation_writer_n"],
               [[held, *[w for w in WRITERS if w != held], 76800, 1920, 640] for held in WRITERS])
    if args.check_only:
        print("CHECK-ONLY PASSED. Source SHA and train/validation indices saved, no models trained.", flush=True)
        return

    (out / "trained_models").mkdir(exist_ok=True)
    fit_records = []
    losses = {}
    print("Training Normal MLP once (same baseline conditions as W02)", flush=True)
    normal = new_mlp()
    t = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normal.fit(official["X_train"], official["y_train"])
    normal_fit = time.perf_counter()-t
    normal_loss = list(normal.named_steps["model"].loss_curve_)
    dump(normal, out / "trained_models" / "normal_mlp.joblib", compress=3)
    losses["Normal"] = normal_loss
    warnings_normal = [f"{w.category.__name__}: {w.message}" for w in caught]
    pred_official_normal = normal.predict(official["X_val"])
    official_normal = metrics(official["y_val"], pred_official_normal)
    fit_records.append(["Normal", "all", 76800, normal_fit,
                        normal.named_steps["model"].n_iter_, normal.named_steps["model"].loss_])
    print("Normal official validation:", official_normal, flush=True)

    rows_pred, rows_fold, rows_official, loss_records = [], [], [], []
    all_true, all_normal, all_combined = [], [], []
    rows_official.append(["Normal", "all", *official_normal.values()])
    for i, val_writer in enumerate(WRITERS, start=1):
        train_writers = [w for w in WRITERS if w != val_writer]
        X_hand = np.concatenate([writers[w]["X"] for w in train_writers], axis=0)
        y_hand = np.concatenate([writers[w]["y"] for w in train_writers], axis=0)
        X_combined = np.concatenate([official["X_train"], X_hand], axis=0)
        y_combined = np.concatenate([official["y_train"], y_hand], axis=0)
        assert X_combined.shape == (78720, 784) and y_combined.shape == (78720,)
        assert val_writer not in train_writers

        print(f"Fold {i}/4: Combined train writers {train_writers}, held-out={val_writer}", flush=True)
        combined = new_mlp()
        t = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            combined.fit(X_combined, y_combined)
        fit_seconds = time.perf_counter()-t
        fit_records.append(["Combined", val_writer, 78720, fit_seconds,
                            combined.named_steps["model"].n_iter_, combined.named_steps["model"].loss_])
        warnings_combined = [f"{w.category.__name__}: {w.message}" for w in caught]
        (out / f"fold_{val_writer}_warnings.json").write_text(json.dumps(warnings_combined, indent=2), encoding="utf-8")
        losses[f"Combined held-out {val_writer}"] = list(combined.named_steps["model"].loss_curve_)
        dump(combined, out / "trained_models" / f"combined_heldout_{val_writer}_mlp.joblib", compress=3)

        Xt, yt = writers[val_writer]["X"], writers[val_writer]["y"]
        t = time.perf_counter()
        pn = normal.predict(Xt)
        tn = time.perf_counter()-t
        t = time.perf_counter()
        pc = combined.predict(Xt)
        tc = time.perf_counter()-t
        mn, mc = metrics(yt, pn), metrics(yt, pc)
        rows_fold.append([val_writer, *train_writers,
                          mn["accuracy"], mn["macro_f1"], mc["accuracy"], mc["macro_f1"],
                          mc["accuracy"] - mn["accuracy"], fit_seconds, tn, tc])
        print(f"  held-out {val_writer}: Normal {mn['accuracy']:.4f}, Combined {mc['accuracy']:.4f}", flush=True)
        save_cm(out / f"fold_{val_writer}_normal_confusion_matrix.csv", yt, pn)
        save_cm(out / f"fold_{val_writer}_combined_confusion_matrix.csv", yt, pc)
        save_per_class(out / f"fold_{val_writer}_combined_per_class.csv", yt, pc)

        for j, (true, n, c) in enumerate(zip(yt, pn, pc)):
            rows_pred.append([val_writer, j, LABELS[true], LABELS[n], LABELS[c], int(n == true), int(c == true)])
        all_true.extend(yt.tolist())
        all_normal.extend(pn.tolist())
        all_combined.extend(pc.tolist())

        oc = metrics(official["y_val"], combined.predict(official["X_val"]))
        rows_official.append(["Combined", val_writer, *oc.values()])

        del X_hand, y_hand, X_combined, y_combined, combined

    y = np.asarray(all_true, dtype=int)
    pn = np.asarray(all_normal, dtype=int)
    pc = np.asarray(all_combined, dtype=int)
    assert len(y) == len(pn) == len(pc) == 2560
    normal_oof = metrics(y, pn)
    combined_oof = metrics(y, pc)
    save_table(out / "fold_comparison.csv",
               ["held_out_writer", "training_writer_1", "training_writer_2", "training_writer_3",
                "normal_acc", "normal_macro_f1", "combined_acc", "combined_macro_f1",
                "combined_minus_normal_accuracy", "combined_fit_seconds", "normal_predict_seconds", "combined_predict_seconds"], rows_fold)
    save_table(out / "OOF_comparison.csv", ["model", "n", "accuracy", "macro_f1"],
               [["Normal", *normal_oof.values()], ["Combined", *combined_oof.values()]])
    save_table(out / "OOF_predictions.csv",
               ["writer", "index_in_writer", "true_label", "normal_pred", "combined_pred", "normal_correct", "combined_correct"], rows_pred)
    save_table(out / "OOF_errors.csv",
               ["writer", "index_in_writer", "true_label", "normal_pred", "combined_pred", "normal_correct", "combined_correct"],
               [row for row in rows_pred if row[-1] == 0 or row[-2] == 0])
    save_table(out / "combined_errors_only.csv",
               ["writer", "index_in_writer", "true_label", "normal_pred", "combined_pred", "normal_correct", "combined_correct"],
               [row for row in rows_pred if row[-1] == 0])
    save_cm(out / "normal_OOF_confusion_matrix.csv", y, pn)
    save_cm(out / "combined_OOF_confusion_matrix.csv", y, pc)
    save_per_class(out / "normal_OOF_per_class.csv", y, pn)
    save_per_class(out / "combined_OOF_per_class.csv", y, pc)
    save_table(out / "fit_times_and_iterations.csv",
               ["model", "held_out_writer", "n_train", "fit_seconds", "n_iter", "final_loss"], fit_records)
    save_table(out / "official_development_validation.csv",
               ["model", "held_out_writer", "n", "accuracy", "macro_f1"], rows_official)
    loss_rows = [[name, i, val] for name, values in losses.items() for i, val in enumerate(values, start=1)]
    save_table(out / "loss_curves.csv", ["model_fold", "iteration", "loss"], loss_rows)
    make_plots(out, y, pn, pc, losses)
    run_info = {
        "experiment": "EXP-CMB-001", "status": "completed", "official_test_used": False,
        "n_official_train": 76800, "n_official_development_val": 19200,
        "n_combined_train_per_fold": 78720, "n_writer_holdout_per_fold": 640,
        "cv": "Leave-one-writer-out, actual writers 41,42,43,44",
        "split_indices": "official_split_indices.npz",
        "split_manifest": "training_validation_manifest.json",
        "trained_models_saved": "trained_models/normal_mlp.joblib and trained_models/combined_heldout_*_mlp.joblib",
        "mlp": "Pipeline(PixelScaler255, MLPClassifier(random_state=42))",
        "normal_fit_seconds": normal_fit, "normal_train_iterations": len(normal_loss),
        "normal_warnings": warnings_normal,
        "normal_handmade_oof": normal_oof,
        "combined_handmade_oof": combined_oof,
        "combined_minus_normal_accuracy_pp": 100*(combined_oof["accuracy"]-normal_oof["accuracy"]),
        "normal_official_dev_validation": official_normal,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__, "input_sha256": sha_info,
        "notes": ["Current standardized writer NPZs used without manual orientation/inversion changes.",
                  "OOF is DEVELOPMENT evaluation, not independent final Test.",
                  "Four Combined folds are four different fitted models, not a single final model."]
    }
    (out / "run_summary.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("FINISHED", json.dumps({"Normal_OOF": normal_oof, "Combined_OOF": combined_oof}, ensure_ascii=False), flush=True)
    print("Outputs:", out, flush=True)


if __name__ == "__main__":
    main()