"""Verify archived predictions and produce paired Extra Trees development diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "Normal" / "scripts"))
from run_extra_trees import DATA, LABELS, WRITERS, load_official, load_writers
from run_extra_trees import official_average, score, write_csv


def read_summary(folder):
    return json.loads((folder / "summary.json").read_text())


def verify_scores(expected, actual):
    for key in ("n", "correct", "accuracy", "macro_f1", "macro_precision",
                "macro_recall", "confusion_matrix", "focused_errors"):
        if expected[key] != actual[key]:
            raise ValueError(f"Prediction and summary disagree for {key}")


def verify_runs(folders, official, writers, table_path=None):
    labels = np.concatenate([writers[w]["y"] for w in WRITERS])
    table = {}
    if table_path is not None:
        with table_path.open(newline="") as stream:
            table = {row["run"]: row for row in csv.DictReader(stream)}
        if set(table) != {folder.name for folder in folders}:
            raise ValueError(f"CSV run list differs from archive: {table_path}")
    for folder in folders:
        summary = read_summary(folder)
        combined = summary["training_source"] == "combined"
        folds = {str(writer) for writer in WRITERS} if combined else {"all"}
        if set(summary["official"]) != folds or set(summary["timing"]) != folds:
            raise ValueError(f"Unexpected folds: {folder}")
        for fold in folds:
            planned_writers = [writer for writer in WRITERS if writer != int(fold)] if combined else []
            records = summary["timing"][fold]
            if records["training_writers"] != planned_writers or (
                records["handmade_train_n"] != sum(len(writers[w]["y"]) for w in planned_writers)
            ):
                raise ValueError(f"Writer holdout mismatch: {folder}, {fold}")
        with np.load(folder / "predictions.npz", allow_pickle=False) as predictions:
            for writer in WRITERS:
                key = str(writer)
                verify_scores(summary["writers"][key],
                              score(writers[writer]["y"], predictions[f"writer_{key}"]))
            pooled = np.concatenate([predictions[f"writer_{w}"] for w in WRITERS])
            verify_scores(summary["hand"], score(labels, pooled))
            for fold, metrics in summary["official"].items():
                verify_scores(metrics, score(official["y_val"], predictions[f"official_{fold}"]))
        if table:
            row = table[folder.name]
            values = {"official_accuracy": official_average(summary),
                      "official_macro_f1": official_average(summary, "macro_f1"),
                      "hand_accuracy": summary["hand"]["accuracy"],
                      "hand_macro_f1": summary["hand"]["macro_f1"]}
            if any(not np.isclose(value, float(row[key]), rtol=0, atol=1e-12)
                   for key, value in values.items()):
                raise ValueError(f"CSV scores differ from predictions: {folder}")


def error_diagnostics(out, writers):
    directional_rows, changed_examples = [], []
    for mode in ("Normal", "Combined"):
        runs = BASE / mode / "results" / "runs"
        with np.load(runs / "P00_raw" / "predictions.npz") as old, np.load(
            runs / "R42_selected" / "predictions.npz"
        ) as new:
            for writer in WRITERS:
                y = writers[writer]["y"]
                previous, current = (run[f"writer_{writer}"] for run in (old, new))
                is_seven = y == 7
                before = is_seven & (previous == 9)
                after = is_seven & (current == 9)
                directional_rows.append(dict(
                    mode=mode, writer=writer, true_sevens=int(is_seven.sum()),
                    correct_before=int((is_seven & (previous == 7)).sum()),
                    correct_after=int((is_seven & (current == 7)).sum()),
                    before=int(before.sum()), after=int(after.sum()),
                    persistent=int((before & after).sum()),
                    resolved_to_correct=int((before & (current == 7)).sum()),
                    resolved_to_other_error=int((before & ~after & (current != 7)).sum()),
                    introduced_from_correct=int((after & (previous == 7)).sum()),
                    introduced_from_other_error=int((after & ~before & (previous != 7)).sum()),
                ))
                for index in np.flatnonzero(before != after):
                    changed_examples.append(dict(
                        comparison=f"{mode}_baseline_to_selected", writer=writer, index=int(index),
                        actual="7", before=LABELS[previous[index]], after=LABELS[current[index]],
                    ))

    runs = BASE / "Combined" / "results" / "runs"
    y = writers[41]["y"]
    class_rows = []
    with np.load(runs / "CTRL_normal" / "predictions.npz") as old, np.load(
        runs / "R42_selected" / "predictions.npz"
    ) as new:
        previous, current = old["writer_41"], new["writer_41"]
        for i, label in enumerate(LABELS):
            in_class = y == i
            before_correct, after_correct = in_class & (previous == y), in_class & (current == y)
            class_rows.append(dict(
                actual=label, n=int(in_class.sum()),
                errors_before=int(in_class.sum() - before_correct.sum()),
                errors_after=int(in_class.sum() - after_correct.sum()),
                corrected=int((after_correct & ~before_correct).sum()),
                regressed=int((before_correct & ~after_correct).sum()),
            ))
        for index in np.flatnonzero((previous == y) != (current == y)):
            changed_examples.append(dict(
                comparison="writer41_official_only_to_combined", writer=41, index=int(index),
                actual=LABELS[y[index]], before=LABELS[previous[index]], after=LABELS[current[index]],
            ))
    write_csv(out / "seven_to_nine_by_writer.csv", directional_rows)
    write_csv(out / "writer41_by_class.csv", class_rows)
    write_csv(out / "paired_error_examples.csv", changed_examples)
    for mode in ("Normal", "Combined"):
        seven_gallery(out, mode, changed_examples, writers)


def seven_gallery(out, mode, examples, writers):
    source = [row for row in examples
              if row["comparison"] == f"{mode}_baseline_to_selected"]
    new_errors = sorted((row for row in source if row["after"] == "9"),
                        key=lambda row: (row["before"] != "7", row["writer"], row["index"]))
    corrected = [row for row in source if row["before"] == "9" and row["after"] == "7"]
    panels = new_errors[:4] + corrected[:4]
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 1.6))
    for ax, row in zip(axes, panels):
        image = writers[row["writer"]]["X"][row["index"]].reshape(28, 28)
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{row['writer']} / {row['index']}\n{row['before']} → {row['after']}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"seven_to_nine_{mode.lower()}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def summarize_controls(out):
    archive = BASE / "Combined" / "results" / "runs"
    rows = []
    for arm in ("stroke00", "features08"):
        for seed in (42, 17, 73):
            reference = read_summary(archive / f"R{seed}_selected")
            control = read_summary(out / "runs" / f"C{seed}_{arm}")
            expected_key = {"stroke00": "prep", "features08": "max_features"}[arm]
            differences = {key for key, value in reference["config"].items()
                           if value != control["config"][key]}
            if differences != {expected_key} or reference["seed"] != control["seed"]:
                raise ValueError(f"Unmatched controls: {arm}, {seed}")
            for name, run in ((arm, control), ("T02_reference", reference)):
                rows.append(dict(
                    comparison=arm, arm=name, seed=seed,
                    official_accuracy=official_average(run), hand_accuracy=run["hand"]["accuracy"],
                    hand_macro_f1=run["hand"]["macro_f1"], hand_7_to_9=run["hand"]["focused_errors"]["7->9"],
                    writer41_accuracy=run["writers"]["41"]["accuracy"],
                    fit_seconds_mean=float(np.mean([f["fit_seconds"] for f in run["timing"].values()])),
                ))
    write_csv(out / "paired_controls.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    official = load_official(DATA / "datasets/MNIST_EMNIST_official_split_112000.npz")
    writers = load_writers(DATA / "Hand_made")
    total = 0
    for mode in ("Normal", "Combined"):
        results = BASE / mode / "results"
        folders = sorted((results / "runs").iterdir())
        verify_runs(folders, official, writers, results / "experiment_summary.csv")
        total += len(folders)
    error_diagnostics(args.out, writers)
    if args.controls:
        folders = sorted((args.out / "runs").iterdir())
        verify_runs(folders, official, writers)
        summarize_controls(args.out)
        total += len(folders)
    print(f"Verified {total} archived prediction and summary pairs.", flush=True)


if __name__ == "__main__":
    main()
