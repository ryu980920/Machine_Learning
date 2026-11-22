"""Audit saved predictions and generate Extra Trees result tables and figures."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_extra_trees import (
    DATA, FOCUS, LABELS, PREPROCESSING, TUNING, WRITERS, choose,
    load_official, load_writers, official_average, score, write_csv,
)

METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1",
           "digits_accuracy", "letters_accuracy")
BLUE, ORANGE, GREEN = "#1261a0", "#b84e1b", "#148471"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.labelcolor": "#243e51",
                     "font.size": 10, "savefig.facecolor": "white"})


def read_json(path):
    return json.loads(path.read_text())


def save(figure, path):
    figure.savefig(path, dpi=180, bbox_inches="tight", pad_inches=.15)
    plt.close(figure)


def load_and_verify(out, combined):
    official = load_official(DATA / "datasets/MNIST_EMNIST_official_split_112000.npz")
    y_val = official["y_val"]
    writers = load_writers(DATA / "Hand_made")
    y_hand = np.concatenate([writers[w]["y"] for w in WRITERS])
    runs, predictions = {}, {}
    for folder in sorted((out / "runs").iterdir()):
        if not (folder / "summary.json").exists():
            continue
        run = read_json(folder / "summary.json")
        name = run["name"]
        runs[name] = run
        with np.load(folder / "predictions.npz", allow_pickle=False) as z:
            data = dict(z)
        predictions[name] = data
        hand = np.concatenate([data[f"writer_{w}"] for w in WRITERS])
        if score(y_hand, hand) != run["hand"]:
            raise ValueError(f"Hand-made metrics mismatch in {name}")
        is_combined = combined and name != "CTRL_normal"
        expected_folds = {str(w) for w in WRITERS} if is_combined else {"all"}
        if set(run["official"]) != expected_folds or set(run["timing"]) != expected_folds:
            raise ValueError(f"Incorrect validation folds in {name}")
        for fold in expected_folds:
            times = run["timing"][fold]
            train_ids = [w for w in WRITERS if w != int(fold)] if is_combined else []
            if times["training_writers"] != train_ids or times["handmade_train_n"] != 640*len(train_ids):
                raise ValueError(f"Incorrect writer split in {name}/{fold}")
            if score(y_val, data[f"official_{fold}"]) != run["official"][fold]:
                raise ValueError(f"Official metrics mismatch in {name}/{fold}")
        for writer in WRITERS:
            if score(writers[writer]["y"], data[f"writer_{writer}"]) != run["writers"][str(writer)]:
                raise ValueError(f"Writer metrics mismatch in {name}/{writer}")
    return runs, predictions, writers


def plot_geometry(geometry, out):
    chosen = [row for row in geometry if row["dataset"] != "official_train"]
    names = ["Official val", "Writer 41", "Writer 42", "Writer 43", "Writer 44"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
    for ax, key, title in zip(axes, ("long_side", "ink_pixels"),
                             ("Bounding-box long side (px)", "Ink pixels at intensity ≥32")):
        x = np.arange(len(names))
        for offset, family, color in ((-.17, "digits", BLUE), (.17, "letters", ORANGE)):
            ax.bar(x+offset, [float(row[key]) for row in chosen if row["family"] == family],
                   .33, label=family.title(), color=color)
        ax.set(title=title, xticks=x, xticklabels=names)
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend(frameon=False)
    save(fig, out / "input_geometry.png")


def plot_stages(preprocessing, tuning, selected, floor, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    for ax, metric, ylabel in zip(axes, ("official", "hand"),
                                  ("Official validation accuracy (%)", "Hand-made macro F1 (%)")):
        for i, group in enumerate((preprocessing, tuning)):
            index = np.arange(len(group)) + 7*i
            values = [official_average(row)*100 if metric == "official"
                      else row["hand"]["macro_f1"]*100 for row in group]
            ax.plot(index, values, marker="o", color=(BLUE, GREEN)[i], linewidth=1.4)
            for x, value, row in zip(index, values, group):
                if row["name"] == selected:
                    ax.scatter(x, value, s=95, facecolors="none", edgecolors=ORANGE,
                               linewidth=2, zorder=4)
        ax.set(xticks=[*range(6), *range(7, 13)],
               xticklabels=[r["name"][:3] for r in preprocessing+tuning], ylabel=ylabel)
        ax.tick_params(axis="x", labelrotation=40)
        ax.grid(axis="y", alpha=.25)
    axes[0].axhline(floor*100, color=ORANGE, linestyle="--", label="Eligibility floor")
    axes[0].legend(frameon=False)
    save(fig, out / "stage_comparison.png")


def plot_matrices(raw, final, out):
    pairs = [("Hand-made baseline", [raw["hand"]]), ("Hand-made selected", [final["hand"]]),
             ("Official validation selected", list(final["official"].values()))]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    for ax, (title, rows) in zip(axes, pairs):
        counts = np.asarray([r["confusion_matrix"] for r in rows]).mean(axis=0)
        normalized = counts/counts.sum(axis=1, keepdims=True)
        im = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
        ax.set(title=title, xticks=range(16), yticks=range(16),
               xticklabels=list(LABELS), yticklabels=list(LABELS), xlabel="Predicted", ylabel="True")
        ax.tick_params(labelsize=7, length=0)
    fig.colorbar(im, ax=axes, shrink=.69, label="Row share")
    save(fig, out / "confusion_matrices.png")


def plot_errors(raw, final, writers, predictions, out):
    fig, ax = plt.subplots(figsize=(8.9, 3), layout="constrained")
    for offset, row, color, name in ((-.17, raw, ORANGE, "Baseline"), (.17, final, BLUE, "Selected")):
        ax.bar(np.arange(len(FOCUS))+offset, list(row["hand"]["focused_errors"].values()),
               .33, color=color, label=name)
    ax.set(xticks=np.arange(len(FOCUS)), xticklabels=[f"{s}→{d}" for s, d in FOCUS],
           ylabel="Wrong predictions (of 160 per class)")
    ax.legend(frameon=False)
    save(fig, out / "directional_errors.png")

    raw_pred, final_pred = (predictions[r["name"]] for r in (raw, final))
    fig, axes = plt.subplots(2, len(FOCUS), figsize=(10, 3), layout="constrained")
    examples = []
    for column, (source, dest) in enumerate(FOCUS):
        correct, wrong = LABELS.index(source), LABELS.index(dest)
        for row_id, outcome in enumerate(("corrected", "remains")):
            ax = axes[row_id, column]
            ax.set_axis_off()
            for w, entry in writers.items():
                before, after = raw_pred[f"writer_{w}"], final_pred[f"writer_{w}"]
                match = (entry["y"] == correct) & (before == wrong)
                match &= after == (correct if outcome == "corrected" else wrong)
                indices = np.flatnonzero(match)
                if indices.size:
                    i = indices[0]
                    ax.imshow(entry["X"][i].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
                    ax.set_title(f"{source}→{dest}  W{w}", fontsize=8)
                    examples.append(dict(outcome=outcome, writer=w, row_index_zero_based=i,
                                         label=source, baseline_pred=dest,
                                         selected_pred=LABELS[after[i]]))
                    break
            else:
                ax.text(.5, .5, "—", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle("Top: corrected examples   •   Bottom: errors retained")
    save(fig, out / "error_examples.png")
    if examples:
        write_csv(out / "error_examples.csv", examples)


def plot_curve(curve, out):
    fig, ax = plt.subplots(figsize=(8.7, 3.5), layout="constrained")
    x = [row["official_train_n"] for row in curve]
    for key, name, color in (("train_accuracy", "Training (resubstitution)", GREEN),
                             ("official_accuracy", "Official validation", BLUE),
                             ("hand_accuracy", "Hand-made development", ORANGE)):
        ax.plot(x, [row[key]*100 for row in curve], "-o", label=name, color=color)
    ax.set(xlabel="Official training images", ylabel="Accuracy (%)", xticks=x)
    ax.grid(alpha=.18)
    ax.legend(frameon=False)
    save(fig, out / "learning_curve.png")


def control(runs, predictions, writers, out):
    left, right = runs["CTRL_normal"], runs["R42_selected"]
    rows = []
    for w in WRITERS:
        key = str(w)
        y = writers[w]["y"]
        old = predictions[left["name"]][f"writer_{w}"] == y
        new = predictions[right["name"]][f"writer_{w}"] == y
        rows.append(dict(writer=w, n=len(y), normal_accuracy=old.mean(),
                         combined_accuracy=new.mean(), delta_accuracy=new.mean()-old.mean(),
                         normal_macro_f1=left["writers"][key]["macro_f1"],
                         combined_macro_f1=right["writers"][key]["macro_f1"],
                         corrected=int((~old & new).sum()), regressed=int((old & ~new).sum())))
    write_csv(out / "matched_writers.csv", rows)
    directions = []
    for key, old in left["hand"]["focused_errors"].items():
        new = right["hand"]["focused_errors"][key]
        directions.append(dict(direction=key, normal_errors=old, combined_errors=new,
                               difference=new-old, support=160))
    write_csv(out / "matched_directional_errors.csv", directions)
    fig, ax = plt.subplots(figsize=(7.8, 3.2), layout="constrained")
    for row in rows:
        y = row["writer"]
        a, b = row["normal_accuracy"]*100, row["combined_accuracy"]*100
        ax.plot([a, b], [y, y], color="#a9b6bf", linewidth=2)
        ax.scatter([a, b], [y, y], color=[ORANGE, BLUE])
    ax.scatter([], [], color=ORANGE, label="Official only")
    ax.scatter([], [], color=BLUE, label="+ 1,920 Hand-made")
    ax.set(yticks=WRITERS, xlabel="Held-out writer accuracy (%)", ylabel="Writer")
    ax.legend(frameon=False)
    save(fig, out / "matched_writers.png")


def aggregate(source, out, mode):
    out.mkdir(parents=True, exist_ok=True)
    combined = mode == "combined"
    runs, predictions, writers = load_and_verify(source, combined)
    selection = read_json(source / "selection.json")
    required = {*PREPROCESSING, *TUNING, "R42_selected", "R17_selected", "R73_selected",
                "LC20", "LC40", "LC60", "LC80"}
    if combined:
        required.add("CTRL_normal")
    if absent := required - runs.keys():
        raise ValueError(f"Missing runs: {absent}")
    raw, final = runs["P00_raw"], runs["R42_selected"]
    prep = [runs[name] for name in PREPROCESSING]
    tune = [runs[name] for name in TUNING]
    picked = choose(prep+tune, selection["official_accuracy_floor"])
    if picked["name"] != selection["selected"] or final["config"] != picked["config"]:
        raise ValueError("Selection could not be reproduced")
    for key, values in predictions[selection["selected"]].items():
        if not np.array_equal(predictions["R42_selected"][key], values):
            raise ValueError(f"Same-seed repeat differs: {key}")

    summary, folds, classes, matrix, pairs = [], [], [], [], []
    for name, r in runs.items():
        measurements = list(r["timing"].values())
        summary.append(dict(run=name, seed=r["seed"], source=r["training_source"],
                            fraction=r["fraction"], **r["config"],
                            **{f"official_{m}": official_average(r, m) for m in METRICS},
                            **{f"hand_{m}": r["hand"][m] for m in METRICS},
                            official_accuracy_fold_sd=np.std([x["accuracy"] for x in r["official"].values()],
                                                             ddof=1) if len(measurements)>1 else "",
                            fit_seconds_mean=np.mean([t["fit_seconds"] for t in measurements]),
                            val_predict_ms_image=np.mean([t["val_predict_seconds"] for t in measurements])*1000/19200,
                            hand_predict_ms_image=np.mean([t["hand_predict_seconds"]*1000/t["hand_eval_n"]
                                                           for t in measurements])))
        for fold, values in r["official"].items():
            folds.append(dict(run=name, evaluation="official_validation", fold=fold,
                              writers_train=" ".join(map(str, r["timing"][fold]["training_writers"])),
                              n=values["n"], **{m: values[m] for m in METRICS},
                              **{key: val for key, val in r["timing"][fold].items()
                                 if key not in ("training_writers", "train_accuracy")}))
        evaluations = [("handmade_pooled", "all", r["hand"])]
        evaluations.extend(("handmade_writer", w, values) for w, values in r["writers"].items())
        evaluations.extend(("official_validation", f, values) for f, values in r["official"].items())
        for name_of_set, fold, values in evaluations:
            scope = dict(run=name, evaluation=name_of_set, fold=fold)
            classes.extend(dict(**scope, **c) for c in values["classes"])
            for actual, counts in zip(LABELS, values["confusion_matrix"]):
                matrix.append(dict(**scope, actual=actual, **dict(zip(LABELS, counts))))
            for pair, n in values["focused_errors"].items():
                support = values["classes"][LABELS.index(pair[0])]["support"]
                pairs.append(dict(**scope, direction=pair, errors=n, support=support, rate=n/support))
    for filename, rows in (("experiment_summary.csv", summary), ("fold_metrics.csv", folds),
                           ("class_metrics.csv", classes), ("confusion_matrices.csv", matrix),
                           ("directional_errors.csv", pairs)):
        write_csv(out / filename, rows)
    curve = []
    for name in ("LC20", "LC40", "LC60", "LC80", "R42_selected"):
        r = runs[name]
        t = list(r["timing"].values())
        curve.append(dict(run=name, fraction=r["fraction"], official_train_n=t[0]["official_train_n"],
                          handmade_train_n=t[0]["handmade_train_n"],
                          train_accuracy=np.mean([x["train_accuracy"] for x in t]),
                          official_accuracy=official_average(r), hand_accuracy=r["hand"]["accuracy"],
                          hand_macro_f1=r["hand"]["macro_f1"],
                          fit_seconds_mean=np.mean([x["fit_seconds"] for x in t])))
    write_csv(out / "learning_curve.csv", curve)
    with (source / "input_geometry.csv").open() as f:
        geometry = list(csv.DictReader(f))
    plot_geometry(geometry, out)
    plot_stages(prep, tune, selection["selected"], selection["official_accuracy_floor"], out)
    plot_matrices(raw, final, out)
    plot_errors(raw, final, writers, predictions, out)
    plot_curve(curve, out)
    if combined:
        control(runs, predictions, writers, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("normal", "combined"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    aggregate(args.source, args.out, args.mode)
