"""Run the Extra Trees development study from the team repository."""
import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[1]
MLP = BASE.parent / "MLP"
sys.path[:0] = [str(MLP / mode / "scripts") for mode in ("Combined", "Normal")]
from combined_mlp_writer_cv import LABELS, WRITERS, load_official, load_writers
from exp006_size_com_comparison import process
from exp_cmb005_balanced_scale_stroke import prepareraw
from skimage.feature import hog

DATA = ROOT / "02_data"
FOCUS = ("0D", "D0", "8B", "B8", "5F", "F5", "79", "97")


@dataclass(frozen=True)
class Config:
    prep: str
    features: str = "pixels"
    n_estimators: int = 100
    max_features: str | float = "sqrt"
    min_samples_leaf: int = 1
    max_depth: int | None = None


PREPROCESSING = {
    "P00_raw": Config("raw"),
    "P01_com": Config("com"),
    "P02_size": Config("size"),
    "P03_stroke": Config("stroke"),
    "P04_hybrid": Config("stroke", "pixels+hog"),
    "P05_hog": Config("stroke", "hog"),
}
TUNING = {
    "T01_feat08": {"max_features": .08},
    "T02_feat18": {"max_features": .18},
    "T03_leaf2": {"min_samples_leaf": 2},
    "T04_depth24": {"max_depth": 24},
    "T05_trees64": {"n_estimators": 64},
    "T06_trees160": {"n_estimators": 160},
}


def write_json(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    pending.replace(path)


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def score(y, prediction):
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=np.arange(16), zero_division=0)
    cm = confusion_matrix(y, prediction, labels=np.arange(16))
    return {
        "n": len(y), "correct": int(np.trace(cm)), "accuracy": float(np.mean(y == prediction)),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "digits_accuracy": float(np.mean(y[y < 10] == prediction[y < 10])),
        "letters_accuracy": float(np.mean(y[y >= 10] == prediction[y >= 10])),
        "confusion_matrix": cm.tolist(),
        "classes": [dict(label=label, precision=p, recall=r, f1=f, support=int(n),
                         errors=int(n-cm[i, i])) for i, (label, p, r, f, n) in
                    enumerate(zip(LABELS, precision, recall, f1, support))],
        "focused_errors": {f"{src}->{dst}": int(cm[LABELS.index(src), LABELS.index(dst)])
                           for src, dst in FOCUS},
    }


def extract(x, features):
    if features == "pixels":
        return x
    descriptors = np.empty((len(x), 324), np.float32)
    for i, image in enumerate(x.reshape(-1, 28, 28)):
        descriptors[i] = hog(image, orientations=9, pixels_per_cell=(7, 7),
                             cells_per_block=(2, 2), block_norm="L2-Hys")
    return descriptors if features == "hog" else np.column_stack((x, descriptors))


class Study:
    def __init__(self, args):
        self.args = args
        self.official = load_official(DATA / "datasets/MNIST_EMNIST_official_split_112000.npz")
        self.writers = load_writers(DATA / "Hand_made")
        self.y_train = self.official["y_train"]
        self.y_val = self.official["y_val"]
        self.y_hand = {w: self.writers[w]["y"] for w in WRITERS}
        self.y_oof = np.concatenate(list(self.y_hand.values()))
        self.cache = {}
        self.geometry = {}

    def audit(self):
        reference = MLP / "Combined/results/EXP_CMB_001/official_split_indices.npz"
        with np.load(reference, allow_pickle=False) as saved:
            keys = ("train_idx", "val_idx", "train_group_ids", "val_group_ids")
            if any(not np.array_equal(saved[key], self.official[key]) for key in keys):
                raise ValueError("Official split differs from archived team split")
        sources = {
            "official_train": self.official["X_train"], "official_validation": self.official["X_val"],
            **{f"writer_{w}": self.writers[w]["X"] for w in WRITERS},
        }
        exact = {key: {row.tobytes() for row in x.astype(np.uint8)} for key, x in sources.items()}
        overlap = {f"{a}/{b}": len(exact[a] & exact[b]) for i, a in enumerate(exact)
                   for b in list(exact)[i+1:] if a.startswith("writer_") or b.startswith("writer_")}
        if any(overlap.values()):
            raise ValueError(f"Hand-made exact pixel duplicates across sources: {overlap}")
        geometry = []
        for key, x in sources.items():
            y = self.y_train if key == "official_train" else self.y_val
            if key.startswith("writer_"):
                y = self.y_hand[int(key.split("_")[-1])]
            mask = x.reshape(-1, 28, 28) >= 32
            height, width = mask.any(2).sum(1), mask.any(1).sum(1)
            # Bounding-box spans include blank internal rows/columns.
            rows, cols = mask.any(2), mask.any(1)
            height = np.where(height, 28-rows[:, ::-1].argmax(1)-rows.argmax(1), 0)
            width = np.where(width, 28-cols[:, ::-1].argmax(1)-cols.argmax(1), 0)
            ink = mask.sum((1, 2))
            for family, subset in (("digits", y < 10), ("letters", y >= 10)):
                geometry.append(dict(dataset=key, family=family, n=int(subset.sum()),
                                     long_side=float(np.maximum(height, width)[subset].mean()),
                                     ink_pixels=float(ink[subset].mean())))
        write_csv(self.args.out / "input_geometry.csv", geometry)
        np.savez_compressed(self.args.out / "official_split_indices.npz",
                            **{key: self.official[key] for key in keys})
        write_json(self.args.out / "data_audit.json", {
            "repository_commit": subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                                         text=True).strip(),
            "sha256_official": self.official["sha256"],
            "sha256_handmade": {w: self.writers[w]["sha256"] for w in WRITERS},
            "official_train_counts": dict(zip(LABELS, self.official["train_class_counts"])),
            "official_validation_counts": dict(zip(LABELS, self.official["val_class_counts"])),
            "official_group_overlap": 0, "archived_split_exact_match": True,
            "unique_exact_pixel_overlap_with_handmade": overlap,
            "writer_class_counts": {w: dict(zip(LABELS, np.bincount(self.y_hand[w]).tolist()))
                                    for w in WRITERS},
            "test_arrays_loaded": [],
        })
        environment = {
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu": next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                         if s.startswith("model name")), platform.processor())
            if Path("/proc/cpuinfo").exists() else platform.processor(),
            "logical_cpus": os.cpu_count(), "jobs": self.args.jobs,
            "packages": {p: importlib.metadata.version(p) for p in
                         ("numpy", "scipy", "scikit-learn", "scikit-image",
                          "opencv-python-headless", "matplotlib")},
        }
        path = self.args.out / "environment.json"
        if path.exists() and json.loads(path.read_text()) != environment:
            raise ValueError("Runtime differs from that of the saved runs. Choose a new --out.")
        write_json(path, environment)

    def inputs(self, config):
        key = (config.prep, config.features)
        if key not in self.cache:
            self.cache.clear()
            started = perf_counter()
            train, val = self.official["X_train"], self.official["X_val"]
            hand = {w: self.writers[w]["X"] for w in WRITERS}
            details, assignments = {}, []
            if config.prep != "raw":
                val = process(val, "com", 20)
                if config.prep == "com":
                    train = process(train, "com", 20)
                    hand = {w: process(x, "com", 20) for w, x in hand.items()}
                else:
                    arm = {"size": "scale30_stroke00", "stroke": "scale30_stroke28"}[config.prep]
                    train, hand, details, assignments = prepareraw(self.official, self.writers, arm)
            matrices = (extract(train, config.features), extract(val, config.features),
                        {w: extract(x, config.features) for w, x in hand.items()})
            self.cache[key] = matrices
            self.geometry[key] = {
                "prepare_seconds_all_sets": perf_counter() - started,
                "feature_dimension": matrices[0].shape[1],
                "augmentation": details, "size_assignments": assignments,
            }
        return self.cache[key]

    def run(self, name, config, seed=42, fraction=1., source=None):
        source = source or self.args.mode
        folder = self.args.out / "runs" / name
        done = folder / "summary.json"
        if done.exists():
            result = json.loads(done.read_text())
            if (result["config"], result["seed"], result["fraction"], result["training_source"]) != (
                    asdict(config), seed, fraction, source):
                raise ValueError(f"Cached configuration mismatch: {name}")
            return result
        folder.mkdir(parents=True, exist_ok=True)
        x_train, x_val, x_hand = self.inputs(config)
        indices = self.subset(fraction)
        x_subset, y_subset = ((x_train, self.y_train) if fraction == 1.
                              else (x_train[indices], self.y_train[indices]))
        by_writer, official, timings, predictions = {}, {}, {}, {}
        for held in (WRITERS if source == "combined" else (None,)):
            training_writers = [w for w in WRITERS if w != held] if held is not None else []
            fold = str(held) if held is not None else "all"
            x_fit = np.concatenate([x_subset, *[x_hand[w] for w in training_writers]]) if training_writers else x_subset
            y_fit = np.concatenate([y_subset, *[self.y_hand[w] for w in training_writers]]) if training_writers else y_subset
            config_args = asdict(config)
            del config_args["prep"], config_args["features"]
            model = ExtraTreesClassifier(**config_args, random_state=seed, n_jobs=self.args.jobs)
            started = perf_counter()
            model.fit(x_fit, y_fit)
            fit_seconds = perf_counter()-started
            started = perf_counter()
            val_pred = model.predict(x_val)
            val_seconds = perf_counter()-started
            evaluated = [held] if held is not None else list(WRITERS)
            started = perf_counter()
            hand_pred = model.predict(np.concatenate([x_hand[w] for w in evaluated]))
            hand_seconds = perf_counter()-started
            for w, pred in zip(evaluated, np.split(hand_pred, len(evaluated))):
                by_writer[w] = score(self.y_hand[w], pred)
                predictions[f"writer_{w}"] = pred.astype(np.uint8)
            official[fold] = score(self.y_val, val_pred)
            predictions[f"official_{fold}"] = val_pred.astype(np.uint8)
            timings[fold] = dict(training_writers=training_writers, official_train_n=len(y_subset),
                                 handmade_train_n=len(y_fit)-len(y_subset), fit_seconds=fit_seconds,
                                 val_predict_seconds=val_seconds, hand_predict_seconds=hand_seconds,
                                 hand_eval_n=len(hand_pred),
                                 **({"train_accuracy": float(model.score(x_fit, y_fit))}
                                    if name.startswith("LC") or name.startswith("R42") else {}))
            print(f"{name} fold={fold}: official={official[fold]['accuracy']:.4%} "
                  f"hand={np.mean([by_writer[w]['accuracy'] for w in evaluated]):.4%} "
                  f"fit={fit_seconds:.1f}s", flush=True)
            del model, x_fit, y_fit
        hand = score(self.y_oof, np.concatenate([predictions[f"writer_{w}"] for w in WRITERS]))
        result = dict(name=name, config=asdict(config), fraction=fraction, seed=seed,
                      training_source=source, official=official, hand=hand, writers=by_writer,
                      timing=timings, preprocessing=self.geometry[(config.prep, config.features)])
        np.savez_compressed(folder / "predictions.npz", **predictions)
        write_json(done, result)
        return result

    def subset(self, fraction):
        if fraction == 1.:
            return np.arange(len(self.y_train))
        rng = np.random.default_rng(891)
        samples = []
        for label in range(16):
            positions = rng.permutation(np.flatnonzero(self.y_train == label))
            samples.append(positions[:round(len(positions)*fraction)])
        return np.sort(np.concatenate(samples))


def official_average(result, metric="accuracy"):
    return float(np.mean([fold[metric] for fold in result["official"].values()]))


def choose(results, floor):
    eligible = [r for r in results if official_average(r) >= floor]
    return max(eligible, key=lambda r: (r["hand"]["macro_f1"], r["hand"]["accuracy"],
                                        official_average(r), -r["config"]["n_estimators"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("normal", "combined"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--check-only", action="store_true", help="Audit input files without fitting")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    study = Study(args)
    study.audit()
    if args.check_only:
        print("PASS: official split, file SHA-256, labels, and writer separation")
        return
    preprocess = [study.run(name, config) for name, config in PREPROCESSING.items()]
    floor = official_average(preprocess[0])-.005
    best_prep = choose(preprocess, floor)
    initial = Config(**best_prep["config"])
    tuning = [study.run(name, replace(initial, **change)) for name, change in TUNING.items()]
    selected = choose(preprocess + tuning, floor)
    config = Config(**selected["config"])
    write_json(args.out / "selection.json",
               dict(preprocessing=best_prep["name"], selected=selected["name"], config=asdict(config),
                    official_accuracy_floor=floor, guardrail_drop=.005))
    if args.mode == "combined":
        study.run("CTRL_normal", config, source="normal")
    for seed in (42, 17, 73):
        study.run(f"R{seed}_selected", config, seed=seed)
    for fraction in (.2, .4, .6, .8):
        study.run(f"LC{int(100*fraction)}", config, fraction=fraction)
    print(f"COMPLETE: {args.mode}; selected {selected['name']}", flush=True)


if __name__ == "__main__":
    main()
