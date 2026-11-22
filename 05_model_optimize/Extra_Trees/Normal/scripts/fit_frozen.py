"""Fit one frozen Extra Trees candidate on the development training data."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from run_extra_trees import BASE, LABELS, WRITERS, Config, Study
from run_extra_trees import extract, process
from sklearn.ensemble import ExtraTreesClassifier


def predict(model, images, features):
    """Predict standardized 28 × 28 images with 0–255 pixel intensities."""
    array = np.asarray(images, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (28, 28) or len(array) == 0:
        raise ValueError("Expected a batch of standardized 28 × 28 images.")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 255:
        raise ValueError("Expected 0–255 pixel intensities.")
    x = process(array.reshape(-1, 784), "com", 20)
    return model.predict(extract(x, features))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("normal", "combined"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    manifest = BASE / "Combined" / "results" / "review" / "freeze.json"
    recipe = json.loads(manifest.read_text())[args.mode]
    config = Config(**recipe["config"])
    args.out.mkdir(parents=True, exist_ok=False)
    study = Study(args)
    study.audit()
    official, _, hands = study.inputs(config)
    y = study.y_train
    training_writers = recipe["training_writers"]
    if training_writers:
        if tuple(training_writers) != WRITERS:
            raise ValueError("Unexpected Combined writer set.")
        official = np.concatenate([official, *[hands[w] for w in training_writers]])
        y = np.concatenate([y, *[study.y_hand[w] for w in training_writers]])
    if len(y) != recipe["official_training_n"] + recipe["handmade_training_n"]:
        raise ValueError("Frozen training size mismatch.")

    model = ExtraTreesClassifier(n_estimators=config.n_estimators, max_features=config.max_features,
                                min_samples_leaf=config.min_samples_leaf, max_depth=config.max_depth,
                                random_state=recipe["seed"], n_jobs=args.jobs)
    model.fit(official, y)
    probe = study.official["X_train"][:16].reshape(-1, 28, 28)
    prediction = predict(model, probe, config.features)
    target = args.out / "extra_trees.joblib"
    joblib.dump({"model": model, "recipe": recipe, "labels": LABELS}, target, compress=3)
    restored = joblib.load(target)
    if not np.array_equal(prediction, predict(restored["model"], probe, config.features)):
        raise ValueError("Model predictions changed after serialization.")
    print(f"Fitted {args.mode} on {len(y):,} training images. Saved {target}.", flush=True)


if __name__ == "__main__":
    main()
