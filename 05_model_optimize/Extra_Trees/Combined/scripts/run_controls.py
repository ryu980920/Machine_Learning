"""Run paired Combined Extra Trees development controls on the archived folds."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Normal" / "scripts"))
from run_extra_trees import Config, Study

SEEDS = (42, 17, 73)
ARMS = {
    "stroke00": {"prep": "size"},
    "features08": {"max_features": 0.08},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    args.mode = "combined"
    args.out.mkdir(parents=True, exist_ok=True)
    selection = Path(__file__).resolve().parents[1] / "results" / "selection.json"
    candidate = Config(**json.loads(selection.read_text())["config"])
    config = replace(candidate, **ARMS[args.arm])
    study = Study(args)
    study.audit()
    for seed in SEEDS:
        study.run(f"C{seed}_{args.arm}", config, seed=seed)


if __name__ == "__main__":
    main()
