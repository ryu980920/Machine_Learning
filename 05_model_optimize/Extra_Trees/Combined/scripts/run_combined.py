"""Run the shared Extra Trees experiment runner in Combined mode."""
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.argv[1:1] = ["--mode", "combined"]
    runner = Path(__file__).resolve().parents[2] / "Normal/scripts/run_extra_trees.py"
    runpy.run_path(str(runner), run_name="__main__")
