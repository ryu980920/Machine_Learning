#!/usr/bin/env python3
"""Reuse exact Stage 1 scoring/gating for EXP004 continued candidates."""
from pathlib import Path
import argparse
import exp_a004_aggregate as original
original.CONDITIONS = [
    "C5_Stroke045", "C6_Stroke060", "C7_MedianWidth035",
    "C8_Stroke030_MedianWidth035", "C9_BBoxLongside20"
]
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    original.main(p.parse_args().root)
