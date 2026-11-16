#!/usr/bin/env python3
"""Reuse EXP004 original pre-registered accuracy gate for size-strength sweeps."""
import argparse
from pathlib import Path
import exp_a004_aggregate as original
original.CONDITIONS=["C10_Size020","C11_Size035","C12_Size050","C13_Size065","C14_Size080"]
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True,type=Path)
    original.main(p.parse_args().root)
