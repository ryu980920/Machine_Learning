#!/usr/bin/env python3
"""Recreate Stage 3 descriptive plots from the actual result CSVs (no retraining)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = "0123456789ABCDEF"


def rows(file: Path):
    with file.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    d = args.result_dir.resolve()
    overview = rows(d / "stage3_sensitivity_summary.csv")
    perclass = rows(d / "stage3_per_class.csv")
    writer = rows(d / "stage3_per_writer.csv")
    variants = [x["variant"] for x in overview if x["domain"] == "Handmade" and x["variant"] != "original"]
    assert len(variants) == 20 and len(set(variants)) == 20

    # Comparison over official and Handmade development data.
    data = {(r["domain"], r["variant"]): r for r in overview}
    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.bar(x - 0.21, [float(data["OfficialVal", v]["accuracy_delta_pp"]) for v in variants],
           width=0.42, label="Official Validation (19,200)")
    ax.bar(x + 0.21, [float(data["Handmade", v]["accuracy_delta_pp"]) for v in variants],
           width=0.42, label="Hand-made (2,560)")
    ax.axhline(0, linewidth=0.8, color="gray")
    ax.set_xticks(x, variants, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy change against unchanged input (percentage points)")
    ax.set_title("Normal MLP: One-variable Input Sensitivity (20 perturbations)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(d / "STAGE3_Accuracy_Change_Comparison.png", dpi=155)
    plt.close(fig)

    # Class-balanced digits 0-9, letters A-F on Handmade only.
    bykey = {(r["domain"], r["variant"], r["group"]): r for r in perclass}
    digit, letter = [], []
    for v in variants:
        vals = []
        for classes in (LABELS[:10], LABELS[10:]):
            current = [bykey["Handmade", v, c] for c in classes]
            original = [bykey["Handmade", "original", c] for c in classes]
            n = sum(int(row["n"]) for row in current)
            ncorrect = sum(round(float(row["accuracy"]) * int(row["n"])) for row in current)
            ncorrect_orig = sum(round(float(row["accuracy"]) * int(row["n"])) for row in original)
            vals.append(100 * (ncorrect - ncorrect_orig) / n)
        digit.append(vals[0]); letter.append(vals[1])
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.bar(x - 0.21, digit, width=0.42, label="Digits (0–9)")
    ax.bar(x + 0.21, letter, width=0.42, label="Letters (A–F)")
    ax.axhline(0, linewidth=0.8, color="gray")
    ax.set_xticks(x, variants, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Hand-made accuracy change (percentage points)")
    ax.set_title("Normal MLP: Opposing Effects Across Digits and Letters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(d / "STAGE3_Digit_Letter_Tradeoff.png", dpi=155)
    plt.close(fig)

    # Writer differences on variants with clear overall effects.
    selection = ["shift_L1", "shift_R1", "shift_D1", "scale_090", "scale_095",
                 "stroke_thin_25", "stroke_thick_40"]
    bywriter = {(r["variant"], r["group"]): r for r in writer}
    xx = np.arange(len(selection))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for w in ("41", "42", "43", "44"):
        values = [float(bywriter[(v, w)]["accuracy_delta_pp"]) for v in selection]
        ax.plot(xx, values, marker="o", label=f"Writer {w}")
    ax.set_xticks(xx, selection, rotation=30, ha="right")
    ax.axhline(0, linewidth=0.8, color="gray")
    ax.set_ylabel("Accuracy change (percentage points)")
    ax.set_title("Normal MLP: Hand-made Input Sensitivity by Writer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(d / "STAGE3_Per_Writer.png", dpi=155)
    plt.close(fig)

    print("Created 3 descriptive plots from CSV only", flush=True)


if __name__ == "__main__":
    main()
