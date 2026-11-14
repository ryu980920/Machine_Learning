#!/usr/bin/env python3
"""Regenerate W02 baseline and rotation confusion-matrix PNGs from saved CSVs only.

No training, inference or edits to source CSVs. Existing Count PNGs are refreshed
and matching Row-normalized PNGs are added with the approved Viridis design.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

LABELS = tuple('0123456789ABCDEF')
BASE_MODELS = {
    'SVC (RBF)': ('SVC_RBF', 'SVC (RBF)'),
    'HistGradientBoostingClassifier': ('HistGradientBoosting', 'HistGradientBoosting'),
    'MLPClassifier': ('MLPClassifier', 'MLPClassifier'),
    'ExtraTreesClassifier': ('ExtraTrees', 'ExtraTreesClassifier'),
    'KNeighborsClassifier': ('KNeighbors', 'KNeighborsClassifier'),
}
ROT_MODELS = {
    'svc': 'SVC (RBF)', 'hgb': 'HistGradientBoosting',
    'mlp': 'MLP', 'extra_trees': 'Extra Trees', 'knn': 'KNN',
}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'font.size': 9, 'axes.labelsize': 11, 'axes.titlesize': 13,
    'axes.titleweight': 'semibold', 'xtick.labelsize': 9, 'ytick.labelsize': 9,
})


def label_index(text: str) -> int:
    return LABELS.index(str(text).strip())


def baseline_matrices(path: Path):
    matrices = {name: np.full((16, 16), -1, dtype=np.int64) for name in BASE_MODELS}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            name = row['model']
            if name not in matrices:
                raise ValueError(f'Unexpected baseline model: {name}')
            i, j = label_index(row['actual_label']), label_index(row['predicted_label'])
            if matrices[name][i, j] != -1:
                raise ValueError(f'Duplicate baseline cell: {name}, {i}, {j}')
            matrices[name][i, j] = int(row['count'])
    for name, cm in matrices.items():
        if np.any(cm < 0) or cm.shape != (16, 16) or int(cm.sum()) != 19200:
            raise ValueError(f'Invalid baseline matrix: {name}')
        if int(cm.max()) > 1400 or np.any(cm.sum(axis=1) <= 0):
            raise ValueError(f'Unexpected baseline class distribution: {name}')
    return matrices


def rotation_matrix(path: Path) -> np.ndarray:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header != ['actual\\pred', *LABELS]:
            raise ValueError(f'Unexpected rotation header: {path}: {header}')
        rows = list(reader)
    if len(rows) != 16 or [row[0] for row in rows] != list(LABELS):
        raise ValueError(f'Unexpected rotation rows: {path}')
    cm = np.array([[int(x) for x in row[1:]] for row in rows], dtype=np.int64)
    if cm.shape != (16, 16) or np.any(cm < 0) or not np.all(cm.sum(axis=1) == 160):
        raise ValueError(f'Unexpected rotation count distribution: {path}')
    return cm


def luminance(rgba) -> float:
    return float(np.dot(rgba[:3], [0.2126, 0.7152, 0.0722]))


def draw(cm: np.ndarray, title: str, kind: str, dataset: str, destination: Path):
    if kind == 'count' and dataset == 'official':
        # Keep original log(1+count) colour transform for subtle off-diagonal errors.
        shown = np.log1p(cm.astype(float))
        norm = Normalize(vmin=0, vmax=np.log1p(1400))
        ticks = np.array([0, 1, 5, 20, 100, 500, 1000, 1400], dtype=float)
        tick_positions = np.log1p(ticks)
        bar_label = 'Number of images (log colour scale)'
    elif kind == 'count':
        shown = cm.astype(float)
        norm = Normalize(vmin=0, vmax=160)
        tick_positions = [0, 40, 80, 120, 160]
        ticks = None
        bar_label = 'Number of images'
    else:
        shown = cm / cm.sum(axis=1, keepdims=True) * 100.0
        norm = Normalize(vmin=0, vmax=100)
        tick_positions = [0, 25, 50, 75, 100]
        ticks = None
        bar_label = 'Percentage of true class (%)'

    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    cmap = matplotlib.colormaps['viridis']
    im = ax.imshow(shown, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    ax.set_xticks(np.arange(16), labels=LABELS)
    ax.set_yticks(np.arange(16), labels=LABELS)
    ax.tick_params(axis='both', which='major', length=0, pad=6)
    ax.set_xlabel('Predicted label', labelpad=10)
    ax.set_ylabel('True label', labelpad=9)
    ax.set_title(title, pad=14)
    ax.set_xticks(np.arange(-.5, 16, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 16, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.38, alpha=.31)
    ax.tick_params(which='minor', bottom=False, left=False)
    for side in ('left', 'right', 'top', 'bottom'):
        ax.spines[side].set_visible(False)

    for i in range(16):
        for j in range(16):
            if cm[i, j] == 0:
                continue
            v = cm[i, j] if kind == 'count' else shown[i, j]
            value_text = str(int(v)) if kind == 'count' else f'{v:.1f}'
            fg = '#14202e' if luminance(cmap(norm(shown[i, j]))) > .53 else '#ffffff'
            ax.text(j, i, value_text, ha='center', va='center', fontsize=6.15,
                    weight='medium', color=fg)

    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.026, shrink=.89, ticks=tick_positions)
    if ticks is not None:
        cb.ax.set_yticklabels([str(int(v)) for v in ticks])
    cb.set_label(bar_label, rotation=90, labelpad=10, fontsize=9)
    cb.outline.set_visible(False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=.14)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--w02', type=Path, default=Path(__file__).resolve().parents[1],
                        help='W02 experiment directory containing results/')
    parser.add_argument('--check-only', action='store_true', help='Validate CSVs without writing images')
    args = parser.parse_args()
    results = args.w02 / 'results'
    baseline = baseline_matrices(results / 'EXP-유-003_confusion_long.csv')
    products = []
    for model, (stem, shown_name) in BASE_MODELS.items():
        cm = baseline[model]
        folder = results / 'confusion_matrices'
        for kind, suffix in (('count', ''), ('normalized', '_normalized')):
            products.append((cm, f'{shown_name} | Official Validation (n=19,200)' +
                             (' | Row-normalized (%)' if kind == 'normalized' else ''),
                             kind, 'official', folder / f'{stem}_confusion_matrix{suffix}.png'))
    folder = results / 'EXP-유-003_rotation_v4'
    for model, shown_name in ROT_MODELS.items():
        for stage in ('before', 'after'):
            cm = rotation_matrix(folder / f'confusion_matrix_{model}_{stage}.csv')
            for kind, suffix in (('count', ''), ('normalized', '_normalized')):
                products.append((cm, f'{shown_name} | Rotation {stage} | Hand-made (n=2,560)' +
                                 (' | Row-normalized (%)' if kind == 'normalized' else ''),
                                 kind, 'handmade',
                                 folder / f'confusion_matrix_{model}_{stage}{suffix}.png'))
    if len(products) != 30:
        raise AssertionError(f'Expected 30 outputs, got {len(products)}')
    print('Validated original CSVs; 15 Count + 15 row-normalized PNG outputs', flush=True)
    if args.check_only:
        return
    for index, (matrix, title, kind, dataset, dest) in enumerate(products, 1):
        draw(matrix, title, kind, dataset, dest)
        print(f'{index:02d}/30 {dest}', flush=True)


if __name__ == '__main__':
    main()
