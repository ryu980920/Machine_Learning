#!/usr/bin/env python3
"""EXP-CMB-001: regenerate only approved four Viridis OOF confusion matrices.

Run in repository root:
 python 05_model_optimize/MLP/Combined/scripts/render_cm_viridis.py
or:
 python 05_model_optimize/MLP/Combined/scripts/render_cm_viridis.py \
     --predictions path/to/OOF_predictions.csv --output-dir path/to/figures

No model training. No edits to the source OOF_predictions.csv.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
LABELS=tuple('0123456789ABCDEF')
N_CLASSES=16
ROW_COUNT=160
COLOR_SCALE='viridis'
COLOR_MIN=0
COUNT_COLOR_MAX=160
NORM_COLOR_MAX=100
plt.rcParams.update({
 'font.family':'DejaVu Sans', 'figure.facecolor':'white', 'axes.facecolor':'white',
 'savefig.facecolor':'white', 'font.size':9,'axes.labelsize':11,
 'axes.titlesize':13,'axes.titleweight':'semibold',
 'xtick.labelsize':9,'ytick.labelsize':9,
})
def luminance(rgb):
    r, g, b = rgb[:3]
    return 0.2126*r + 0.7152*g + 0.0722*b


def draw_heatmap(ax, cm, kind, title, show_ylabel=True):
    if kind == 'count':
        matrix = cm.astype(float)
        maximum, textfmt = COUNT_COLOR_MAX, lambda v: str(int(v))
    else:
        matrix = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        maximum, textfmt = NORM_COLOR_MAX, lambda v: f'{v:.1f}'
    norm = Normalize(vmin=COLOR_MIN, vmax=maximum)
    im = ax.imshow(matrix, cmap=COLOR_SCALE, norm=norm, interpolation='nearest', aspect='equal')
    ax.set_xticks(np.arange(N_CLASSES), labels=LABELS)
    ax.set_yticks(np.arange(N_CLASSES), labels=LABELS)
    ax.tick_params(axis='both', which='major', length=0, pad=6)
    ax.set_xlabel('Predicted label', labelpad=10)
    ax.set_ylabel('True label' if show_ylabel else '', labelpad=9)
    ax.set_title(title, pad=14)

    # A subtle square grid improves locating the row/column while keeping the original colors.
    ax.set_xticks(np.arange(-.5, N_CLASSES, 1), minor=True)
    ax.set_yticks(np.arange(-.5, N_CLASSES, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=.38, alpha=.31)
    ax.tick_params(which='minor', bottom=False, left=False)
    for side in ('left', 'right', 'top', 'bottom'):
        ax.spines[side].set_visible(False)

    cmap = matplotlib.colormaps[COLOR_SCALE]
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if cm[i, j] == 0:
                continue  # Important: suppress only zero values, never true counts.
            value = matrix[i, j]
            rgba = cmap(norm(value))
            fg = '#14202e' if luminance(rgba) > .53 else '#ffffff'
            ax.text(j, i, textfmt(value), ha='center', va='center', fontsize=6.15,
                    weight='medium', color=fg)
    return im


def add_colorbar(fig, im, ax, kind, horizontal=False):
    kw = {'ax': ax, 'fraction': .045, 'pad': .026, 'shrink': .89}
    cb = fig.colorbar(im, **kw)
    cb.set_label('Number of images' if kind == 'count' else 'Percentage of true class (%)',
                 rotation=90, labelpad=10, fontsize=9)
    cb.outline.set_visible(False)
    return cb


def save(fig, stem: Path, *, dpi=300):
    fig.savefig(stem.with_suffix('.png'), dpi=dpi, facecolor='white', bbox_inches='tight', pad_inches=.14)
    plt.close(fig)




def load_cm_from_predictions(path:Path,model:str)->np.ndarray:
    with path.open(encoding='utf-8-sig',newline='') as f:
        data=list(csv.DictReader(f))
    if len(data)!=2560:raise ValueError(f'Expected 2560 predictions, got {len(data)}')
    col={'Normal':'normal_pred','Combined':'combined_pred'}[model]
    true=np.array([LABELS.index(r['true_label']) for r in data],dtype=np.int64)
    pred=np.array([LABELS.index(r[col]) for r in data],dtype=np.int64)
    cm=np.zeros((16,16),dtype=np.int64)
    np.add.at(cm,(true,pred),1)
    if not np.array_equal(cm.sum(axis=1),np.full(16,ROW_COUNT)):
        raise ValueError('Expected 160 true samples per class')
    return cm


def main():
    here=Path(__file__).resolve().parents[1]
    default=here/'results'/'EXP_CMB_001'
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',type=Path,default=default/'OOF_predictions.csv')
    ap.add_argument('--output-dir',type=Path,default=default/'figures')
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for model in ('Normal','Combined'):
        cm=load_cm_from_predictions(args.predictions,model)
        for kind in ('count','normalized'):
            fig,ax=plt.subplots(figsize=(10,9),constrained_layout=True)
            title=f'MLP | {model} | Hand-made OOF'+(' | Row-normalized (%)' if kind=='normalized' else '')
            im=draw_heatmap(ax,cm,kind,title)
            add_colorbar(fig,im,ax,kind)
            save(fig,args.output_dir/f'{model.lower()}_{kind}_viridis',dpi=300)
    print('Rendered exactly four Viridis CM PNG files, without retraining.')

if __name__=='__main__':main()