#!/usr/bin/env python3
"""EXP-유-003 appendix 8: compare Rotation before/after on current four writers.

Only writer 41 and 44 are rotation-corrected; writer 42 (replacement
Grayscale) and 43 are unchanged. Existing trained EXP-유-003 model joblibs are
loaded without refitting; output contains five-model totals, per-writer
comparisons, 16x16 confusion matrices and row-level paired predictions.

From repository root:
    python -m pip install numpy scikit-learn joblib matplotlib
    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python 03_experiments/W02_2026-11-09_11-15/scripts/experiment3_handmade/run_rotation_comparison.py \
      --model-zip '/path/to/W02_Experiment2_5models_complete(1).zip' \
      --handmade-dir 02_data/Hand_made \
      --out 03_experiments/W02_2026-11-09_11-15/results/EXP-유-003_rotation_v4

Requires a trustworthy model archive (joblib deserialization executes code).
Never uses the official Test split and never retrains a model. The script
performs a fresh evaluation; the published 2026-11-09 review originally
reused verified old predictions for unchanged writers, as documented in
rotation_review_provenance.json. Wall-clock runtimes may be long for SVC.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

WRITERS = (41, 42, 43, 44)
LABELS = "0123456789ABCDEF"
MODEL_NAMES = (('svc', 'SVC (RBF)'), ('hgb', 'HistGradientBoosting'),
               ('mlp', 'MLP'), ('extra_trees', 'Extra Trees'), ('knn', 'KNN'))
EXPECTED_SHA256 = {
    41: '0a829172a72787c770139c73de34cc8b751181829ff6fe08ec260247b9be5a2f',
    42: 'f364fb453d05b4fe2a705c2b8a80683b404608c7c257d664e3ec7ac86932d4c5',
    43: '10db20997c727c1bc1f1299f8e89626fadd3ec9618468d6759c86b0d2ddc4a66',
    44: '0e266a36e7b29edfcca0514836b5c383d13939948b4e335d70f61bb40bdc084d',
}


def sha256(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_data(folder: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for writer in WRITERS:
        p = folder / f'ml26_{writer}.npz'
        if sha256(p) != EXPECTED_SHA256[writer]:
            raise ValueError(f'Unexpected dataset SHA256: {p}; compare with EXP-유-005')
        with np.load(p, allow_pickle=False) as d:
            if d.files != ['img', 'label', 'label_id']:
                raise ValueError(f'Wrong NPZ keys in {p}: {d.files}')
            img, labels, ids = (d[k] for k in ('img', 'label', 'label_id'))
        if img.shape != (640, 28, 28) or img.dtype != np.uint8:
            raise ValueError(f'Bad image shape/dtype {p}')
        if ids.shape != (640,) or ids.dtype != np.uint8:
            raise ValueError(f'Bad labels {p}')
        if not np.array_equal(ids, np.array([LABELS.index(c) for c in labels], dtype=np.uint8)):
            raise ValueError(f'Label ID mismatch {p}')
        if Counter(labels) != Counter({c: 40 for c in LABELS}):
            raise ValueError(f'Class count mismatch {p}')
        if writer == 42 and np.unique(img).size != 256:
            raise ValueError('Writer 42 must use new 256-level grayscale original')
        out[writer] = (img, ids.astype(np.int64))
    return out


def data_for_stage(writer: int, img: np.ndarray, stage: str) -> np.ndarray:
    if stage == 'before' and writer in (41, 44):
        # Standard NPZ is already CCW-rotated. Inverse it to reconstruct the
        # original pre-correction orientation without interpolation.
        img = np.rot90(img, k=-1, axes=(1, 2))
    return img.reshape(640, 784).astype(np.float32)


def csv_write(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def confusion_png(path: Path, cm: np.ndarray, name: str, stage: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.4, 9.2), dpi=135)
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=160)
    ax.set_xticks(range(16), list(LABELS))
    ax.set_yticks(range(16), list(LABELS))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{name} — Rotation {stage} (n=2,560)')
    for i in range(16):
        for j in range(16):
            v = int(cm[i, j])
            if v:
                ax.text(j, i, str(v), ha='center', va='center', fontsize=7,
                        color='white' if v >= 60 else 'black')
    fig.colorbar(im, ax=ax, fraction=.045)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_saved_models(archive: Path, target: Path) -> dict[str, Path]:
    # Only extract required trained models and their scaler class definition.
    with ZipFile(archive) as z:
        for entry in z.infolist():
            if not entry.filename.endswith(('.joblib', 'pixel_transform.py')):
                continue
            dest = (target / entry.filename).resolve()
            if not dest.is_relative_to(target):
                raise ValueError(f'Unsafe path in ZIP: {entry.filename}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(entry) as source, dest.open('wb') as output:
                shutil.copyfileobj(source, output)
    scaler = list(target.rglob('pixel_transform.py'))
    if len(scaler) != 1:
        raise ValueError('Model archive must include exactly one pixel_transform.py')
    sys.path.insert(0, str(scaler[0].parent))
    import pixel_transform  # noqa: F401, required for custom saved Pipeline loading
    result = {}
    for key, _ in MODEL_NAMES:
        files = list(target.rglob(f'{key}.joblib'))
        if len(files) != 1:
            raise ValueError(f'Missing or duplicate model: {key}')
        result[key] = files[0]
    return result


def run(args):
    data = load_data(args.handmade_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    by_writer = []
    all_preds = {'before': [], 'after': []}
    per_class = []
    confusion_rows = []
    with tempfile.TemporaryDirectory(prefix='exp003_rotation_') as d:
        files = load_saved_models(args.model_zip, Path(d).resolve())
        for key, name in MODEL_NAMES:
            model = joblib.load(files[key])
            stage_y = {}
            stage_pred = {}
            for stage in ('before', 'after'):
                yp, yt = [], []
                for writer in WRITERS:
                    img, labels = data[writer]
                    # Same image/labels for both stages for writer 42 and 43.
                    if stage == 'after' and writer in (42, 43):
                        # The same model, same X; avoid duplicate expensive predict.
                        this_pred = stage_pred['before'][writer]
                    else:
                        x = data_for_stage(writer, img, stage)
                        this_pred = np.asarray(model.predict(x), dtype=np.int64)
                    if this_pred.shape != (640,):
                        raise ValueError(f'Bad prediction shape {name}, {writer}, {stage}')
                    yp.append(this_pred)
                    yt.append(labels)
                    by_writer.append({
                        'model': name, 'stage': stage, 'writer': writer,
                        'n': 640, 'correct': int((this_pred == labels).sum()),
                        'accuracy': float(accuracy_score(labels, this_pred)),
                        'macro_f1': float(f1_score(labels, this_pred, labels=list(range(16)),
                                                   average='macro', zero_division=0)),
                    })
                    for idx, (true, pred) in enumerate(zip(labels, this_pred)):
                        all_preds[stage].append({'model': name, 'key': key, 'writer': writer,
                            'image_idx_in_writer': idx, 'actual': LABELS[int(true)],
                            'predicted': LABELS[int(pred)], 'correct': int(true == pred)})
                stage_y[stage] = np.concatenate(yt)
                stage_pred[stage] = {w: yp[i] for i, w in enumerate(WRITERS)}
                cm = confusion_matrix(stage_y[stage], np.concatenate(yp), labels=list(range(16)))
                if cm.sum() != 2560 or not np.all(cm.sum(axis=1) == 160):
                    raise AssertionError(f'Confusion Matrix support mismatch {key}, {stage}')
                csv_rows = [{'actual\\pred': LABELS[i], **{LABELS[j]: int(cm[i,j]) for j in range(16)}}
                            for i in range(16)]
                csv_write(args.out / f'confusion_matrix_{key}_{stage}.csv', csv_rows)
                confusion_png(args.out / f'confusion_matrix_{key}_{stage}.png', cm, name, stage)
                for i in range(16):
                    for j in range(16):
                        confusion_rows.append({'model': name, 'stage': stage,
                                               'actual': LABELS[i], 'predicted': LABELS[j],
                                               'count': int(cm[i,j])})
                pred = np.concatenate(yp)
                accuracy = float(accuracy_score(stage_y[stage], pred))
                mf1 = float(f1_score(stage_y[stage], pred, labels=list(range(16)),
                                       average='macro', zero_division=0))
                summary.append({'model': name, 'stage': stage, 'n': 2560,
                                'correct': int((stage_y[stage] == pred).sum()),
                                'accuracy': accuracy, 'macro_f1': mf1})
                for i, c in enumerate(LABELS):
                    tr = stage_y[stage] == i
                    pr = pred == i
                    tp = int((tr & pr).sum())
                    precision = tp / int(pr.sum()) if pr.sum() else 0.
                    recall = tp / int(tr.sum())
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
                    per_class.append({'model': name, 'stage': stage, 'class': c,
                                      'support': int(tr.sum()), 'correct': tp,
                                      'precision': precision, 'recall': recall, 'f1': f1})
            print(f'Finished {name}', flush=True)
    by_model = {}
    for r in summary:
        by_model.setdefault(r['model'], {})[r['stage']] = r
    comparison = []
    for model, stages in by_model.items():
        b, a = stages['before'], stages['after']
        comparison.append({'model': model, 'n': 2560, 'before_correct': b['correct'],
            'after_correct': a['correct'], 'before_accuracy': b['accuracy'],
            'after_accuracy': a['accuracy'],
            'change_percentage_points': (a['accuracy'] - b['accuracy']) * 100,
            'before_macro_f1': b['macro_f1'], 'after_macro_f1': a['macro_f1']})
    writer_comp = []
    for model, _ in [(n,n) for _,n in MODEL_NAMES]:
        for w in WRITERS:
            b = next(x for x in by_writer if x['model'] == model and x['writer'] == w and x['stage'] == 'before')
            a = next(x for x in by_writer if x['model'] == model and x['writer'] == w and x['stage'] == 'after')
            writer_comp.append({'model': model, 'writer': w,
                'before_correct': b['correct'], 'after_correct': a['correct'],
                'before_accuracy': b['accuracy'], 'after_accuracy': a['accuracy'],
                'delta_percentage_points': (a['accuracy'] - b['accuracy'])*100,
                'before_macro_f1': b['macro_f1'], 'after_macro_f1': a['macro_f1']})
    csv_write(args.out / 'rotation_before_after_5models.csv', comparison)
    csv_write(args.out / 'rotation_before_after_by_writer.csv', writer_comp)
    csv_write(args.out / 'rotation_before_predictions.csv', all_preds['before'])
    csv_write(args.out / 'rotation_after_predictions.csv', all_preds['after'])
    csv_write(args.out / 'rotation_confusion_long.csv', confusion_rows)
    csv_write(args.out / 'rotation_per_class.csv', per_class)
    (args.out / 'rotation_reproduction_metadata.json').write_text(json.dumps({
        'data_sha256': EXPECTED_SHA256, 'model_archive_sha256': sha256(args.model_zip),
        'rotation_corrected_writers': [41,44], 'unchanged_writers': [42,43],
        'used_existing_models': True, 'models_retrained': False,
        'comparison': comparison}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved', args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-zip', type=Path, required=True)
    p.add_argument('--handmade-dir', type=Path, default=Path('02_data/Hand_made'))
    p.add_argument('--out', type=Path, default=Path('EXP003_rotation_results'))
    run(p.parse_args())

if __name__ == '__main__':
    main()
