#!/usr/bin/env python3
"""Recreate the current four standard Hand-made Grayscale NPZ files.

Requires BOTH the original bundle (writers 41/43/44) AND the later corrected
writer-42 submission ZIP. The original ML-Project.zip contains the superseded
writer-42 data; it must not silently overwrite the replacement.

Examples:
  python 02_data/notebooks/convert_hand_made.py \\
    --source /path/to/ML-Project.zip \\
    --writer42-replacement '/path/to/ml26_42(2).zip' \\
    --output-dir 02_data/Hand_made
  python 02_data/notebooks/convert_hand_made.py \\
    --input-dir /path/to/original_individual_zips \\
    --writer42-replacement '/path/to/ml26_42(2).zip' \\
    --output-dir 02_data/Hand_made

Current transformations:
- 41: CCW 90-degree rotation, no inversion
- 42 NEW: no rotation, no inversion; preserve the original submitted NPZ X
- 43: no rotation, no inversion
- 44: inversion (255-X) then CCW 90-degree rotation

Other than these documented transforms, no thresholding, smoothing, resize,
augmentation, or normalization is performed. Historical EXP-유-004 scores were
computed with the previous (superseded) 42 and must be reproduced separately
from its historical Git commit, not from this current dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

LABELS = '0123456789ABCDEF'
WRITERS = (41, 42, 43, 44)
ORIGINAL_WRITERS = (41, 43, 44)
INVERT_IDS = {44}
ROTATE_IDS = {41, 44}


def expected_files(writer: int) -> list[str]:
    return [f'{ch}_{i:02d}_{writer}.jpg' for i in range(1, 41) for ch in LABELS]


def read_original_writer(archive: ZipFile, writer: int):
    assert writer in ORIGINAL_WRITERS
    names = expected_files(writer)
    files = [n for n in archive.namelist() if n.lower().endswith('.jpg')]
    by_name = {Path(n).name: n for n in files}
    if len(files) != 640 or len(by_name) != 640 or set(by_name) != set(names):
        raise ValueError(f'Writer {writer}: missing, extra or duplicate JPEGs')
    imgs = []
    for name in names:
        with Image.open(BytesIO(archive.read(by_name[name]))) as im:
            if im.size != (28, 28):
                raise ValueError(f'Writer {writer}: unexpected size in {name}: {im.size}')
            imgs.append(np.asarray(im.convert('L'), dtype=np.uint8))
    images = np.stack(imgs)
    if writer in INVERT_IDS:
        images = 255 - images
    if writer in ROTATE_IDS:
        images = np.rot90(images, k=1, axes=(1, 2)).copy()
    labels = np.array([name[0] for name in names], dtype='<U1')
    return images, labels, np.array([LABELS.index(c) for c in labels], dtype=np.uint8)


def read_replacement_42(archive: ZipFile):
    """Validate new 42's NPZ and JPEGs; use NPZ X without pixel modifications.

    The submitted JPGs have tiny lossy-encoding differences (max 2 gray levels)
    from the submitted NPZ's X, so X is the authoritative source for 42.
    """
    npzs = [n for n in archive.namelist() if Path(n).name == 'ml26_42.npz']
    if len(npzs) != 1:
        raise ValueError('Replacement ZIP must contain exactly one ml26_42.npz')
    names = expected_files(42)
    jpg_entries = [n for n in archive.namelist() if n.lower().endswith('.jpg')]
    jpg_by_name = {Path(n).name:n for n in jpg_entries}
    if len(jpg_entries) != 640 or len(jpg_by_name) != 640 or set(jpg_by_name) != set(names):
        raise ValueError('Replacement ZIP: unexpected 42 JPEG sample names/counts')
    with np.load(BytesIO(archive.read(npzs[0])), allow_pickle=False) as data:
        for k in ('X','y','images','labels','ids','label_names','filenames'):
            if k not in data:
                raise ValueError(f'Replacement 42 NPZ lacks {k}')
        img = data['X'].copy()
        y = data['y'].copy()
        if img.shape != (640,28,28) or img.dtype != np.uint8:
            raise ValueError('Replacement 42 images must be uint8[640,28,28]')
        if y.shape != (640,) or y.dtype != np.uint8:
            raise ValueError('Replacement 42 y must be uint8[640]')
        if not (np.array_equal(img, data['images']) and np.array_equal(y, data['labels'])):
            raise ValueError('Replacement 42 redundant image/label arrays disagree')
        if (data['filenames'].astype(str).tolist() != names or
            data['label_names'].astype(str).tolist() != list(LABELS) or
            not np.array_equal(data['ids'],np.repeat(np.arange(1,41,dtype=np.uint8),16))):
            raise ValueError('Replacement 42 ordering/IDs do not match standard')
    expected_y = np.tile(np.arange(16,dtype=np.uint8),40)
    if not np.array_equal(y, expected_y):
        raise ValueError('Replacement 42 labels inconsistent with filenames')
    for index, name in enumerate(names):
        with Image.open(BytesIO(archive.read(jpg_by_name[name]))) as im:
            if im.size != (28,28) or im.mode != 'L':
                raise ValueError(f'Replacement 42 is not 28x28 grayscale: {name}')
            jpg = np.asarray(im, dtype=np.uint8)
            if np.max(np.abs(jpg.astype(np.int16)-img[index].astype(np.int16))) > 2:
                raise ValueError(f'Replacement 42 JPG vs NPZ pixel mismatch >2: {name}')
    labels = np.array([LABELS[i] for i in y],dtype='<U1')
    return img, labels, y


def write_writer(images, labels, ids, writer: int, output_dir: Path):
    if images.shape != (640,28,28) or images.dtype != np.uint8:
        raise AssertionError('Image size/dtype invalid')
    if len(ids) != 640 or Counter(labels) != {c:40 for c in LABELS}:
        raise AssertionError('Labels invalid')
    path = output_dir / f'ml26_{writer}.npz'
    np.savez_compressed(path, img=images, label=labels, label_id=ids)
    with np.load(path,allow_pickle=False) as check:
        if (check.files != ['img','label','label_id'] or
            not np.array_equal(check['img'],images) or
            not np.array_equal(check['label'],labels) or
            not np.array_equal(check['label_id'],ids)):
            raise AssertionError('NPZ round-trip failed')
    digest=sha256(path.read_bytes()).hexdigest()
    print(f'writer={writer} shape={images.shape} unique={len(np.unique(images))} '
          f'inverted={writer in INVERT_IDS} rotated_ccw90={writer in ROTATE_IDS} sha256={digest}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source',type=Path,help='ML-Project.zip with original writer 41/43/44 ZIPs')
    group.add_argument('--input-dir',type=Path,help='Directory with original ml26_41.zip, ml26_43.zip, ml26_44.zip')
    parser.add_argument('--writer42-replacement',type=Path,required=True,
                        help='New corrected 42 ZIP containing the authoritative X array; required to prevent old-data overwrite')
    parser.add_argument('--output-dir',type=Path,default=Path('02_data/Hand_made'))
    args=parser.parse_args()
    if not args.writer42_replacement.is_file():
        parser.error(f'Corrected writer42 ZIP not found: {args.writer42_replacement}')
    args.output_dir.mkdir(exist_ok=True,parents=True)
    # Read & validate ALL four inputs before writing anything.
    inputs = {}
    if args.source:
        with ZipFile(args.source) as bundle:
            for writer in ORIGINAL_WRITERS:
                matches=[n for n in bundle.namelist() if n.endswith(f'/ml26_{writer}.zip')]
                if len(matches)!=1:
                    raise ValueError(f'Expected one original ZIP for {writer}')
                with ZipFile(BytesIO(bundle.read(matches[0]))) as original:
                    inputs[writer]=read_original_writer(original,writer)
    else:
        for writer in ORIGINAL_WRITERS:
            with ZipFile(args.input_dir / f'ml26_{writer}.zip') as original:
                inputs[writer]=read_original_writer(original,writer)
    with ZipFile(args.writer42_replacement) as replacement:
        inputs[42]=read_replacement_42(replacement)
    for writer in WRITERS:
        write_writer(*inputs[writer],writer,args.output_dir)


if __name__ == '__main__':
    main()