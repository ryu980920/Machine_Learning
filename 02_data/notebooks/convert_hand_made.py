#!/usr/bin/env python3
"""Rebuild four Grayscale (not binary) Hand-made NPZs from original JPGs.

Inputs (choose exactly one):
  --source ML-Project.zip      : a bundle that contains ml26_41.zip .. ml26_44.zip
  --input-dir /path/to/zips    : four individual ZIPs in a local directory

Outputs:
  02_data/Hand_made/ml26_{41,42,43,44}.npz
    img      uint8 (640, 28, 28), native JPEG grayscale, dark background
    label    <U1   (640,), characters 0-9 and A-F
    label_id uint8 (640,), integer IDs 0..15

The original JPEGs are 28x28. The code does NOT threshold, resize, augment,
or normalize pixels. Writers 42 and 44 have light backgrounds and are
inverted as 255 - gray. Existing binary NPZs are not used as sources.

Usage:
  pip install numpy pillow
  python 02_data/notebooks/convert_hand_made.py --source /path/to/ML-Project.zip
  python 02_data/notebooks/convert_hand_made.py --input-dir /path/to/zips
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

LABELS = "0123456789ABCDEF"
IDS = (41, 42, 43, 44)
INVERT_IDS = {42, 44}


def read_writer(archive: ZipFile, writer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = [f"{ch}_{i:02d}_{writer}.jpg" for i in range(1, 41) for ch in LABELS]
    jpgs = {Path(n).name: n for n in archive.namelist() if n.lower().endswith(".jpg")}
    if len(jpgs) != 640 or set(jpgs) != set(expected):
        raise ValueError(f"Missing, additional or duplicate JPG filenames for writer {writer}")
    pixels = []
    for name in expected:
        with Image.open(BytesIO(archive.read(jpgs[name]))) as im:
            if im.size != (28, 28):
                raise ValueError(f"Unexpected 28x28 size: {name} = {im.size}")
            pixels.append(np.asarray(im.convert("L"), dtype=np.uint8))
    images = np.stack(pixels, axis=0)
    if writer in INVERT_IDS:
        images = 255 - images
    labels = np.array([name.split("_")[0] for name in expected], dtype="<U1")
    ids = np.array([LABELS.index(ch) for ch in labels], dtype=np.uint8)
    if images.shape != (640, 28, 28) or images.dtype != np.uint8:
        raise AssertionError("image array invariant failed")
    if Counter(labels) != {ch: 40 for ch in LABELS}:
        raise AssertionError("class histogram invariant failed")
    return images, labels, ids


def write_npz(archive: ZipFile, writer: int, output_dir: Path) -> Path:
    img, label, label_id = read_writer(archive, writer)
    output_path = output_dir / f"ml26_{writer}.npz"
    np.savez_compressed(output_path, img=img, label=label, label_id=label_id)
    with np.load(output_path, allow_pickle=False) as result:
        if list(result.files) != ["img", "label", "label_id"]:
            raise AssertionError("Unexpected NPZ keys")
        if not (np.array_equal(result["img"], img)
                and np.array_equal(result["label"], label)
                and np.array_equal(result["label_id"], label_id)):
            raise AssertionError("NPZ round-trip verification failed")
    digest = sha256(output_path.read_bytes()).hexdigest()
    print(f"ml26_{writer}: samples=640 classes=16 shape={img.shape} "
          f"gray_levels={len(np.unique(img))} inverted={writer in INVERT_IDS} "
          f"sha256={digest}")
    return output_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sources = ap.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source", type=Path, help="Outer ML-Project.zip containing four writer ZIPs")
    sources.add_argument("--input-dir", type=Path, help="Directory containing ml26_41.zip .. ml26_44.zip")
    ap.add_argument("--output-dir", type=Path, default=Path("02_data/Hand_made"))
    args = ap.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.source:
        with ZipFile(args.source) as bundle:
            for writer in IDS:
                matches = [n for n in bundle.namelist() if n.endswith(f"/ml26_{writer}.zip")]
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one ml26_{writer}.zip in outer ZIP")
                with ZipFile(BytesIO(bundle.read(matches[0]))) as archive:
                    write_npz(archive, writer, args.output_dir)
    else:
        for writer in IDS:
            with ZipFile(args.input_dir / f"ml26_{writer}.zip") as archive:
                write_npz(archive, writer, args.output_dir)


if __name__ == "__main__":
    main()