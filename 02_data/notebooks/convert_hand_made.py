#!/usr/bin/env python3
"""Convert ml26_41..44.zip to the three-array format of ml26_43.npz.

Input:  four private ZIPs containing 640 named 28x28 JPEG images each.
Output:  02_data/Hand_made/ml26_{41,42,43,44}.npz; test-only datasets.

Install: pip install numpy pillow
Usage: python 02_data/notebooks/convert_hand_made.py --input-dir /path/to/zips

Reference 43 NPZ is copied unchanged after a complete sample-by-sample check.
41 and 43 JPEGs have dark backgrounds; 42 and 44 are white-background
black glyphs, so 42/44 are inverted to match the reference foreground=1.
"""
from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

LABELS = "0123456789ABCDEF"


def convert(archive: Path, destination: Path, dataset_id: int) -> None:
    names = [f"{label}_{n:02d}_{dataset_id}.jpg" for n in range(1, 41) for label in LABELS]
    with ZipFile(archive) as z:
        jpgs = {Path(name).name: name for name in z.namelist() if name.lower().endswith(".jpg")}
        if len(jpgs) != 640 or set(jpgs) != set(names):
            raise ValueError(f"Unexpected or missing JPG filenames in {archive.name}")
        arrays = []
        for basename in names:
            with Image.open(BytesIO(z.read(jpgs[basename]))) as im:
                if im.size != (28, 28):
                    raise ValueError(f"Unexpected image size: {basename}: {im.size}")
                arrays.append(np.asarray(im.convert("L")))
        gray = np.asarray(arrays, dtype=np.uint8)
        imgs = (gray > 127).astype(np.uint8)
        if dataset_id in (42, 44):
            imgs = 1 - imgs  # normalize white-background source to dark background
        label = np.array([name[0] for name in names], dtype="<U1")
        label_id = np.array([LABELS.index(name[0]) for name in names], dtype=np.uint8)
        if imgs.shape != (640, 28, 28) or Counter(label) != {c: 40 for c in LABELS}:
            raise ValueError(f"Invalid image/label structure in {archive.name}")
        if dataset_id == 43:
            reference = [name for name in z.namelist() if name.endswith(".npz")]
            if len(reference) != 1:
                raise ValueError("Exactly one reference NPZ expected in 43")
            original = z.read(reference[0])
            with np.load(BytesIO(original), allow_pickle=False) as d:
                if d.files != ["img", "label", "label_id"]:
                    raise ValueError("Reference NPZ keys differ")
                for key, expected in (("img", imgs), ("label", label), ("label_id", label_id)):
                    if not np.array_equal(d[key], expected):
                        raise ValueError(f"Reference {key} does not match 43 JPEGs")
            destination.write_bytes(original)
        else:
            np.savez_compressed(destination, img=imgs, label=label, label_id=label_id)
    with np.load(destination, allow_pickle=False) as d:
        assert d.files == ["img", "label", "label_id"]
        assert d["img"].shape == (640, 28, 28) and d["img"].dtype == np.uint8
        assert set(np.unique(d["img"]).tolist()) <= {0, 1}
        assert d["label"].dtype == np.dtype("<U1")
        assert d["label_id"].dtype == np.dtype("uint8")
        assert np.array_equal(d["label"], label)
        assert np.array_equal(d["label_id"], label_id)
    print(f"Verified {destination}: 640 images / 16 classes / exactly 3 keys")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with ml26_41.zip .. ml26_44.zip")
    parser.add_argument("--output-dir", type=Path, default=Path("02_data/Hand_made"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in (41, 42, 43, 44):
        convert(args.input_dir / f"ml26_{i}.zip", args.output_dir / f"ml26_{i}.npz", i)


if __name__ == "__main__":
    main()