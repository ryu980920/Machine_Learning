#!/usr/bin/env python3
"""EXP유004 continued experiment: train-fitted morphology and size variants.

Same official Train, same 3 seeds, same MLP fit and pass criteria as Stage 1.
No writer/label information in preprocessing. No official or professor Test.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
from stroke_normalization import StrokeWidthNormalizer, grayscale, topology, center_of_mass
from width_normalization import bounds
import exp_a004_official_mlp as base

CONDITIONS = ("C5_Stroke045", "C6_Stroke060", "C7_MedianWidth035",
              "C8_Stroke030_MedianWidth035", "C9_BBoxLongside20")

def bbox_warp(img, scale_x, scale_y):
    mask = bounds(img, 64)
    if mask is None:
        return img.copy()
    cx, cy = (mask[0] + mask[1])/2, (mask[2] + mask[3])/2
    matrix = np.array([[scale_x, 0, (1-scale_x)*cx],
                       [0, scale_y, (1-scale_y)*cy]], np.float32)
    return cv2.warpAffine(img, matrix, (28,28), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

class SizeToTrainingMedian:
    def __init__(self, kind, strength=.35, threshold=64, preserve_topology=True):
        self.kind = kind
        self.strength = strength
        self.threshold = threshold
        self.preserve_topology = preserve_topology

    def _measured(self, im):
        b = bounds(im, self.threshold)
        if b is None:
            return None
        w, h = b[1] - b[0] + 1, b[3] - b[2] + 1
        return w if self.kind == "width" else max(w, h)

    def fit(self, x):
        vals = [self._measured(im) for im in grayscale(x)]
        vals = np.array([v for v in vals if v is not None], float)
        if len(vals) < 20:
            raise RuntimeError("Insufficient official training images")
        self.median_ = float(np.median(vals))
        self.q20_, self.q80_ = map(float, np.quantile(vals, [.2, .8]))
        self.fit_count_ = int(len(vals))
        return self

    def transform(self, x):
        if not hasattr(self, "median_"):
            raise RuntimeError("Fit reference on Training first")
        a = grayscale(x)
        out = np.empty_like(a)
        for i, im in enumerate(a):
            current = self._measured(im)
            updated = im.copy()
            if current and self.strength > 0:
                desired = 1.0 + self.strength * (self.median_ / current - 1)
                desired = float(np.clip(desired, .75, 1.25))
                if abs(desired - 1.0) > 1e-4:
                    before = topology(im)
                    for _ in range(5):
                        sx, sy = (desired, 1.0) if self.kind == "width" else (desired, desired)
                        candidate = bbox_warp(im, sx, sy)
                        if not self.preserve_topology or topology(candidate) == before:
                            updated = candidate
                            break
                        desired = 1 + (desired-1)/2
            out[i] = updated
        return out.reshape(len(a),784)

class Stage2Preprocessing:
    def __init__(self, policy):
        if policy not in CONDITIONS:
            raise ValueError(policy)
        self.policy = policy
        self.stroke = None
        self.shape = None

    def fit(self, X):
        self.n_fit = len(X)
        if self.policy in ("C5_Stroke045", "C6_Stroke060", "C8_Stroke030_MedianWidth035"):
            strength = .45 if self.policy == "C5_Stroke045" else .60 if self.policy == "C6_Stroke060" else .30
            self.stroke = StrokeWidthNormalizer(strength=strength,
                                                 preserve_topology=True,
                                                 apply_center=False).fit(X)
        if self.policy in ("C7_MedianWidth035", "C8_Stroke030_MedianWidth035", "C9_BBoxLongside20"):
            kind = "longside" if self.policy == "C9_BBoxLongside20" else "width"
            strength = 1.0 if kind == "longside" else .35
            train = self.stroke.transform(X) if self.stroke else X
            self.shape = SizeToTrainingMedian(kind=kind, strength=strength,
                                              preserve_topology=True).fit(train)
        return self

    def transform(self, X):
        x = grayscale(X)
        if self.stroke:
            x = self.stroke.transform(x).reshape(-1,28,28)
        if self.shape:
            x = self.shape.transform(x).reshape(-1,28,28)
        result = np.empty((len(x),784),np.float32)
        for i,im in enumerate(x):
            result[i] = center_of_mass(im).reshape(-1)
        return result

    def description(self):
        info = {"condition":self.policy,"training_fit_count":self.n_fit,
                "fit_data":"OfficialTrainOnly","uses_labels":False,
                "uses_writer":False,"com_last":True,"topology_guard":True}
        if self.stroke:
            t=self.stroke
            info["stroke"]={"strength":t.strength,"median":float(t.median_),
                            "q20":float(t.q_low_),"q80":float(t.q_high_)}
        if self.shape:
            t=self.shape
            info["shape"]={"kind":t.kind,"strength":t.strength,
                           "median":t.median_,"q20":t.q20_,"q80":t.q80_}
        return info

def test_preprocessors():
    x=np.zeros((40,28,28),np.float32)
    for i in range(40):
        cv2.line(x[i],(4+i%8,4),(4+i%8,22),255,1+i%5)
    for condition in CONDITIONS:
        p=Stage2Preprocessing(condition).fit(x)
        t=p.transform(x)
        assert t.shape==(40,784) and np.isfinite(t).all()
        assert 0 <= t.min() <= t.max() <= 255
        assert np.array_equal(t,p.transform(x))
    print("STAGE2 SYNTHETIC SELF TEST PASSED; no real training claimed",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo-root",type=Path)
    parser.add_argument("--out-dir",type=Path)
    parser.add_argument("--condition",choices=CONDITIONS)
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    parser.add_argument("--max-iter",type=int,default=200)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        test_preprocessors()
    else:
        if args.repo_root is None or args.out_dir is None or args.condition is None:
            parser.error("repo-root, out-dir, condition required")
        # The original runner performs the identical official SHA/split check and
        # identical MLP training, metrics, model export, and label-blind validation.
        base.Preprocessing = Stage2Preprocessing
        base.run(args.repo_root,args.out_dir,args.condition,args.seeds,args.max_iter)
