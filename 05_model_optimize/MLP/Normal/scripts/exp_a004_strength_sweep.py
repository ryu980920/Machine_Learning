#!/usr/bin/env python3
"""Hypothesis A EXP004 Stage3: strength sweep of training-only glyph-size normalization."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
import exp_a004_official_mlp as base
from exp_a004_followup import Stage2Preprocessing, SizeToTrainingMedian
STRENGTHS = {"C10_Size020":.20,"C11_Size035":.35,"C12_Size050":.50,
             "C13_Size065":.65,"C14_Size080":.80}

class SizeSweepPreprocessing(Stage2Preprocessing):
    def __init__(self,policy):
        if policy not in STRENGTHS:
            raise ValueError(policy)
        self.policy=policy
        self.stroke=None
        self.shape=None
    def fit(self,X):
        self.n_fit=len(X)
        self.shape=SizeToTrainingMedian(kind="longside",
                                         strength=STRENGTHS[self.policy],
                                         preserve_topology=True).fit(X)
        return self

def self_test():
    x=np.zeros((40,28,28),np.float32)
    for i in range(40):
        cv2.line(x[i],(4+i%8,4),(4+i%8,22),255,1+i%5)
    for condition in STRENGTHS:
        p=SizeSweepPreprocessing(condition).fit(x)
        z=p.transform(x)
        assert z.shape==(40,784) and np.isfinite(z).all()
        assert np.array_equal(z,p.transform(x))
    print("Stage3 self-test passed: deterministic train-only geometry",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",type=Path)
    p.add_argument("--out-dir",type=Path)
    p.add_argument("--condition",choices=tuple(STRENGTHS))
    p.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    p.add_argument("--max-iter",type=int,default=200)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        self_test()
    else:
        if a.repo_root is None or a.out_dir is None or a.condition is None:
            p.error("repo-root, out-dir, condition required")
        base.Preprocessing=SizeSweepPreprocessing
        base.run(a.repo_root,a.out_dir,a.condition,a.seeds,a.max_iter)
