#!/usr/bin/env python3
"""EXP유002 Normal MLP input sensitivity diagnosis (stage 3 ONLY).

Repeat unchanged Normal MLP baseline, validate predictions, then apply ONE
controlled image change at a time to all 19,200 official Validation and 2,560
current Handmade images. Official Test is neither loaded nor scored.
No fitting on Handmade; no model tuning; does not commit results to GitHub.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.ndimage import maximum_filter, minimum_filter, affine_transform
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from Pixcel_Attribution import (
    read_handmade, prior_official_cm, prior_handmade, official_indices,
    fixed_split, pipeline, build_baseline, LABELS, DEFAULT_DATA, diverse_selection,
)

@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    value: object

VARIANTS = [Variant("original", "original", None)]
VARIANTS += [
    Variant(f"shift_{direction}{pixels}", "shift", (dx, dy))
    for pixels in (1, 2)
    for direction, dx, dy in (("L", -pixels, 0), ("R", pixels, 0),
                              ("U", 0, -pixels), ("D", 0, pixels))
]
VARIANTS += [Variant(f"brightness_{int(factor*100):03d}", "brightness", factor)
             for factor in (0.80, 0.90, 1.10, 1.20)]
VARIANTS += [Variant(f"stroke_{direction}_{int(frac*100):02d}", "stroke", (direction, frac))
             for direction in ("thin", "thick") for frac in (0.25, 0.40)]
VARIANTS += [Variant(f"scale_{int(factor*100):03d}", "scale", factor)
             for factor in (0.90, 0.95, 1.05, 1.10)]

PAIR_LIST = [(a, b) for a, b in [
    ("5","F"), ("4","F"), ("7","F"), ("9","F"),
    ("B","8"), ("8","B"), ("0","D"), ("D","0"),
    ("9","4"), ("4","9"), ("E","C"), ("3","5"), ("7","2"),
]]
REF_DIR = ROOT / "05_model_optimize" / "MLP" / "Normal" / "results" / "Pixcel_Attribution"


def transform(batch: np.ndarray, condition: Variant):
    a = np.asarray(batch, dtype=np.float32).reshape(-1, 28, 28)
    group = condition.family
    if group == "original":
        z = a.copy()
    elif group == "shift":
        dx, dy = condition.value
        z = np.zeros_like(a)
        ys = slice(max(0, -dy), min(28, 28-dy))
        yd = slice(max(0, dy), min(28, 28+dy))
        xs = slice(max(0, -dx), min(28, 28-dx))
        xd = slice(max(0, dx), min(28, 28+dx))
        z[:, yd, xd] = a[:, ys, xs]
    elif group == "brightness":
        z = np.clip(a * float(condition.value), 0, 255)
    elif group == "stroke":
        direction, fraction = condition.value
        if direction == "thick":
            modified = maximum_filter(a, size=(1, 3, 3), mode="constant", cval=0)
        else:
            modified = minimum_filter(a, size=(1, 3, 3), mode="constant", cval=0)
        z = (1-fraction)*a + fraction*modified
    elif group == "scale":
        inv = 1.0 / float(condition.value)
        offset = 13.5 * (1.0 - inv)
        z = affine_transform(a, matrix=np.diag([1, inv, inv]),
                             offset=(0.0, offset, offset),
                             output_shape=a.shape, order=1, mode="constant",
                             cval=0.0, prefilter=False)
    else:
        raise ValueError(f"Unknown transformation: {condition}")
    return np.rint(np.clip(z, 0, 255)).astype(np.float32).reshape(len(a), 784)


def test_transform():
    x = np.zeros((2, 784), dtype=np.float32)
    x[0, 13*28+14] = 255
    x[1, 5*28+4] = 180
    assert len(VARIANTS) == 21 and len({v.name for v in VARIANTS}) == 21
    for v in VARIANTS:
        out = transform(x, v)
        assert out.shape == x.shape and out.dtype == np.float32
        assert out.min() >= 0 and out.max() <= 255
    # Spatial shift must never wrap border values.
    x2 = np.zeros((1, 784), dtype=np.float32)
    x2[0, 0] = 255
    assert not transform(x2, Variant("L1", "shift", (-1, 0))).any()
    assert np.array_equal(transform(x, VARIANTS[0]), x)
    print("TRANSFORM SELF-TEST PASSED (21 original/perturbed conditions)", flush=True)


def get_output(estimator, raw):
    """Return class predictions, softmax probabilities, true logit margin input.

    Internal one-hidden-layer ReLU forward pass; checked against sklearn
    Pipeline.predict_proba on original input before any sensitivity analysis.
    """
    net = estimator.named_steps["model"]
    if len(net.coefs_) != 2 or net.activation != "relu":
        raise RuntimeError("Baseline MLP architecture changed")
    x01 = np.asarray(raw, dtype=np.float32) / np.float32(255.0)
    hidden = np.maximum(x01 @ net.coefs_[0] + net.intercepts_[0], 0)
    logits = hidden @ net.coefs_[1] + net.intercepts_[1]
    z = logits.astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    probability = np.exp(z)
    probability /= probability.sum(axis=1, keepdims=True)
    return logits, probability


def margin(logits, y):
    ids = np.arange(len(y))
    true_score = logits[ids, y].astype(np.float64)
    competing = logits.copy()
    competing[ids, y] = -np.inf
    return true_score - competing.max(axis=1)


def record_summary(domain, variant, y, oldpred, pred, pold, pnew,
                   marginold, marginnew, mask, group_name, group_value,
                   clipping=None, mass_delta=None):
    ix = np.flatnonzero(mask)
    yy, old, new = y[ix], oldpred[ix], pred[ix]
    correct_before = (old == yy)
    correct_after = (new == yy)
    orig_correct = int(correct_before.sum())
    orig_wrong = len(ix) - orig_correct
    corrections = int((~correct_before & correct_after).sum())
    induced = int((correct_before & ~correct_after).sum())
    flips = int((old != new).sum())
    avgclip = float(np.mean(clipping[ix])) if clipping is not None and len(ix) else 0.
    avgmass = float(np.mean(mass_delta[ix])) if mass_delta is not None and len(ix) else 0.
    return {
        "domain":domain, "variant":variant.name, "family":variant.family,
        "group_type":group_name, "group":group_value, "n":len(ix),
        "original_correct":orig_correct, "original_wrong":orig_wrong,
        "original_accuracy":orig_correct/len(ix), "accuracy":int(correct_after.sum())/len(ix),
        "accuracy_delta_pp":100*(int(correct_after.sum())-orig_correct)/len(ix),
        "flip_count":flips, "flip_rate":flips/len(ix),
        "corrections":corrections,
        "correction_rate_among_original_errors":corrections/orig_wrong if orig_wrong else None,
        "induced_errors":induced,
        "induction_rate_among_original_correct":induced/orig_correct if orig_correct else None,
        "other_wrong_to_wrong_changes":int((~correct_before & ~correct_after & (old != new)).sum()),
        "mean_true_probability_delta":float((pnew[ix, yy] - pold[ix, yy]).mean()),
        "mean_true_logit_margin_delta":float((marginnew[ix]-marginold[ix]).mean()),
        "mean_fraction_brightness_clipped":avgclip,
        "mean_relative_ink_mass_delta":avgmass,
    }


def write_dict_rows(writer, domain, condition, y, writers, indices,
                    oldpred, pred, pold, pnew, marginold, marginnew,
                    clip, mass_delta):
    n = len(y)
    for i in range(n):
        yi = int(y[i])
        writer.writerow({
            "domain": domain, "writer": int(writers[i]), "sample_index": int(indices[i]),
            "true": LABELS[yi], "variant": condition.name, "family": condition.family,
            "original_prediction": LABELS[int(oldpred[i])],
            "transformed_prediction": LABELS[int(pred[i])],
            "original_correct": int(oldpred[i] == yi),
            "transformed_correct": int(pred[i] == yi),
            "original_p_true":float(pold[i, yi]),
            "transformed_p_true":float(pnew[i, yi]),
            "original_margin":float(marginold[i]),
            "transformed_margin":float(marginnew[i]),
            "delta_margin":float(marginnew[i]-marginold[i]),
            "brightness_clipped_pixel_fraction":float(clip[i]),
            "relative_ink_mass_change":float(mass_delta[i]),
        })


def pair_records(domain, condition, y, p0, p1):
    out = []
    for a, b in PAIR_LIST:
        ai, bi = LABELS.index(a), LABELS.index(b)
        old_pair = (y == ai) & (p0 == bi)
        new_pair = (y == ai) & (p1 == bi)
        count = int(old_pair.sum())
        corrected = int((old_pair & (p1 == ai)).sum())
        unchanged = int((old_pair & (p1 == bi)).sum())
        to_other = int((old_pair & (p1 != bi) & (p1 != ai)).sum())
        out.append({
            "domain":domain, "variant":condition.name, "family":condition.family,
            "actual":a, "previous_error":b, "original_pair_errors":count,
            "new_pair_errors":int(new_pair.sum()), "pair_count_delta":int(new_pair.sum())-count,
            "original_pair_corrected":corrected,
            "pair_correction_rate":corrected/count if count else None,
            "original_pair_still_same_error":unchanged,
            "original_pair_changed_to_other_error":to_other,
        })
    return out


def csv_out(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        dw = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        dw.writeheader()
        dw.writerows(rows)


def select_focus(xva, yva, pva, xv_indices, xhm, yhm, phm, whm, ih):
    path = REF_DIR / "selected_cases.csv"
    focus = []
    official_map = {int(v): i for i, v in enumerate(xv_indices)}
    handmade_map = {(int(w), int(i)): k for k,(w,i) in enumerate(zip(whm, ih))}
    used = set()
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["domain"] == "OfficialVal":
                k = official_map[int(r["index"])]
                domain = "OfficialVal"
                actual = yva[k]
                pred = pva[k]
                identifier = ("OfficialVal", int(r["index"]))
                original = xva[k]
                writer = -1
            elif r["domain"] == "Handmade":
                writer, idx = int(r["writer"]), int(r["index"])
                k = handmade_map[(writer, idx)]
                domain = "Handmade"
                actual = yhm[k]
                pred = phm[k]
                identifier = ("Handmade", writer, idx)
                original = xhm[k]
            else:
                raise RuntimeError(f"Unexpected prior selection domain {r['domain']}")
            if LABELS[int(actual)] != r["true"] or LABELS[int(pred)] != r["predicted"]:
                raise RuntimeError(f"Stage 1 selected image no longer has original prediction {identifier}")
            if identifier in used:
                raise RuntimeError(f"Duplicate Stage 1 focus image {identifier}")
            focus.append({"domain":domain, "writer":writer,
                          "index":int(r["index"]), "true":int(actual),
                          "original_pred":int(pred), "pixels":original.copy(),
                          "case":r["case"], "origin":"stage1_2"})
            used.add(identifier)
    if len(focus) != 98:
        raise RuntimeError(f"Expected 98 Stage 1-2 cases, received {len(focus)}")
    for a, b in (("9", "F"), ("7", "F")):
        matches = np.flatnonzero((yhm==LABELS.index(a)) & (phm==LABELS.index(b)))
        extras = diverse_selection(matches, whm, 4, True)
        if len(extras) != 4:
            raise RuntimeError(f"Only {len(extras)} valid samples for {a}->{b}")
        for k in extras:
            w, idx = int(whm[k]), int(ih[k])
            identifier = ("Handmade", w, idx)
            if identifier in used:
                raise RuntimeError(f"Duplicate extra sample {identifier}")
            focus.append({"domain":"Handmade", "writer":w,"index":idx,
                          "true":int(yhm[k]),"original_pred":int(phm[k]),
                          "pixels":xhm[k].copy(),"case":f"{a}_to_{b}","origin":"stage3_extra"})
            used.add(identifier)
    if len(focus) != 106:
        raise RuntimeError("Expected exactly 106 focus images")
    return focus


def analyse_focus(focus, net, out):
    cols = []
    variant_names = {v.name:v for v in VARIANTS}
    for ex in focus:
        original = ex["pixels"].reshape(1,784)
        for variant in VARIANTS:
            changed = transform(original, variant)
            z, prob = get_output(net, changed)
            p = int(np.argmax(z, axis=1)[0])
            y = int(ex["true"])
            rival = z[0].copy()
            rival[y] = -np.inf
            score_margin = float(z[0,y] - rival.max())
            cols.append({
                "domain":ex["domain"], "writer":ex["writer"],"index":ex["index"],
                "true":LABELS[y],"original_prediction":LABELS[ex["original_pred"]],
                "case":ex["case"], "origin":ex["origin"],
                "variant":variant.name, "family":variant.family,
                "new_prediction":LABELS[p], "new_correct":int(y==p),
                "p_true":float(prob[0,y]),
                "p_original_prediction":float(prob[0,ex["original_pred"]]),
                "true_logit_margin":score_margin,
            })
    csv_out(out/"stage3_focus_106.csv", cols)
    chosen = []
    for a,b in (("5","F"),("B","8"),("9","4"),("0","D"),("7","F")):
        ch = next((f for f in focus if f["domain"]=="Handmade" and
                   f["true"]==LABELS.index(a) and f["original_pred"]==LABELS.index(b)), None)
        if ch:
            chosen.append(ch)
    display_conditions = ("original", "shift_R2", "brightness_080", "stroke_thick_40", "scale_090")
    fig, axs = plt.subplots(len(chosen),len(display_conditions),
                            figsize=(12.6,2.65*len(chosen)), squeeze=False)
    for i, ex in enumerate(chosen):
        for j,name in enumerate(display_conditions):
            var=variant_names[name]
            x=transform(ex["pixels"].reshape(1,784),var)
            logits,probs=get_output(net,x)
            pred=int(logits[0].argmax())
            axs[i,j].imshow(x.reshape(28,28), cmap="gray",vmin=0,vmax=255)
            axs[i,j].set_title(f"{name}\n{LABELS[ex['true']]}->{LABELS[pred]} p(true)={probs[0,ex['true']]:.3f}",fontsize=8)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
    fig.suptitle("Controlled one-variable perturbations (illustrative; not model optimization)",fontsize=12)
    fig.tight_layout()
    fig.savefig(out/"stage3_example_grid.png",dpi=145)
    plt.close(fig)
    fig, axs=plt.subplots(3,1,figsize=(13.3,9.2),squeeze=False)
    for i, ex in enumerate(chosen[:3]):
        ax=axs[i,0]
        rows=[r for r in cols if r["domain"]==ex["domain"] and
              r["writer"]==ex["writer"] and r["index"]==ex["index"]]
        ax.plot(range(len(rows)),[r["p_true"] for r in rows],marker="o",label="P(actual)")
        ax.plot(range(len(rows)),[r["p_original_prediction"] for r in rows],
                marker="x",label="P(original prediction)")
        ax.set_xticks(range(len(rows)),[r["variant"] for r in rows],rotation=65,ha="right",fontsize=7)
        ax.set_ylim(-0.05,1.05)
        ax.set_title(f"Writer {ex['writer']} image #{ex['index']}: "
                     f"{LABELS[ex['true']]} -> {LABELS[ex['original_pred']]}")
        ax.set_ylabel("Predicted probability")
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out/"stage3_probability_curves.png",dpi=150)
    plt.close(fig)
    return cols


def run(args):
    start = time.perf_counter()
    test_transform()
    if args.batch_size < 64:
        raise ValueError("batch-size must be at least 64")
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    xtr,ytr,xva,yva,offsha = fixed_split(DEFAULT_DATA)
    va_indices = official_indices(DEFAULT_DATA,yva)
    xhm,yhm,whm,ih = read_handmade()
    model_def,scaled = build_baseline()
    if not scaled or model_def.get_params()["random_state"] != 42:
        raise RuntimeError("Unexpected baseline hyperparameter change")
    model=pipeline(model_def,scaled)
    print("Re-fitting Normal MLP from official train only",flush=True)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        model.fit(xtr,ytr)
    pva=model.predict(xva).astype(int)
    phm=model.predict(xhm).astype(int)
    cm_l1=int(np.abs(confusion_matrix(yva,pva,labels=np.arange(16))
                   -prior_official_cm()).sum())
    hand_hist=prior_handmade()
    diffs=sum(hand_hist[(int(w),int(i))] != (LABELS[int(a)],LABELS[int(p)])
              for w,i,a,p in zip(whm,ih,yhm,phm))
    if cm_l1 != 0 or diffs != 0:
        raise RuntimeError(f"Historical baseline mismatch: official_cm_L1={cm_l1}, handmade_diff={diffs}")
    if len(yva)!=19200 or len(yhm)!=2560:
        raise RuntimeError("Expected source sizes changed")
    print(f"BASELINE CONFIRMED: official={accuracy_score(yva,pva):.8f}, "
          f"handmade={accuracy_score(yhm,phm):.8f}, cm_L1={cm_l1}, hand_pred_diff={diffs}",flush=True)
    # Drop training arrays to reduce peak runner RAM.
    del xtr,ytr
    focus = select_focus(xva,yva,pva,va_indices,xhm,yhm,phm,whm,ih)
    focus_rows = analyse_focus(focus, model, out)
    focus_ids=len(focus)

    domains=[
      ("OfficialVal",xva,yva,np.full(len(yva),-1,dtype=np.int32),va_indices,pva),
      ("Handmade",xhm,yhm,whm,ih,phm)
    ]
    summaries, per_class, per_writer, pairs = [],[],[],[]
    fields=["domain","writer","sample_index","true","variant","family",
            "original_prediction","transformed_prediction","original_correct",
            "transformed_correct","original_p_true","transformed_p_true",
            "original_margin","transformed_margin","delta_margin",
            "brightness_clipped_pixel_fraction","relative_ink_mass_change"]
    total_rows=0
    with gzip.open(out/"stage3_predictions.csv.gz","wt",newline="",encoding="utf-8") as gz:
        writer=csv.DictWriter(gz,fieldnames=fields)
        writer.writeheader()
        for domain, x, y, authors, srcidx, oldpred in domains:
            base_logits, base_probs=get_output(model,x)
            pfromlogits=np.argmax(base_logits,axis=1)
            if not np.array_equal(pfromlogits,oldpred):
                raise RuntimeError(f"Internal forward pass is inconsistent for {domain}")
            test_proba=model.predict_proba(x[:25])
            if not np.allclose(base_probs[:25],test_proba,rtol=2e-5,atol=2e-6):
                raise RuntimeError("Internal softmax outputs deviate from Pipeline.predict_proba")
            m0=margin(base_logits,y)
            p0=base_probs
            for j, variant in enumerate(VARIANTS):
                n=len(y)
                preds=np.empty(n,dtype=np.int64)
                pn=np.empty((n,16),dtype=np.float64)
                mn=np.empty(n,dtype=np.float64)
                clip=np.zeros(n,dtype=np.float64)
                mass_delta=np.zeros(n,dtype=np.float64)
                for left in range(0,n,args.batch_size):
                    right=min(n,left+args.batch_size)
                    xb=x[left:right]
                    xt=transform(xb,variant)
                    logits,proba=get_output(model,xt)
                    pred=logits.argmax(axis=1)
                    preds[left:right]=pred
                    pn[left:right]=proba
                    mn[left:right]=margin(logits,y[left:right])
                    orig_sum=xb.sum(axis=1)
                    trans_sum=xt.sum(axis=1)
                    mass_delta[left:right]=(trans_sum-orig_sum)/np.maximum(orig_sum,1.0)
                    if variant.family=="brightness":
                        frac=float(variant.value)
                        clip[left:right]=((xb*frac)>255).sum(axis=1)/784.0
                if variant.family=="original" and (
                    not np.array_equal(preds,oldpred) or
                    not np.allclose(pn,p0,rtol=1e-7,atol=1e-7)):
                    raise RuntimeError(f"Identity condition altered baseline: {domain}")
                write_dict_rows(writer,domain,variant,y,authors,srcidx,
                                oldpred,preds,p0,pn,m0,mn,clip,mass_delta)
                total_rows+=n
                allmask=np.ones(n,dtype=bool)
                summaries.append(record_summary(domain,variant,y,oldpred,preds,p0,pn,
                                               m0,mn,allmask,"all","all",clip,mass_delta))
                for li, label in enumerate(LABELS):
                    per_class.append(record_summary(domain,variant,y,oldpred,preds,p0,pn,
                                                     m0,mn,y==li,"class",label,clip,mass_delta))
                if domain=="Handmade":
                    for w in (41,42,43,44):
                        per_writer.append(record_summary(domain,variant,y,oldpred,preds,p0,pn,
                                                        m0,mn,authors==w,"writer",str(w),clip,mass_delta))
                pairs.extend(pair_records(domain,variant,y,oldpred,preds))
                print(f"{domain:11s} {j+1:02d}/21 {variant.name:20s}: "
                      f"acc={summaries[-1]['accuracy']:.4f}, flip={summaries[-1]['flip_rate']:.4f}, "
                      f"corrected={summaries[-1]['corrections']}, induced={summaries[-1]['induced_errors']}",
                      flush=True)
    if total_rows != (19200+2560)*21:
        raise RuntimeError(f"Unexpected prediction total {total_rows}")
    csv_out(out/"stage3_sensitivity_summary.csv",summaries)
    csv_out(out/"stage3_per_class.csv",per_class)
    csv_out(out/"stage3_per_writer.csv",per_writer)
    csv_out(out/"stage3_error_pairs.csv",pairs)

    metadata={
        "scope":"Normal MLP Stage 3 input-only sensitivity experiment",
        "source_code":"05_model_optimize/MLP/Normal/scripts/Input_Sensitivity.py",
        "official_test_read":False,
        "handmade_training":False,
        "model_hyperparameters_modified":False,
        "data_sha256":offsha,
        "official_n":19200,"handmade_n":2560,
        "n_variants_including_original":len(VARIANTS),
        "n_nontrivial_variants":len(VARIANTS)-1,
        "total_prediction_rows":total_rows,
        "original_accuracy":{"OfficialVal":float(accuracy_score(yva,pva)),
                            "Handmade":float(accuracy_score(yhm,phm))},
        "baseline_validation_cm_L1_difference":cm_l1,
        "baseline_handmade_prediction_difference":diffs,
        "focus_n":focus_ids, "focus_rows":len(focus_rows),
        "transforms":[{"name":v.name,"family":v.family,"value":v.value} for v in VARIANTS],
        "floating_precision":"mixed float32 fitted MLP forward, float64 softmax",
        "brightness_clip_fraction_basis":"all 784 image pixels",
        "ink_mass_change_basis":"relative sum of 0-255 pixel intensity",
        "warnings":[f"{w.category.__name__}: {w.message}" for w in ws],
        "elapsed_seconds":time.perf_counter()-start,
        "versions":{"python":platform.python_version(),"sklearn":sklearn.__version__,
                    "numpy":np.__version__,"scipy":scipy.__version__,
                    "matplotlib":matplotlib.__version__},
    }
    (out/"stage3_run_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    full=[r for r in summaries if r["group_type"]=="all" and r["variant"]!="original"]
    lines=[
      "# Normal MLP Input_Sensitivity — Stage 3 result",
      "",
      f"- Source: {metadata['official_n']:,} official Validation + {metadata['handmade_n']:,} Handmade.",
      f"- Original accuracy: official {metadata['original_accuracy']['OfficialVal']:.6%}; Handmade {metadata['original_accuracy']['Handmade']:.6%}.",
      f"- Historical reproduction: official CM L1={cm_l1}; Handmade prediction discrepancies={diffs}.",
      f"- Controlled modifications: 8 shift, 4 brightness, 4 stroke, 4 scale = 20, plus original.",
      f"- Total scored rows: {total_rows:,}; focus examples: {focus_ids} (98 previous + 8 extra).",
      "- No official Test; no Handmade fitting, refit only once on Normal official Training.",
      "",
      "## Highest observed prediction flip conditions (each dataset)",
      "",
      "| Dataset | Transform | Flip rate | Accuracy delta(pp) | Corrected wrong | Newly wrong |",
      "|---|---|---:|---:|---:|---:|",
    ]
    for domain in ("OfficialVal","Handmade"):
        group=sorted((r for r in full if r["domain"]==domain),
                     key=lambda t:t["flip_rate"],reverse=True)
        for r in group[:5]:
            lines.append(
                f"| {domain} | {r['variant']} | {r['flip_rate']:.2%} | "
                f"{r['accuracy_delta_pp']:+.2f} | {r['corrections']} | {r['induced_errors']} |"
            )
    lines.extend([
       "",
       "## How to read these results",
       "",
       "- Flip rate: original predicted class differs from transformed prediction.",
       "- Correction rate: among originally incorrect inputs, fraction now correct.",
       "- Induction rate: among originally correct inputs, fraction now incorrect.",
       "- Accuracy delta: corrected count minus newly wrong count divided by all examples.",
       "- A high flip rate is sensitivity, not automatically an improvement.",
       "- Brightness clipping is measured explicitly. Thickness/scale transforms can alter discriminative strokes.",
       "- The comparison cannot by itself prove the model learned semantics rather than pixel correlations.",
       "- This is a diagnostic, not a selected preprocessing policy or a final performance gain.",
       "",
       "## Output index",
       "",
       "- stage3_predictions.csv.gz: ALL sample-level predictions (gzip-compressed CSV).",
       "- stage3_sensitivity_summary.csv: 21 conditions x 2 datasets.",
       "- stage3_per_class.csv: per-class metrics.",
       "- stage3_per_writer.csv: writer-level metrics.",
       "- stage3_error_pairs.csv: selected confusion-pair changes.",
       "- stage3_focus_106.csv: 106 original/focus images x 21 conditions.",
       "- stage3_example_grid.png and stage3_probability_curves.png: focused before/after views.",
       "- stage3_run_metadata.json: environment, conditions, reproducibility checks.",
    ])
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"STAGE3 COMPLETED: {total_rows} predictions, {len(full)} variant summaries; "
          f"output={out}",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,default=Path("mlp_stage3_results"))
    ap.add_argument("--batch-size",type=int,default=1024)
    args=ap.parse_args()
    run(args)


if __name__=="__main__":
    main()
