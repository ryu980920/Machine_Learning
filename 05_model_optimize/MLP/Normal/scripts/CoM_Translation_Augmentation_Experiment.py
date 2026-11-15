#!/usr/bin/env python3
"""EXP유005: CoM-centered MLP with controlled 1px translation augmentation.

Experiment A: the verified EXP유004 CoM MLP (frozen, not retrained).
Experiment B: same 76,800 official train images after CoM centering, repeated
              5x (384,000 identical images) to control sample/optimizer exposure.
Experiment C: the same CoM centered images each appearing five times:
              [original, left 1px, right 1px, up 1px, down 1px].
Evaluation: CoM centering ONLY on official Validation and Handmade; never
augmentation on Validation/Handmade and never reads any official Test split.
No class-dependent preprocessing and no Handmade fitting. Output REVIEW ONLY;
do not update repo or auto-select candidate without an explicit user decision.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))

from Pixcel_Attribution import (
    fixed_split, DEFAULT_DATA, read_handmade, LABELS, build_baseline,
)
from Automatic_Centering_Transformer import (
    AutomaticCentering, center_batch, test_centering, THRESHOLD, TARGET,
)
from Input_Sensitivity import transform, Variant
from pixel_transform import PixelScaler255

DIR = ROOT / "05_model_optimize" / "MLP" / "Normal"
REFERENCE_MODEL = DIR / "results" / "Automatic_Centering" / "exp004_mass.joblib"
REFERENCE_PREDICTIONS = DIR / "results" / "Automatic_Centering" / "exp004_predictions.csv.gz"
REFERENCE_MODEL_SHA = "3c1aac362d75ff5c4d9652188c1c4fd97524525f5409ec00d5a6c56fa3ad27ef"
REFERENCE_DATA_SHA = "0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879"
CONDITIONS = ("EXP004_CoM", "EXP005_repeat5_control", "EXP005_shift1_augmented")
DIRECTIONS = (
    ("original", 0, 0),
    ("left1", -1, 0),
    ("right1", 1, 0),
    ("up1", 0, -1),
    ("down1", 0, 1),
)
PAIR_ERRORS = (("5", "F"), ("4", "F"), ("B", "8"), ("8", "B"),
               ("0", "D"), ("D", "0"), ("9", "4"), ("E", "C"))


def checksum(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError("No rows for " + str(path))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def shift_into(source: np.ndarray, target: np.ndarray, dx: int, dy: int) -> None:
    """Move pre-centered pixel arrays without wrapping, interpolation or recentering."""
    assert source.shape == target.shape and source.ndim == 3 and source.shape[1:] == (28, 28)
    assert abs(dx) <= 1 and abs(dy) <= 1
    assert not np.shares_memory(source, target)
    target.fill(0)
    sy0, sy1 = max(0, -dy), min(28, 28 - dy)
    sx0, sx1 = max(0, -dx), min(28, 28 - dx)
    if sy0 < sy1 and sx0 < sx1:
        target[:, sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = source[:, sy0:sy1, sx0:sx1]


def test_augmentation():
    test_centering()
    x = np.zeros((3, 28, 28), dtype=np.float32)
    x[0, 14, 14], x[1, 0, 0], x[2, 27, 27] = 255, 190, 110
    for label, dx, dy in DIRECTIONS:
        actual = np.empty_like(x)
        shift_into(x, actual, dx, dy)
        expected = transform(x.reshape(3, 784),
                             Variant(label, "original", None) if label == "original" else
                             Variant(label, "shift", (dx, dy)))
        assert np.array_equal(actual.reshape(3, 784), expected), label
        assert np.all(x[[0, 1, 2], [14, 0, 27], [14, 0, 27]] == [255, 190, 110])
    print("EXP005 SELF-TEST PASSED: ±1 px zero-filled translations exactly match EXP유002", flush=True)


def build_training(centered: np.ndarray, kind: str) -> tuple[np.ndarray, dict]:
    """Construct 5 blocks in fixed order for both conditions; convert to /255 in-place."""
    n = len(centered)
    assert n == 76800 and centered.shape == (n, 784)
    raw = np.asarray(centered, dtype=np.float32).reshape(n, 28, 28)
    result = np.empty((5 * n, 784), dtype=np.float32)
    mass_before = raw.sum(axis=(1, 2), dtype=np.float64)
    clip = {}
    for j, (name, dx, dy) in enumerate(DIRECTIONS):
        out = result[j*n:(j+1)*n].reshape(n, 28, 28)
        if kind == "repeat5":
            out[:] = raw
        else:
            assert kind == "shift1"
            shift_into(raw, out, dx, dy)
        mass_after = out.sum(axis=(1, 2), dtype=np.float64)
        lost = np.maximum(0.0, mass_before - mass_after)
        clip[name] = {
            "images_with_ink_loss": int(np.count_nonzero(lost > 1e-2)),
            "mean_relative_ink_loss": float(np.mean(lost / np.maximum(mass_before, 1))),
        }
    if kind == "repeat5":
        for j in range(1, 5):
            assert np.array_equal(result[0:n], result[j*n:(j+1)*n])
    else:
        assert np.array_equal(result[:n], centered), "CoM must precede translation"
        # One-pixel augmentation should actually change some original training images.
        assert any(not np.array_equal(result[:n], result[j*n:(j+1)*n]) for j in range(1, 5))
    result /= np.float32(255.0)
    assert result.dtype == np.float32 and np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1
    return result, clip


def metric_row(dataset, condition, group_type, group, y, p, reference):
    n = len(y)
    good = p == y
    old = reference == y
    return {
        "dataset": dataset, "condition": condition, "group_type": group_type,
        "group": str(group), "n": n, "correct": int(good.sum()),
        "accuracy": float(good.mean()),
        "macro_f1_present_classes": float(f1_score(y, p, average="macro")),
        "corrected_vs_EXP004_CoM": int((~old & good).sum()),
        "induced_vs_EXP004_CoM": int((old & ~good).sum()),
        "changed_predictions_vs_EXP004_CoM": int((p != reference).sum()),
        "delta_accuracy_pp_vs_EXP004_CoM": float((good.mean() - old.mean()) * 100),
    }


def group_rows(dataset, y, writer, all_preds):
    records = []
    ref = all_preds["EXP004_CoM"]
    for condition, pred in all_preds.items():
        groups = [("all", "all", np.ones(len(y), dtype=bool))]
        if dataset == "Handmade":
            groups.extend([("category", "digits", y < 10),
                           ("category", "letters", y >= 10)])
            for w in (41, 42, 43, 44):
                groups.append(("writer", w, writer == w))
                groups.append(("writer_category", f"{w}_digits", (writer == w) & (y < 10)))
                groups.append(("writer_category", f"{w}_letters", (writer == w) & (y >= 10)))
        for cls in range(16):
            groups.append(("class", LABELS[cls], y == cls))
        for typ, name, mask in groups:
            records.append(metric_row(dataset, condition, typ, name,
                                      y[mask], pred[mask], ref[mask]))
    return records


def paired_transition(group, y, pred_before, pred_after, condition_a, condition_b):
    ca = pred_before == y
    cb = pred_after == y
    return {
        "group": group,
        "from": condition_a, "to": condition_b, "n": len(y),
        "accuracy_before": float(ca.mean()),
        "accuracy_after": float(cb.mean()),
        "delta_accuracy_pp": float((cb.mean() - ca.mean()) * 100),
        "corrected": int((~ca & cb).sum()),
        "new_errors": int((ca & ~cb).sum()),
        "changed_predictions": int((pred_before != pred_after).sum()),
    }


def compare_transitions(y, writer, preds):
    rows = []
    groups = [("all", np.ones(len(y), dtype=bool)),
              ("digits", y < 10), ("letters", y >= 10)]
    for w in (41, 42, 43, 44):
        groups.append((f"writer_{w}", writer == w))
    for a, b in (
        ("EXP004_CoM", "EXP005_repeat5_control"),
        ("EXP004_CoM", "EXP005_shift1_augmented"),
        ("EXP005_repeat5_control", "EXP005_shift1_augmented"),
    ):
        for name,mask in groups:
            rows.append(paired_transition(name, y[mask], preds[a][mask],
                                          preds[b][mask], a, b))
    return rows


def verify_exp004(yva, yhm, whm, ihm, predv, predh):
    assert checksum(REFERENCE_MODEL) == REFERENCE_MODEL_SHA
    assert (predv == yva).sum() == 18608
    assert (predh == yhm).sum() == 1974
    exact = {"OfficialVal": [], "Handmade": {}}
    with gzip.open(REFERENCE_PREDICTIONS, "rt", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == "OfficialVal":
                exact["OfficialVal"].append((row["true"], row["Center_of_Mass_pred"]))
            elif row["dataset"] == "Handmade":
                exact["Handmade"][(int(row["writer"]), int(row["image_index"]))] = (
                    row["true"], row["Center_of_Mass_pred"])
            else:
                raise ValueError("Unexpected dataset in published EXP004")
    assert len(exact["OfficialVal"]) == 19200
    assert len(exact["Handmade"]) == 2560
    for i, (true, pred) in enumerate(exact["OfficialVal"]):
        assert true == LABELS[int(yva[i])] and pred == LABELS[int(predv[i])]
    for i,(w,ix) in enumerate(zip(whm,ihm)):
        true,pred=exact["Handmade"][(int(w),int(ix))]
        assert true==LABELS[int(yhm[i])] and pred==LABELS[int(predh[i])]
    print("EXP004 REPRODUCED: Official Val 18608/19200; Handmade 1974/2560; all 21760 predictions exact",flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    test_augmentation()
    xtr, ytr, xva, yva, input_sha = fixed_split(DEFAULT_DATA)
    xhm, yhm, whm, ihm = read_handmade()
    assert (len(xtr),len(xva),len(xhm)) == (76800,19200,2560)
    assert input_sha == REFERENCE_DATA_SHA
    assert sorted(np.unique(ytr)) == list(range(16))
    assert len(DIRECTIONS) == 5
    print("DATA VERIFIED: 76800 Official Train / 19200 Official Val / 2560 Handmade. No Test read.",flush=True)

    xtr_com = center_batch(xtr,"mass")
    # One consistent inference preprocessing, NO evaluation augmentation.
    xva_com01 = center_batch(xva,"mass") / np.float32(255)
    xhm_com01 = center_batch(xhm,"mass") / np.float32(255)
    # The original EXP004 centering Transformer remains in the final pipeline.
    original_model = joblib.load(REFERENCE_MODEL)
    assert isinstance(original_model.named_steps["centering"], AutomaticCentering)
    assert original_model.named_steps["centering"].method == "mass"
    estimator_params = build_baseline()[0].get_params()
    assert original_model.named_steps["model"].get_params() == estimator_params
    pred = {
        "OfficialVal": {"EXP004_CoM": original_model.predict(xva)},
        "Handmade": {"EXP004_CoM": original_model.predict(xhm)},
    }
    assert np.array_equal(pred["OfficialVal"]["EXP004_CoM"],
                          original_model.named_steps["model"].predict(xva_com01))
    assert np.array_equal(pred["Handmade"]["EXP004_CoM"],
                          original_model.named_steps["model"].predict(xhm_com01))
    verify_exp004(yva,yhm,whm,ihm,pred["OfficialVal"]["EXP004_CoM"],
                   pred["Handmade"]["EXP004_CoM"])

    y_train = np.tile(ytr,5)
    assert len(y_train)==384000
    durations, warnings_by_condition, stopping, clipping = {}, {}, {}, {}
    for condition, kind, checkpoint in (
        ("EXP005_repeat5_control", "repeat5", "exp005_repeat5_control.joblib"),
        ("EXP005_shift1_augmented", "shift1", "exp005_shift1_augmented.joblib"),
    ):
        print(f"TRAIN {condition}: constructing exactly 384000 centered / shifted inputs...",flush=True)
        t0=time.monotonic()
        training, clipping[kind] = build_training(xtr_com,kind)
        print(f"TRAIN {condition}: data ready, shape={training.shape}, "
              f"dtype={training.dtype}, MB={training.nbytes/1e6:.1f}",flush=True)
        estimator, uses_scaler = build_baseline()
        assert uses_scaler and estimator.get_params()==estimator_params
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            estimator.fit(training, y_train)
        durations[condition] = float(time.monotonic()-t0)
        warnings_by_condition[condition] = [
            f"{w.category.__name__}: {w.message}" for w in records]
        stopping[condition] = {
            "n_iter": int(estimator.n_iter_), "loss": float(estimator.loss_),
            "loss_curve": [float(t) for t in estimator.loss_curve_],
            "n_epochs_max": int(estimator.max_iter),
            "trained_examples_per_epoch":384000,
            "training_parameter_count":sum(int(v.size) for v in estimator.coefs_+estimator.intercepts_),
        }
        assert estimator.get_params()==estimator_params
        del training
        gc.collect()

        # Inference pipeline (CoM ONLY) is identical in structure to EXP004.
        pipe=Pipeline([("centering",AutomaticCentering(method="mass")),
                       ("scale_0_1",PixelScaler255()),("model",estimator)])
        pva=estimator.predict(xva_com01)
        phm=estimator.predict(xhm_com01)
        assert np.array_equal(pva,pipe.predict(xva))
        assert np.array_equal(phm,pipe.predict(xhm))
        pred["OfficialVal"][condition]=pva
        pred["Handmade"][condition]=phm
        joblib.dump(pipe,out/checkpoint,compress=3)
        print(f"TRAINED {condition}: {durations[condition]:.1f}s, iterations={estimator.n_iter_}, "
              f"Official Val={accuracy_score(yva,pva):.6%}, "
              f"Handmade={accuracy_score(yhm,phm):.6%}, "
              f"Handmade digits={(phm[yhm<10]==yhm[yhm<10]).mean():.6%}, "
              f"letters={(phm[yhm>=10]==yhm[yhm>=10]).mean():.6%}",flush=True)
        del estimator, pipe
        gc.collect()

    # Official Val and Handmade class/writer/category metrics
    metrics=group_rows("OfficialVal",yva,np.full(len(yva),-1,dtype=np.int32),
                       pred["OfficialVal"])
    metrics+=group_rows("Handmade",yhm,whm,pred["Handmade"])
    write_csv(out/"exp005_group_metrics.csv",metrics)
    transitions=compare_transitions(yhm,whm,pred["Handmade"])
    write_csv(out/"exp005_paired_transitions.csv",transitions)

    # Per-image predictions and binary correct flags preserve exact source labels.
    rows=[]
    for i,(w,ix,y) in enumerate(zip(whm,ihm,yhm)):
        rec={"writer":int(w),"image_index":int(ix),"true":LABELS[int(y)]}
        for condition in CONDITIONS:
            pi=int(pred["Handmade"][condition][i])
            rec[condition+"_prediction"]=LABELS[pi]
            rec[condition+"_correct"]=int(pi==int(y))
        rows.append(rec)
    with gzip.open(out/"exp005_handmade_predictions.csv.gz","wt",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    confusions=[]
    for domain,y in (("OfficialVal",yva),("Handmade",yhm)):
        for condition in CONDITIONS:
            cm=confusion_matrix(y,pred[domain][condition],labels=np.arange(16))
            assert cm.sum()==len(y)
            for i,j in np.ndindex((16,16)):
                confusions.append({"dataset":domain,"condition":condition,
                                   "true":LABELS[i],"predicted":LABELS[j],
                                   "count":int(cm[i,j])})
    write_csv(out/"exp005_confusion_matrices.csv",confusions)
    pairs=[]
    for domain,y in (("OfficialVal",yva),("Handmade",yhm)):
        for condition in CONDITIONS:
            for a,b in PAIR_ERRORS:
                pairs.append({"dataset":domain,"condition":condition,
                              "true":a,"predicted":b,
                              "errors":int(((y==LABELS.index(a)) &
                               (pred[domain][condition]==LABELS.index(b))).sum())})
    write_csv(out/"exp005_confusion_pairs.csv",pairs)

    loss=[]
    for condition in ("EXP005_repeat5_control","EXP005_shift1_augmented"):
        for epoch,l in enumerate(stopping[condition]["loss_curve"],start=1):
            loss.append({"condition":condition,"epoch":epoch,"training_loss":l})
    write_csv(out/"exp005_learning_curves.csv",loss)

    def stat(dataset, condition, group):
        return next(r for r in metrics if r["dataset"]==dataset and
                    r["condition"]==condition and
                    ((r["group_type"]=="all" and group=="all") or
                     (r["group_type"]=="category" and r["group"]==group)))
    overview=[
        "# EXP유005 — CoM 정렬 후 ±1px 위치 증강 (REVIEW ONLY, 실측)",
        "",
        "연구 질문: CoM 위치 정렬은 유지하면서 MLP의 절대 픽셀 위치 의존성을 줄이면 "
        "기존 문자 정확도 66.98%를 개선하고 숫자 83.19%를 유지할 수 있는가?",
        "",
        "**비교 A:** 기존 EXP유004 CoM frozen 모델, 공식 76,800장 Training으로 학습한 후보 그대로.",
        "**비교 B:** 같은 CoM 정렬 공식 Train 76,800장을 5회 정확히 반복해 384,000장으로 학습.",
        "**비교 C:** CoM 정렬 공식 Train 76,800장에 원본/좌/우/상/하 1px 변형 5종을 적용해 384,000장으로 학습.",
        "비교 B/C의 샘플 수·순서·레이블·MLP 구조·random_state=42는 동일. "
        "모델은 두 조건에서 각각 새로 학습하므로 총 epoch/optimizer update 수는 "
        "실제 수렴 경로에 따라 달라질 수 있다.",
        "",
        "**중요:** 위치 증강은 **CoM 정렬 이후 Training에만** 적용. "
        "Official Validation과 Handmade에는 CoM 정렬만 적용.",
        "**Test 사용 없음:** 공식 Original Test, 교수님 Test, "
        "Handmade fit/hyperparameter selection 모두 미실시.",
        "",
        "| 평가 | EXP유004 CoM | EXP유005 반복5 대조군 | EXP유005 CoM+±1px |",
        "|---|---:|---:|---:|",
    ]
    for domain,group in (("OfficialVal","all"),("Handmade","all"),
                         ("Handmade","digits"),("Handmade","letters")):
        accs=[stat(domain,c,group)["accuracy"] for c in CONDITIONS]
        overview.append(f"| {domain} {group} | "+" | ".join(f"{v:.4%}" for v in accs)+" |")
    overview.extend([
        "",
        "## 정확도 개선 vs 재발생 오류 (Handmade paired)",
        "| Group | Comparison | delta (pp) | Corrected | New errors |",
        "|---|---|---:|---:|---:|"])
    for row in transitions:
        if row["group"] in ("all","digits","letters"):
            overview.append(f"| {row['group']} | {row['from']}→{row['to']} | "
                            f"{row['delta_accuracy_pp']:+.3f} | "
                            f"{row['corrected']} | {row['new_errors']} |")
    overview.extend(["","## 학습·평가 조건",""])
    for condition in ("EXP005_repeat5_control","EXP005_shift1_augmented"):
        s=stopping[condition]
        overview.append(f"- {condition}: {durations[condition]:.1f}초, "
                        f"epochs={s['n_iter']}, last training loss={s['loss']:.6g}, "
                        f"warnings={warnings_by_condition[condition]}.")
    overview.extend([
        "",
        "**해석 주의:** 4명의 Hand-made는 이미 여러 번 분석된 개발용 데이터다. "
        "독립된 신규 작성자에 대한 일반화 성능은 아직 확인되지 않았다. "
        "공식 Test나 교수님 Test를 중간 비교에 사용하지 않는다.",
        "",
        "원본 검증 파일: exp005_group_metrics.csv, exp005_paired_transitions.csv, "
        "exp005_confusion_matrices.csv, exp005_confusion_pairs.csv, "
        "exp005_handmade_predictions.csv.gz, exp005_learning_curves.csv, "
        "exp005_metadata.json, 두 후보 joblib.",
        "",
        "**중간 판단은 배포/최종 모델 채택이 아님.**"])
    (out/"SUMMARY.md").write_text("\n".join(overview)+"\n",encoding="utf-8")
    info={
        "study":"EXP유005 CoM_Translation_Augmentation",
        "status":"review-only; no official results/logs publication",
        "official_npz_sha256":input_sha,
        "exp004_com_checkpoint_sha256":REFERENCE_MODEL_SHA,
        "official_train":len(ytr), "official_validation":len(yva),
        "handmade":len(yhm), "handmade_writers":[41,42,43,44],
        "handmade_used_in_fit":False, "handmade_used_for_preprocessing_selection":False,
        "official_original_test_read":False, "professor_test_read":False,
        "baseline_exp004_21760_pred_verified":True,
        "training_conditions":{
            "control":"CoM aligned identical 76,800 source images repeated 5 times",
            "candidate":"CoM aligned first; same original plus 1px L/R/U/D",
            "train_size":384000, "labels_repeated":5,
            "order_of_blocks":[a for a,_,_ in DIRECTIONS],
        },
        "evaluation_preprocess":"AutomaticCentering(method=mass), PixelScaler255 only (no shift)",
        "centering":{"threshold_0_to_255":THRESHOLD,
                     "target_xy":list(TARGET),
                     "translation":"np.rint integer offset, zero-fill, no wrap, no interpolation"},
        "training_model_params":{k:repr(v) for k,v in estimator_params.items()},
        "model_training_seconds":durations,
        "model_warnings":warnings_by_condition,
        "epochs_loss_parameters":stopping,
        "translation_ink_clipping":clipping,
        "software":{"python":platform.python_version(),"numpy":np.__version__,
                    "scipy":scipy.__version__,"sklearn":sklearn.__version__},
        "elapsed_seconds":float(time.monotonic()-start),
    }
    (out/"exp005_metadata.json").write_text(
        json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\n".join(overview),flush=True)
    print("EXP005 COMPLETE; all files:",sorted(q.name for q in out.iterdir()),flush=True)


if __name__ == "__main__":
    main()
