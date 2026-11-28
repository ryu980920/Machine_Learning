"""Shared setup for historical reproduction and follow-up. Trusted inputs only."""
from pathlib import Path
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="2"
import hashlib, json, zipfile, sys, time, platform
import numpy as np
import pandas as pd
import joblib, sklearn
from threadpoolctl import threadpool_limits
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
HERE=Path(__file__).resolve().parent
LABELS=list("0123456789ABCDEF")

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def verify_archive(inputs,name):
    row=next(r for r in json.loads((HERE/"INPUT_MANIFEST.json").read_text()) if r["filename"]==name)
    path=Path(inputs)/name
    if not path.is_file(): raise FileNotFoundError(path)
    if path.stat().st_size!=row["bytes"] or sha(path)!=row["sha256"]:
        raise ValueError(f"Input digest mismatch: {path}")
    return path

def unpack(path,target,members=None):
    target=Path(target).resolve()
    with zipfile.ZipFile(path) as z:
        names=z.namelist() if members is None else members
        for name in names:
            if not (target/name).resolve().is_relative_to(target):
                raise ValueError("Unsafe archive member")
            z.extract(name,target)

def prepare(kind,inputs,out,mode="historical",current_svc=None):
    if kind not in ("SVC","KNN") or mode not in ("historical","followup"):
        raise ValueError("Invalid model or mode")
    if sklearn.__version__!="1.8.0": raise RuntimeError("Use scikit-learn 1.8.0")
    inputs=Path(inputs).expanduser().resolve();root=Path(out).expanduser().resolve()
    archives={n:verify_archive(inputs,n) for n in (
        "Final_Validation_Private_Evaluation_Kit_20261126.zip",
        "final-validation-development-sources.zip")}
    current=json.loads((HERE/"Followup_Protocol.json").read_text())["baseline"][kind]
    model_id=kind+"_Combined"
    model_zip=None
    if mode=="followup" and kind=="SVC":
        if current_svc is None: raise ValueError("Provide trusted C2_G2 model.joblib")
        model_path=Path(current_svc).expanduser().resolve()
        if sha(model_path)!=current["sha256"]: raise ValueError("C2_G2 checkpoint mismatch")
    else:
        model_zip=verify_archive(inputs,"final-fit-"+model_id+".zip")
    reference_zip=verify_archive(inputs,"Final_10_Model_Validation_Results.zip") if mode=="historical" else None
    root.mkdir(parents=True,exist_ok=False)
    for d in ("00_baseline","01_error_analysis","02_SVC","03_KNN","runtime"):
        (root/d).mkdir()
    rt=root/"runtime";d=root/"00_baseline"/model_id;d.mkdir()
    unpack(archives["Final_Validation_Private_Evaluation_Kit_20261126.zip"],rt)
    unpack(archives["final-validation-development-sources.zip"],rt/"development",["handmade_training_only.npz","official_development_only.npz","source_audit.json"])
    if model_zip:
        unpack(model_zip,d)
        model_path=d/"model.joblib"
        manifest=json.loads((d/"fit_manifest.json").read_text())
        if sha(model_path)!=manifest["checkpoint_sha256"]: raise ValueError("Checkpoint digest mismatch")
    if mode=="followup" and sha(model_path)!=current["sha256"]: raise ValueError("Selected checkpoint mismatch")
    sys.path.insert(0,str(rt))
    if "final_benchmark" in sys.modules:
        raise RuntimeError("Restart kernel before running a different setup")
    import final_benchmark as fb
    lim=threadpool_limits(limits=2)
    v=np.load(rt/"handmade_validation_1280.npz",allow_pickle=False)
    raw=v["X"].reshape(-1,784);y=np.asarray(v["y"],int)
    assert raw.shape==(1280,784) and np.array_equal(np.bincount(y,minlength=16),np.full(16,80))
    off,wr=fb.read_training()
    assert len(off["X_train"])==76800 and len(off["X_val"])==19200
    assert sum(len(wr[w]["X"]) for w in fb.W)==2560
    Xv=fb.hybrid(raw);Xo=fb.hybrid(off["X_val"]);yo=np.asarray(off["y_val"],int)
    model=joblib.load(model_path)
    assert np.array_equal(model.classes_,np.arange(16))
    pred=model.predict(Xv)
    if reference_zip:
        with zipfile.ZipFile(reference_zip) as z:
            import io
            old=pd.read_csv(io.BytesIO(z.read(model_id+"/predictions.csv")),dtype={"true":str,"pred":str})
        assert np.array_equal(v["collector_id"],old["collector"]) and np.array_equal(v["sample_index"],old["index"])
        assert np.array_equal(y,[LABELS.index(x) for x in old["true"]])
        assert np.array_equal(pred,[LABELS.index(x) for x in old["pred"]])
    expected=current["correct"] if mode=="followup" else {"SVC":1199,"KNN":1182}[kind]
    assert int((pred==y).sum())==expected
    np.save(d/"reproduced_pred.npy",pred)
    proto=json.loads((HERE/"Historical_Protocol.json").read_text())
    (root/"EXPERIMENT_PROTOCOL.json").write_text(json.dumps(proto,indent=2),encoding="utf-8")
    rawh=np.concatenate([wr[w]["X"] for w in fb.W]);yh=np.concatenate([wr[w]["y"] for w in fb.W])
    log={"mode":mode,"model":kind,"baseline_correct":expected,"checkpoint_sha256":sha(model_path),
         "validation_sha256":sha(rt/"handmade_validation_1280.npz"),
         "inputs":{k:sha(p) for k,p in archives.items()},"environment":{"python":sys.version,"sklearn":sklearn.__version__,"platform":platform.platform(),"numpy":np.__version__},
         "original_test_used":False,"professor_test_used":False}
    if model_zip: log["inputs"][model_zip.name]=sha(model_zip)
    if reference_zip: log["inputs"][reference_zip.name]=sha(reference_zip)
    (root/"setup_manifest.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
    print(log)
    return dict(root=root,rt=rt,fb=fb,lim=lim,raw=raw,y=y,v=v,off=off,wr=wr,Xv=Xv,Xo=Xo,yo=yo,model=model,base=pred,labels=LABELS,rawh=rawh,yh=yh,proto=proto,model_path=model_path)

def metrics(y,pred,base):
    pp,rr,ff,_=precision_recall_fscore_support(y,pred,labels=np.arange(16),zero_division=0)
    return dict(n=len(y),correct=int((pred==y).sum()),accuracy=float(np.mean(pred==y)),
        macro_precision=float(pp.mean()),macro_f1=float(ff.mean()),
        corrected=int(((base!=y)&(pred==y)).sum()),regressed=int(((base==y)&(pred!=y)).sum()),
        **{f"recall_{c}":float(r) for c,r in zip(LABELS,rr)})

def evaluate_followup(state,kind,experiment_id,predict_features,config,fit_seconds=None):
    """Predict on held-out development features only. Caller supplies fitted candidate.
    All files local; candidate must have been fit on allowed training data only.
    """
    if not experiment_id or not all(c.isalnum() or c in "-_" for c in experiment_id):
        raise ValueError("Safe unique experiment ID required")
    root=state["root"];out=root/experiment_id;out.mkdir(exist_ok=False)
    (out/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    y=state["y"];yo=state["yo"];base=state["base"]
    t=time.perf_counter()
    try:
        pred=np.asarray(predict_features(state["Xv"]),int)
        po=np.asarray(predict_features(state["Xo"]),int)
        assert pred.shape==y.shape and po.shape==yo.shape
        assert np.isin(pred,np.arange(16)).all() and np.isin(po,np.arange(16)).all()
        m=metrics(y,pred,base);b=metrics(y,base,base)
        spec=json.loads((HERE/"Followup_Protocol.json").read_text())["baseline"][kind]
        bo=state["model"].predict(state["Xo"])
        assert int((bo==yo).sum())==spec["official_correct"],"Baseline official count mismatch"
        assert b["correct"]==spec["correct"] and abs(b["macro_precision"]-spec["precision"])<1e-12 and abs(b["macro_f1"]-spec["f1"])<1e-12
        m.update(official_correct=int((po==yo).sum()),fit_seconds=fit_seconds,evaluation_seconds=time.perf_counter()-t)
        m["target_pass"]=sum(m["recall_"+c] for c in "BDC")>sum(b["recall_"+c] for c in "BDC")+1e-12
        m["protected_pass"]=all(m["recall_"+c]>=b["recall_"+c]-2/80-1e-12 for c in "086")
        m["accuracy_guardrails_pass"]=bool(m["correct"]>b["correct"] and m["macro_precision"]>=b["macro_precision"] and m["macro_f1"]>b["macro_f1"] and m["official_correct"]>=spec["official_min"] and m["target_pass"] and m["protected_pass"])
        m["matched_batch1_timing_status"]="not_measured";m["adopted"]=False
        pd.DataFrame(confusion_matrix(y,pred,labels=np.arange(16)),index=LABELS,columns=LABELS).to_csv(out/"confusion_matrix.csv")
        pd.DataFrame([m]).to_csv(out/"metrics.csv",index=False)
        (out/"metrics.json").write_text(json.dumps(m,indent=2),encoding="utf-8")
        (out/"Execution.log").write_text("COMPLETED; selection pending\n"+json.dumps(m),encoding="utf-8")
        return m
    except Exception:
        import traceback
        (out/"Execution.log").write_text("FAILED\n"+traceback.format_exc(),encoding="utf-8")
        raise
