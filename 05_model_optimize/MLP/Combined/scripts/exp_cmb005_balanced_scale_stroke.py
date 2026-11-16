#!/usr/bin/env python3
"""EXP-CMB-005. Match Normal's class-balanced Scale/Stroke, retain Combined writer CV.

Four pre-registered arms: com_only, scale70_stroke00, scale80_stroke00,
scale80_stroke14. Model hyperparameters fixed and equal across arms; 3 seeds
42/43/44; 4 held-out writers per arm. No Original/Professor Test or new 480.
No fitting, augmentation, or selection using held-out writers/OfficialVal.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,time,warnings
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,roc_auc_score,
                             precision_recall_fscore_support,confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[4]
CMB=ROOT/"05_model_optimize"/"MLP"/"Combined"
NORMAL=ROOT/"05_model_optimize"/"MLP"/"Normal"
sys.path.insert(0,str(CMB/"scripts"))
sys.path.insert(0,str(NORMAL/"scripts"))
from combined_mlp_writer_cv import (load_official,load_writers,PixelScaler255,
                                    WRITERS,LABELS,V3_SHA,HANDMADE_SHA)
from exp006_size_com_comparison import process,image_geometry
from exp016_high_fraction_grid import nested_subset,scale_base,stroke_jitter
from exp014_balanced_augmentation import balanced_indices
from exp011_012_causal_probes import fixed_scale

SCALES=(.8,.9,1.,1.1,1.2)
PAIRS=(("0","D"),("D","0"),("8","B"),("B","8"),("5","F"),("F","5"),
       ("7","9"),("7","A"),("4","F"))
ARMS={"com_only":(0.,0.),
      "scale70_stroke00":(.7,0.),
      "scale80_stroke00":(.8,0.),
      "scale80_stroke14":(.8,.14)}
TARGETS=(16,18,20,22,24)
SEEDS=(42,43,44)
CLASSES=np.arange(16,dtype=int)

def savecsv(p,records):
    records=list(records)
    if not records:raise RuntimeError("missing rows "+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="",encoding="utf-8-sig") as f:
        wr=csv.DictWriter(f,fieldnames=list(records[0]))
        wr.writeheader();wr.writerows(records)

def metrics(y,p,subset=None):
    labels=CLASSES if subset is None else np.asarray(subset)
    prec,rec,f1,sup=precision_recall_fscore_support(y,p,labels=labels,zero_division=0)
    return {"n":int(len(y)),"correct":int(np.sum(p==y)),
            "accuracy":float(np.mean(p==y)),
            "macro_f1":float(f1.mean()),
            "macro_precision":float(prec.mean()),
            "macro_recall":float(rec.mean())}

def confusion_pairs(y,p,writer,arm,seed,source):
    out=[]
    for a,b in PAIRS:
        z=y==LABELS.index(a)
        out.append({"arm":arm,"seed":seed,"dataset":source,"writer":writer,
             "actual":a,"predicted":b,"n_actual":int(z.sum()),
             "errors":int(np.sum(p[z]==LABELS.index(b)))})
    return out

def geom_rows(name,x,y,arm,seed):
    geo=image_geometry(x)
    out=[]
    for nameg,mask in (("digits",y<10),("letters",y>=10)):
        z={"arm":arm,"seed":seed,"dataset":name,"group":nameg,
           "n":int(mask.sum())}
        for field in ("long_side","width","height","mass"):
            zz=np.asarray(geo[field])[mask]
            z[field+"_mean"]=float(zz.mean())
            z[field+"_std"]=float(zz.std())
        out.append(z)
    return out

def binary_size_probe(xtrain,ytrain,xval,yval):
    train=image_geometry(xtrain)["long_side"].reshape(-1,1)
    val=image_geometry(xval)["long_side"].reshape(-1,1)
    model=Pipeline([("std",StandardScaler()),
      ("logistic",LogisticRegression(class_weight="balanced",random_state=42,max_iter=1000))])
    model.fit(train,(ytrain>=10).astype(int))
    q=model.predict_proba(val)[:,1]
    return {"bal_acc":float(balanced_accuracy_score(yval>=10,q>=.5)),
            "auc":float(roc_auc_score(yval>=10,q))}

def prepareraw(official,writers,arm):
    ratio,stroke=ARMS[arm]
    raw=official["X_train"]
    yy=official["y_train"]
    # CoM-only is exactly the EXP-CMB-004 reference condition.
    main=process(raw,"com",20)
    changed={"method":arm,"requested_ratio":ratio,"actual_ratio":0.,
             "n_aug":0,"n_stroke_changed":0,"n_stroke_rejected":0,
             "official_train_rows":len(raw)}
    targets=[]
    if ratio:
        ids,target_map=nested_subset(yy,ratio)
        half,_=balanced_indices(yy,seed=20261116)
        augmented=scale_base(raw,ids,target_map)
        augmented,stroke_stats=stroke_jitter(augmented,stroke,ids,half)
        main[ids]=process(augmented,"com",20)
        changed.update({"n_aug":len(ids),"actual_ratio":len(ids)/len(yy),
             "n_stroke_changed":stroke_stats["modified"],
             "n_stroke_rejected":stroke_stats["rejected"]})
        for cl in CLASSES:
            for length in TARGETS:
                ct=int(np.sum((yy[ids]==cl)&(target_map[ids]==length)))
                targets.append({"arm":arm,"class":LABELS[cl],"target_size":length,
                                "train_images_assigned":ct})
                assert ct>=1
        for cl in CLASSES:
            cc=[x["train_images_assigned"] for x in targets if x["class"]==LABELS[cl]]
            assert max(cc)-min(cc)<=1
        assert abs(len(ids)/len(yy)-ratio)<.002
    else:
        for cl in CLASSES:
            for length in TARGETS:
                targets.append({"arm":arm,"class":LABELS[cl],"target_size":length,
                                "train_images_assigned":0})
    transformed_w={w:process(writers[w]["X"],"com",20) for w in WRITERS}
    assert main.shape==(76800,784) and len(transformed_w)==4
    return main,transformed_w,changed,targets

def make_scaled(official,writers):
    ref={}
    for scale in SCALES:
        # exact Normal EXP012 intervention; raw image BBox-centered affine, then CoM
        ref[("official",scale)]=process(fixed_scale(official["X_val"],scale),"com",20)
        for w in WRITERS:
            ref[(w,scale)]=process(fixed_scale(writers[w]["X"],scale),"com",20)
    return ref

def eval_scales(model,raw_map,y,dataset,writer,arm,seed):
    rows=[]; pair_rows=[]; focus=[]
    probs={}
    for sc in SCALES:
        xx=raw_map[(writer if dataset=="OOF" else "official",sc)]
        q=np.asarray(model.predict_proba(xx),dtype=np.float64)
        assert np.array_equal(model.named_steps["model"].classes_,CLASSES)
        p=q.argmax(axis=1).astype(int)
        letter=q[:,10:].sum(axis=1)
        probs[sc]=(p,letter)
    _,base_letter=probs[1.]
    base_family=base_letter>=.5
    for sc in SCALES:
        p,letter=probs[sc]
        for group,z,cls in (("all",np.ones(len(y),dtype=bool),CLASSES),
             ("digits",y<10,np.arange(10)),("letters",y>=10,np.arange(10,16))):
            m=metrics(y[z],p[z],cls)
            flip=(letter[z]>=.5)!=(base_family[z])
            rows.append({"arm":arm,"seed":seed,"writer":writer,"dataset":dataset,
               "scale":sc,"group":group,**m,
               "family_flip":float(flip.mean()),
               "mean_abs_delta_p_letter":float(np.abs(letter[z]-base_letter[z]).mean()),
               "mean_p_letter":float(letter[z].mean())})
        for cls_i in CLASSES:
            z=y==cls_i
            pz=p[z];ll=letter[z];bb=base_letter[z]
            focus.append({"arm":arm,"seed":seed,"writer":writer,"dataset":dataset,
                "scale":sc,"actual":LABELS[cls_i],"n":int(z.sum()),
                "correct":int(np.sum(pz==cls_i)),
                "accuracy":float(np.mean(pz==cls_i)),
                "family_flip":float(np.mean((ll>=.5)!=(bb>=.5))),
                "mean_abs_delta_p_letter":float(np.mean(np.abs(ll-bb))),
                "mean_p_letter":float(ll.mean())})
        if dataset=="OOF":
            pair_rows+=confusion_pairs(y,p,writer,arm,seed,"OOF_scale_"+str(sc))
    return rows,pair_rows,focus,probs[1.][0]

def output_plots(y,p,out,arm,seed):
    cm=confusion_matrix(y,p,labels=CLASSES)
    for k in ("counts","row_normalized"):
        v=cm if k=="counts" else 100*cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
        fig,ax=plt.subplots(figsize=(10,8))
        im=ax.imshow(v,cmap="viridis",vmin=0,vmax=160 if k=="counts" else 100)
        fig.colorbar(im,ax=ax,fraction=.045,pad=.04)
        ax.set(xticks=np.arange(16),yticks=np.arange(16),
               xticklabels=list(LABELS),yticklabels=list(LABELS),
               xlabel="Predicted",ylabel="True",
               title=f"EXP-CMB-005 {arm} seed={seed} Combined OOF {k}")
        fig.tight_layout()
        fig.savefig(out/f"OOF_confusion_{k}.png",dpi=145)
        plt.close(fig)

def run(arm,seed,out):
    if arm not in ARMS or seed not in SEEDS:raise ValueError((arm,seed))
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    (out/"models").mkdir()
    t0=time.perf_counter()
    original=ROOT/"02_data"/"datasets"/"MNIST_EMNIST_official_split_112000.npz"
    handmade=ROOT/"02_data"/"Hand_made"
    off=load_official(original)
    writers=load_writers(handmade)
    assert off["sha256"]==V3_SHA
    for w in WRITERS:assert writers[w]["sha256"]==HANDMADE_SHA[w]
    assert len(off["X_train"])==76800 and len(off["X_val"])==19200
    tpre=time.perf_counter()
    xtrain_off,xwriters,info,targets=prepareraw(off,writers,arm)
    xval=process(off["X_val"],"com",20)
    scale_map=make_scaled(off,writers)
    preprocess_time=time.perf_counter()-tpre
    official_probe=binary_size_probe(xtrain_off,off["y_train"],xval,off["y_val"])
    savecsv(out/"target_assignments.csv",targets)
    features=geom_rows("official_train",xtrain_off,off["y_train"],arm,seed)
    features+=geom_rows("official_val",xval,off["y_val"],arm,seed)
    for w in WRITERS:features+=geom_rows("writer_"+str(w),xwriters[w],writers[w]["y"],arm,seed)
    savecsv(out/"geometry_distributions.csv",features)
    foldrows=[];pairs=[];scale_rows=[];class_rows=[];pred_rows=[];loss=[];fit=[]
    pooled=[]
    for held in WRITERS:
        trwriters=[w for w in WRITERS if w!=held]
        X=np.concatenate([xtrain_off]+[xwriters[w] for w in trwriters])
        y=np.concatenate([off["y_train"]]+[writers[w]["y"] for w in trwriters])
        assert X.shape==(78720,784) and len(y)==78720
        mlp=Pipeline([("scale_0_1",PixelScaler255()),
                      ("model",MLPClassifier(random_state=seed))])
        started=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mlp.fit(X,y)
        sec=time.perf_counter()-started
        mlpnet=mlp.named_steps["model"]
        assert mlpnet.n_features_in_==784
        for it,lossval in enumerate(mlpnet.loss_curve_):
            loss.append({"arm":arm,"seed":seed,"heldout":held,
                        "iteration":it+1,"loss":float(lossval)})
        fit.append({"arm":arm,"seed":seed,"heldout":held,
                    "train_n":len(y),"heldout_n":640,
                    "fit_seconds":sec,"iterations":mlpnet.n_iter_,
                    "last_loss":float(mlpnet.loss_),
                    "warnings":" | ".join(str(w.message) for w in caught)})
        fout=out/"models"/f"heldout_{held}.joblib"
        joblib.dump(mlp,fout,compress=3)
        handwritten_y=writers[held]["y"]
        outcome=mlp.predict(xwriters[held]).astype(int)
        pp=mlp.predict(xval).astype(int)
        for ds,ytrue,pred in (("OOF",handwritten_y,outcome),
                              ("OfficialVal",off["y_val"],pp)):
            maskset=(("all",np.ones(len(ytrue),bool),CLASSES),
                     ("digits",ytrue<10,np.arange(10)),
                     ("letters",ytrue>=10,np.arange(10,16)))
            for g,z,classes in maskset:
                foldrows.append({"arm":arm,"seed":seed,"heldout":held,"dataset":ds,
                      "group":g,**metrics(ytrue[z],pred[z],classes)})
        for ci in CLASSES:
            z=handwritten_y==ci
            prec,rec,f1,_=precision_recall_fscore_support(handwritten_y,outcome,
                  labels=CLASSES,zero_division=0)
            class_rows.append({"arm":arm,"seed":seed,"writer":held,
               "class":LABELS[ci],"n":int(z.sum()),"correct":int(np.sum(outcome[z]==ci)),
               "precision":float(prec[ci]),"recall":float(rec[ci]),"f1":float(f1[ci])})
        for i in range(640):
            pred_rows.append({"arm":arm,"seed":seed,"writer":held,"index":i,
                "actual":LABELS[int(handwritten_y[i])],
                "pred":LABELS[int(outcome[i])]})
        pooled.append(outcome)
        pairs+=confusion_pairs(handwritten_y,outcome,held,arm,seed,"OOF_original")
        for ds,ytrue,wid in (("OOF",handwritten_y,held),
                             ("OfficialVal",off["y_val"],held)):
            z,p2,c,base=eval_scales(mlp,scale_map,ytrue,ds,wid,arm,seed)
            scale_rows+=z;pairs+=p2;class_rows+=c
            if ds=="OOF":assert np.array_equal(base,outcome)
            else:assert np.array_equal(base,pp)
        print("FOLD_COMPLETED",arm,seed,held,
              "OOF",round(np.mean(outcome==handwritten_y),6),
              "VAL",round(np.mean(pp==off["y_val"]),6),
              "fit_s",round(sec,2),flush=True)
        del X,y,mlp
    hand_y=np.concatenate([writers[w]["y"] for w in WRITERS])
    pooled_p=np.concatenate(pooled)
    assert len(hand_y)==2560
    output_plots(hand_y,pooled_p,out,arm,seed)
    savecsv(out/"fold_metrics.csv",foldrows)
    savecsv(out/"per_class_and_scale.csv",class_rows)
    savecsv(out/"pair_confusions.csv",pairs)
    savecsv(out/"scale_sensitivity.csv",scale_rows)
    savecsv(out/"oof_predictions.csv",pred_rows)
    savecsv(out/"training_loss.csv",loss)
    savecsv(out/"training_time.csv",fit)
    overall=metrics(hand_y,pooled_p)
    meta={"experiment":"EXP-CMB-005","status":"arm completed, aggregation pending",
          "arm":arm,"seed":seed,"handmade_cv":"Leave-One-Writer-Out 4-Fold",
          "official_n":76800,"official_val_n":19200,"training_handmade_each_fold":1920,
          "oof_n":2560,"scale_interventions":SCALES,"class_balanced_targets":TARGETS,
          "official_train_only_augmentation":True,
          "handmade_train_preprocess":"CoM-only",
          "heldout_and_official_val_preprocess":"CoM-only",
          "augmentation_seed":20261116,"augmentation":info,
          "official_sha256":off["sha256"],
          "handmade_sha256":{str(w):writers[w]["sha256"] for w in WRITERS},
          "official_train_size_binary_probe":official_probe,
          "oof_metrics":overall,"fit_total_seconds":sum(r["fit_seconds"] for r in fit),
          "preprocessing_seconds":preprocess_time,
          "wall_seconds":time.perf_counter()-t0,
          "original_test_used":False,"professor_test_used":False,
          "new_handmade_480_used":False,
          "versions":{"numpy":np.__version__,"scikit_learn":sklearn.__version__},
          "model_sha256":{str(w):hashlib.sha256((out/"models"/f"heldout_{w}.joblib").read_bytes()).hexdigest()
                          for w in WRITERS}}
    (out/"manifest.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("COMPLETED",arm,seed,json.dumps(overall),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",choices=tuple(ARMS),required=True)
    ap.add_argument("--seed",type=int,choices=SEEDS,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ar=ap.parse_args()
    run(ar.arm,ar.seed,ar.out)
