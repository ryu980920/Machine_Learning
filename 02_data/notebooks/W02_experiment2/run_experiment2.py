#!/usr/bin/env python3
"""W02 Experiment 2: five Normal (MNIST+EMNIST) baseline models.

Data: final v3 official TRAIN only (96k) => group-aware train 76,800 / val 19,200.
Never loads official X_test/y_test or Handmade for this experiment.
Defaults: sklearn default hyperparameters; only random_state=42 when present.
Scaling: SVC/MLP/KNN pixel/255, tree ensembles unscaled.

Execution from any directory:
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
   python run_experiment2.py --prepare
  ... python run_experiment2.py --model mlp
  ... python run_experiment2.py --all --timeout 3600

Each model writes metrics/<key>.json, metrics/<key>_confusion.csv,
metrics/<key>_per_class.csv, metrics/<key>_predictions.npz, models/<key>.joblib.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, subprocess, sys, time, warnings
from pathlib import Path
import numpy as np
from sklearn import __version__ as SK_VERSION
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from joblib import dump
from pixel_transform import PixelScaler255

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/data/ML_04_MNIST_EMNIST_fixed_aug_v3_train_D_duplicate_fixed.npz')
EXPECTED_SHA='0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879'
LABELS='0123456789ABCDEF'
SPECS={
    'extra_trees': ('ExtraTreesClassifier',False,lambda:ExtraTreesClassifier(random_state=42)),
    'mlp': ('MLPClassifier',True,lambda:MLPClassifier(random_state=42)),
    'knn': ('KNeighborsClassifier',True,lambda:KNeighborsClassifier()),
    'svc': ('SVC (RBF)',True,lambda:SVC()),
    'hgb': ('HistGradientBoostingClassifier',False,lambda:HistGradientBoostingClassifier(random_state=42)),
}

def verify_source(path:Path):
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if sha!=EXPECTED_SHA:raise ValueError(f'Unexpected NPZ SHA256: {sha}')
    return sha

def load_training(path:Path):
    sha=verify_source(path)
    with np.load(path,allow_pickle=False) as data:
        x=data['X_train']
        y=data['y_train'].astype(np.int64)
        src=data['train_source_index'].astype(np.int64)
        sp=data['train_source_split'].astype(np.int64)
        aug=data['train_is_augmented']
    assert x.shape==(96000,28,28) and len(y)==96000
    assert len(np.unique(y))==16 and int(aug.sum())==5349
    # Exact source group: class distinguishes MNIST digits / EMNIST letters,
    # split distinguishes original partitions; augmentation source reuses same idx.
    groups=y*1_000_000_000+sp*100_000_000+src
    return sha,x,y,groups,aug

def make_split(path):
    sha,x,y,groups,aug=load_training(path)
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    tr,va=next(cv.split(x,y,groups))
    tr=np.asarray(tr,dtype=np.int32);va=np.asarray(va,dtype=np.int32)
    assert len(tr)==76800 and len(va)==19200
    assert len(set(tr).intersection(va))==0
    assert len(set(groups[tr]).intersection(groups[va]))==0
    assert sorted(np.r_[tr,va].tolist())==list(range(96000))
    (ROOT/'metrics').mkdir(exist_ok=True,parents=True)
    np.savez_compressed(ROOT/'metrics'/'split_indices.npz',train_idx=tr,val_idx=va,train_group_ids=groups[tr],val_group_ids=groups[va])
    meta={
      'study':'W02 Experiment 2 Normal dataset baseline',
      'data_path':str(path),'data_sha256':sha,
      'source_training_n':len(y),'source_official_test_n_NOT_USED':16000,
      'train_n':len(tr),'val_n':len(va),'train_to_validation':'8:2 exact',
      'split_method':'StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42), first fold',
      'split_random_state':42,'train_val_source_group_overlap':0,
      'train_class_counts':{c:int(np.sum(y[tr]==i)) for i,c in enumerate(LABELS)},
      'validation_class_counts':{c:int(np.sum(y[va]==i)) for i,c in enumerate(LABELS)},
      'train_augmented_n':int(aug[tr].sum()),'val_augmented_n':int(aug[va].sum()),
      'feature_shape':[28,28], 'flattened_dim':784,
      'runtime':{'python':platform.python_version(),'sklearn':SK_VERSION,'numpy':np.__version__, 'platform':platform.platform()},
      'evaluation_policy':'Official Test 16,000 and Hand-made 2,560 are NOT used in Experiment 2. Fit only on group-aware Training, evaluate fixed Validation.',
      'historical_note':'A separate interrupted prior preliminary attempt had previously assessed some models on the official Test. Those earlier results are not used here, but the Test is not literally never-viewed in the history of this project.'
    }
    (ROOT/'metrics'/'split_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'SPLIT OK train={len(tr)} val={len(va)} group overlap=0, source_sha256={sha}',flush=True)
    print('class counts:',meta['train_class_counts'],meta['validation_class_counts'],flush=True)

def run_model(key,path,save_models=True):
    if not (ROOT/'metrics'/'split_indices.npz').exists():raise FileNotFoundError('Run --prepare first')
    sha,x,y,groups,aug=load_training(path)
    with np.load(ROOT/'metrics'/'split_indices.npz',allow_pickle=False) as idx:
        tr=idx['train_idx'];va=idx['val_idx']
        stored_train_ids=idx['train_group_ids'];stored_val_ids=idx['val_group_ids']
    assert len(tr)==76800 and len(va)==19200
    assert np.array_equal(groups[tr],stored_train_ids) and np.array_equal(groups[va],stored_val_ids)
    assert not set(groups[tr]).intersection(groups[va])
    Xtr=x[tr].reshape(-1,784).astype(np.float32)
    Xva=x[va].reshape(-1,784).astype(np.float32)
    ytr=y[tr];yva=y[va]
    name,scale,build=SPECS[key]
    impl=build()
    model=Pipeline([('scale_0_1',PixelScaler255()),('model',impl)]) if scale else impl
    print(f'START {key} name={name} train={len(tr)} val={len(va)} scaled={scale}',flush=True)
    t=time.perf_counter()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        model.fit(Xtr,ytr)
    fit=time.perf_counter()-t
    print(f'FIT {key}: {fit:.3f}s',flush=True)
    st=time.perf_counter();yvpred=model.predict(Xva);pred=time.perf_counter()-st
    # Important: same fixed training diagnostic sample, not full training evaluation.
    rng=np.random.default_rng(56)
    sampled=np.concatenate([rng.choice(np.flatnonzero(ytr==i),size=100,replace=False) for i in range(16)])
    st=time.perf_counter();trpred=model.predict(Xtr[sampled]);train_pred=time.perf_counter()-st
    train_probe_acc=accuracy_score(ytr[sampled],trpred)
    cm=confusion_matrix(yva,yvpred,labels=np.arange(16))
    acc=accuracy_score(yva,yvpred)
    f1m=f1_score(yva,yvpred,average='macro',labels=np.arange(16),zero_division=0)
    precis,recall,f1,support=precision_recall_fscore_support(yva,yvpred,labels=np.arange(16),zero_division=0)
    warning_text=[f'{w.category.__name__}: {w.message}' for w in ws]
    niter=getattr(impl,'n_iter_',None)
    if niter is not None:
        niter=np.asarray(niter).tolist()
    result={
        'experiment':'EXP-유-003 / W02 Experiment 2 — Normal baseline',
        'key':key,'model':name,'status':'completed','data_sha256':sha,
        'n_train':len(tr),'n_validation':len(va),'validation_accuracy':float(acc),'validation_macro_f1':float(f1m),
        'training_seconds':float(fit),'validation_inference_seconds':float(pred),
        'validation_ms_per_image':float(pred*1000/len(va)),
        'training_probe_accuracy':float(train_probe_acc),'training_probe_size':1600,
        'training_probe_predict_seconds':float(train_pred),
        'warnings':warning_text,'n_iter':niter,
        'scaled_pixels_0_to_1':scale,'hyperparameters':{str(k):repr(v) for k,v in impl.get_params().items()},
        'model_filename':f'models/{key}.joblib' if save_models else None,
        'class_label_order':LABELS,
        'per_class':{c:{'precision':float(precis[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i,c in enumerate(LABELS)},
        'confusion_matrix':cm.tolist(),
        'runtime':{'python':platform.python_version(),'scikit-learn':SK_VERSION,'numpy':np.__version__},
    }
    assert sum(cm.diagonal())/len(yva)==acc
    assert abs(np.mean(f1)-f1m)<1e-12
    (ROOT/'metrics').mkdir(exist_ok=True)
    with (ROOT/'metrics'/f'{key}_confusion_matrix.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['actual\\pred']+list(LABELS));
        for i,c in enumerate(LABELS):w.writerow([c]+cm[i].tolist())
    with (ROOT/'metrics'/f'{key}_per_class.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['label','precision','recall','f1','support']);
        for c in LABELS:
            a=result['per_class'][c];w.writerow([c,a['precision'],a['recall'],a['f1'],a['support']])
    np.savez_compressed(ROOT/'metrics'/f'{key}_predictions.npz',y_true=yva.astype(np.uint8),y_pred=yvpred.astype(np.uint8),validation_indices=va)
    (ROOT/'metrics'/f'{key}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    if save_models:
        (ROOT/'models').mkdir(exist_ok=True)
        dump(model,ROOT/'models'/f'{key}.joblib',compress=3)
    print(f'END {key}: acc={acc:.6f} macro_f1={f1m:.6f} train_probe_acc={train_probe_acc:.6f} training_s={fit:.3f} val_predict_s={pred:.3f} warnings={len(warning_text)}',flush=True)

def aggregate():
    import csv
    rows=[]
    for key,(name,scale,build) in SPECS.items():
        p=ROOT/'metrics'/f'{key}.json'
        if p.exists():
            d=json.loads(p.read_text(encoding='utf-8'))
            rows.append({
                'model':name,'status':d['status'],'train_n':d['n_train'],'val_n':d['n_validation'],
                'train_sample_accuracy':d['training_probe_accuracy'],'validation_accuracy':d['validation_accuracy'],
                'macro_f1':d['validation_macro_f1'],'fit_seconds':d['training_seconds'],
                'predict_seconds':d['validation_inference_seconds'],'predict_ms_per_image':d['validation_ms_per_image'],
                'scaling':'pixel / 255' if d['scaled_pixels_0_to_1'] else 'raw pixels 0-255',
                'n_iter':d['n_iter'],'warnings':' | '.join(d['warnings'])
            })
        else:rows.append({'model':name,'status':'not_completed'})
    (ROOT/'metrics').mkdir(exist_ok=True)
    cols=['model','status','train_n','val_n','train_sample_accuracy','validation_accuracy','macro_f1','fit_seconds','predict_seconds','predict_ms_per_image','scaling','n_iter','warnings']
    with (ROOT/'metrics'/'comparison_5models.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for row in rows:w.writerow(row)
    (ROOT/'metrics'/'comparison_5models.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    print('AGGREGATED', [(r['model'],r['status']) for r in rows],flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--model',choices=SPECS)
    ap.add_argument('--all',action='store_true')
    ap.add_argument('--timeout',type=int,default=3600)
    ap.add_argument('--input',type=Path,default=SOURCE)
    ap.add_argument('--no-save-models',action='store_true')
    a=ap.parse_args()
    if a.prepare:make_split(a.input)
    elif a.model:run_model(a.model,a.input,not a.no_save_models)
    elif a.all:
        if not (ROOT/'metrics'/'split_indices.npz').exists():make_split(a.input)
        env={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2'}
        for key in SPECS:
            if (ROOT/'metrics'/f'{key}.json').exists():continue
            print(f'==== START {key} ==== ',flush=True)
            st=time.perf_counter()
            cmd=['timeout','--signal=TERM','--kill-after=5s',f'{a.timeout}s',sys.executable,str(Path(__file__).resolve()),'--model',key,'--input',str(a.input)]
            if a.no_save_models:cmd.append('--no-save-models')
            p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,cwd=ROOT)
            print(p.stdout[-2500:],flush=True)
            if p.returncode:
                reason={'key':key,'model':SPECS[key][0],'status':'timeout' if p.returncode==124 else 'failed','exit_code':p.returncode,'wall_elapsed_seconds':time.perf_counter()-st,'log_tail':p.stdout[-3000:]}
                (ROOT/'metrics'/f'{key}_failure.json').write_text(json.dumps(reason,ensure_ascii=False,indent=2))
            aggregate()
        aggregate()
    else:ap.error('Choose --prepare, --model or --all')
if __name__=='__main__':main()