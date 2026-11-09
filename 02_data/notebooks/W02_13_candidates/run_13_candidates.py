#!/usr/bin/env python3
"""Run 13 W01 classifier candidates with scikit-learn defaults on a common split.

Usage: python run_13_candidates.py --all --timeout 95
       python run_13_candidates.py --model 1

13 entries (SVC RBF representing default SVC; OvR logistic requires wrapper).
Random_state=42 is set only where available for reproducibility, all other
estimator hyperparameters remain their installed sklearn defaults.
"""
from __future__ import annotations
import argparse,csv,json,os,platform,subprocess,sys,time,warnings
from pathlib import Path
import numpy as np
from sklearn import __version__ as SKVERSION
from sklearn.linear_model import LogisticRegression,SGDClassifier,Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report

ROOT=Path(__file__).resolve().parent
CANDIDATES={
 1:('Logistic Regression (OvR)',True,lambda:OneVsRestClassifier(LogisticRegression(random_state=42))),
 2:('Softmax Regression',True,lambda:LogisticRegression(random_state=42)),
 3:('SGDClassifier',True,lambda:SGDClassifier(random_state=42)),
 4:('LinearSVC',True,lambda:LinearSVC(random_state=42)),
 5:('SVC (RBF, default)',True,lambda:SVC()),
 6:('KNeighborsClassifier (KNN)',True,lambda:KNeighborsClassifier()),
 7:('DecisionTreeClassifier',False,lambda:DecisionTreeClassifier(random_state=42)),
 8:('RandomForestClassifier',False,lambda:RandomForestClassifier(random_state=42)),
 9:('ExtraTreesClassifier',False,lambda:ExtraTreesClassifier(random_state=42)),
 10:('AdaBoostClassifier',False,lambda:AdaBoostClassifier(random_state=42)),
 11:('HistGradientBoostingClassifier',False,lambda:HistGradientBoostingClassifier(random_state=42)),
 12:('Perceptron',True,lambda:Perceptron(random_state=42)),
 13:('MLPClassifier',True,lambda:MLPClassifier(random_state=42)),
}

def estimator(i):
    name,scale,build=CANDIDATES[i]
    model=build()
    if scale:
        model=Pipeline([('scale_0_1',FunctionTransformer(lambda x:x/255.0,validate=False)),('model',model)])
    return model

def run_model(i):
    name,scale,_=CANDIDATES[i]
    with np.load(ROOT/'common_subset.npz',allow_pickle=False) as f:
        xtr=f['X_train'].reshape(-1,784).astype('float32')
        ytr=f['y_train'].astype(int)
        xv=f['X_val'].reshape(-1,784).astype('float32')
        yv=f['y_val'].astype(int)
    clf=estimator(i)
    t=time.perf_counter()
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always')
        clf.fit(xtr,ytr)
    fit_time=time.perf_counter()-t
    t=time.perf_counter()
    yp=clf.predict(xv)
    predict_time=time.perf_counter()-t
    e=clf.named_steps['model'] if isinstance(clf,Pipeline) else clf
    iterations=getattr(e,'n_iter_',None)
    if isinstance(iterations,np.ndarray): iterations=iterations.tolist()
    if isinstance(iterations,np.integer): iterations=int(iterations)
    diag_idx=np.arange(0,len(xtr),max(1,len(xtr)//1600))[:1600]
    train_probe_accuracy=float(accuracy_score(ytr[diag_idx],clf.predict(xtr[diag_idx])))
    result={'id':i,'model':name,'status':'completed','train_n':int(len(xtr)),
       'validation_n':int(len(xv)), 'accuracy':float(accuracy_score(yv,yp)),
       'macro_f1':float(f1_score(yv,yp,average='macro',labels=list(range(16)),zero_division=0)),
       'train_probe_n':len(diag_idx),'train_probe_accuracy':train_probe_accuracy,
       'fit_seconds':round(fit_time,6),'predict_seconds':round(predict_time,6),
       'predict_ms_per_image':round(predict_time*1000/len(xv),6),
       'scaling':'x/255' if scale else 'none (raw 0-255 pixels)',
       'n_iter':iterations,
       'warnings':[f'{w.category.__name__}: {str(w.message)}'[:800] for w in warns],
       'params':str(clf.get_params(deep=True)),
       'confusion_matrix':confusion_matrix(yv,yp,labels=list(range(16))).tolist(),
       'per_class':classification_report(yv,yp,labels=list(range(16)),target_names=list('0123456789ABCDEF'),output_dict=True,zero_division=0),
       'versions':{'python':platform.python_version(),'numpy':np.__version__,'scikit-learn':SKVERSION}}
    (ROOT/f'candidate_{i:02d}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'DONE {i:02d} {name} acc={result["accuracy"]:.4f} macroF1={result["macro_f1"]:.4f} fit={fit_time:.3f}s predict={predict_time:.3f}s',flush=True)

def aggregate():
    rows=[]
    for i,(name,scale,_) in CANDIDATES.items():
        path=ROOT/f'candidate_{i:02d}.json'
        rows.append(json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'id':i,'model':name,'status':'not_run'})
    columns=['id','model','status','train_n','validation_n','accuracy','macro_f1','fit_seconds',
            'predict_seconds','predict_ms_per_image','scaling','train_probe_accuracy','n_iter','warnings','reason']
    with (ROOT/'results_13_models.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for d in rows:
            row={key:d.get(key,'') for key in columns}
            row['warnings']=' | '.join(d.get('warnings',[])) if isinstance(d.get('warnings',[]),list) else str(d.get('warnings',''))
            w.writerow(row)
    m=json.loads((ROOT/'common_data_metadata.json').read_text(encoding='utf-8'))
    (ROOT/'results_13_models.json').write_text(json.dumps({'metadata':m,'models':rows},ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--all',action='store_true');pa.add_argument('--model',type=int,choices=list(CANDIDATES));pa.add_argument('--timeout',type=int,default=95)
    a=pa.parse_args()
    if a.model:run_model(a.model)
    elif a.all:
        for i,(name,scale,_) in CANDIDATES.items():
            if (ROOT/f'candidate_{i:02d}.json').exists():continue
            print(f'START {i:02d} {name}',flush=True)
            t=time.perf_counter()
            env={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2'}
            p=subprocess.run(['timeout','--signal=TERM','--kill-after=4s',f'{a.timeout}s',sys.executable,str(Path(__file__).resolve()),'--model',str(i)],
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,cwd=ROOT)
            elapsed=time.perf_counter()-t
            print(p.stdout[-2000:],flush=True)
            if p.returncode !=0:
                status='timeout' if p.returncode==124 else 'failed'
                reason=f'Exit {p.returncode} elapsed {elapsed:.1f}s with max {a.timeout}s (see execution_log.txt)'
                (ROOT/f'candidate_{i:02d}.json').write_text(json.dumps({'id':i,'model':name,'status':status,'reason':reason,
                  'elapsed_seconds':elapsed,'trace_tail':p.stdout[-2000:]},ensure_ascii=False,indent=2),encoding='utf-8')
                print(f'{status.upper()} {i:02d} {name} {reason}',flush=True)
            aggregate()
            print(f'PROGRESS {i}/13',flush=True)
        aggregate();print('ALL 13 candidates attempted',flush=True)
    else:pa.error('Specify --all or --model')

if __name__=='__main__':main()