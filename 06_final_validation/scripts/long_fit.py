"""Long-duration fit only; exact EXP_FINAL_001 recipes. Validation inputs never present."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'): os.environ[k]='2'
import sys,time,json,hashlib,platform,gc,warnings,importlib.metadata
from pathlib import Path
import numpy as np,joblib,sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.base import clone
from threadpoolctl import threadpool_limits,threadpool_info
BASE=Path(os.environ['BENCH_ROOT'])
ROOT=BASE/'sources'
for p in ('05_model_optimize/MLP/Normal/scripts','05_model_optimize/MLP/Combined/scripts','05_model_optimize/KNN/Normal/scripts','05_model_optimize/KNN/Combined/scripts','05_model_optimize/SVC/Normal/scripts','05_model_optimize/SVC/Combined/scripts','05_model_optimize/Extra_Trees/Normal/scripts','03_experiments/model_optimization'):
    sys.path.insert(0,str(ROOT/p))
from exp006_size_com_comparison import process,PixelScaler255
from exp_cmb005_balanced_scale_stroke import prepareraw
from exp016_high_fraction_grid import nested_subset,scale_base,stroke_jitter as mlp_stroke
from exp014_balanced_augmentation import balanced_indices
from EXP004_KNN_Normal_CoM import BatchedCoM
from EXP011_KNN_Normal_Deskew import deskew,test_deskew
from EXP012_KNN_Normal_HOG import hog_features,hybrid_features,test_hog
from final_knn_normal import assemble_training
from EXP007_KNN_Normal_Augmentation import size_jitter,stroke_jitter as knn_stroke
from EXP_CMB_003_KNN_Combined_Handmade_Augmentation import balanced_nested_indices,SEED as HAND_SEED,AUG_STRENGTH
from EXP_CMB_002_KNN_Combined_Optimized import fresh_knn
from EXP_001_SVC_Normal_Baseline import baseline as svc_baseline
from EXP_CMB_005_SVC_Combined_Targeted_DB_Augmentation import TARGET_CLASSES,TARGET_SEED
from run_extra_trees import extract as et_extract
W=(41,42,43,44);SEED=42
MODELS=['MLP_Normal','MLP_Combined','SVC_Normal','SVC_Combined','KNN_Normal','KNN_Combined','HGB_Normal','HGB_Combined','ExtraTrees_Normal','ExtraTrees_Combined']
COUNTS=dict(zip(MODELS,(76800,79360,76800,82080,107520,113920,76800,79360,76800,79360)))
def digest(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_json(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
def environment():
    return {'python':platform.python_version(),'platform':platform.platform(),'cpu':next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),platform.processor()),'cpu_affinity':sorted(os.sched_getaffinity(0)),'threads':2,'versions':{p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','scikit-image','joblib','opencv-python-headless')},'threadpools':threadpool_info()}
def read_training():
    audit=json.loads((BASE/'development/source_audit.json').read_text())
    assert audit['official_source_sha256']=='0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879'
    assert not audit['original_test_accessed'] and not audit['new_validation_accessed']
    with np.load(BASE/'development/official_development_only.npz',allow_pickle=False) as z:off={k:z[k] for k in z.files}
    with np.load(BASE/'development/handmade_training_only.npz',allow_pickle=False) as z:wr={w:{'X':z[f'X{w}'].astype(np.float32),'y':z[f'y{w}'].astype(np.int64),'sha256':audit['handmade_sources'][str(w)]} for w in W}
    off['sha256']=audit['official_source_sha256']
    assert off['X_train'].shape==(76800,784) and off['X_val'].shape==(19200,784)
    assert all(wr[w]['X'].shape==(640,784) and np.array_equal(np.bincount(wr[w]['y'],minlength=16),np.full(16,40)) for w in W)
    return off,wr
def hybrid(raw):
    a=BatchedCoM().transform(np.asarray(raw,dtype=np.float32).reshape(-1,784));b,_=deskew(a);del a
    h=hog_features(b);return hybrid_features(b,h,pixel_weight=.2)
def feature(name,raw):
    raw=np.asarray(raw,dtype=np.float32).reshape(-1,784)
    if name.startswith('MLP'):return process(raw,'com',20)
    if name.startswith(('KNN','SVC')):return hybrid(raw)
    return et_extract(process(raw,'com',20),'pixels+hog')
def make_model(name):
    if name.startswith('MLP'):
        h,a,b=((256,128),.03,200) if name=='MLP_Normal' else ((256,),.0001,128)
        return Pipeline([('scale_0_1',PixelScaler255()),('model',MLPClassifier(hidden_layer_sizes=h,alpha=a,batch_size=b,learning_rate_init=.001,random_state=42,activation='relu',solver='adam',early_stopping=False,max_iter=200,tol=.0001,n_iter_no_change=10))])
    if name.startswith('SVC'):
        svc,scaled,_=svc_baseline();assert scaled and svc.kernel=='rbf'
        return clone(svc).set_params(C=1.,gamma='scale',cache_size=200)
    if name.startswith('KNN'):return fresh_knn().set_params(n_jobs=2)
    if name.startswith('HGB'):return HistGradientBoostingClassifier(learning_rate=.1,max_leaf_nodes=31,l2_regularization=1.,max_iter=300,early_stopping=True,validation_fraction=.1,n_iter_no_change=10,random_state=42)
    return ExtraTreesClassifier(n_estimators=100,max_features=.18,min_samples_leaf=1,max_depth=None,random_state=42,n_jobs=2)
def hand_aug(raw,indices,seed):
    resized,_=size_jitter(raw,indices,seed=seed);altered,_=knn_stroke(resized,strength=AUG_STRENGTH);return hybrid(altered)
def official_only_30(off):
    from exp015_scale_stroke_grid import nested_subset as low_subset
    x,y=off['X_train'],off['y_train'];main=process(x,'com',20);idx,t=low_subset(y,.3);half,_=balanced_indices(y,seed=20261116)
    resized=scale_base(x,idx,t);changed,stats=mlp_stroke(resized,.28,idx,half);main[idx]=process(changed,'com',20)
    return main,{'n_aug':len(idx),**stats}
def train_features(name,off,wr):
    x=off['X_train'];y=off['y_train'];combo=name.endswith('Combined');more={}
    if name=='MLP_Normal':
        Y=np.asarray(y,dtype=np.int32);out=process(x,'com',20);idx,t=nested_subset(Y,.8);half,_=balanced_indices(Y,seed=20261116)
        resized=scale_base(x,idx,t);changed,stats=mlp_stroke(resized,.14,idx,half);out[idx]=process(changed,'com',20);assert len(idx)==61440
        return out,Y,{'augmented_rows':len(idx),**stats}
    if name in ('HGB_Normal','ExtraTrees_Normal'):
        a,meta=official_only_30(off);return et_extract(a,'pixels+hog'),y,meta
    if name.startswith(('MLP','HGB','ExtraTrees')):
        a,b,stats,_=prepareraw(off,wr,'scale30_stroke28')
        if name.startswith(('HGB','ExtraTrees')):
            a=et_extract(a,'pixels+hog');b={w:et_extract(b[w],'pixels+hog') for w in W}
        a=np.concatenate([a]+[b[w] for w in W]);y=np.concatenate([y]+[wr[w]['y'] for w in W])
        return a,np.asarray(y,dtype=np.int32),stats
    if name.startswith('KNN'):a,y=assemble_training(x,y,BatchedCoM())
    else:a=hybrid(x)
    if not combo:return a,y,more
    raw=np.concatenate([wr[w]['X'] for w in W]);ry=np.concatenate([wr[w]['y'] for w in W]);wid=np.repeat(np.asarray(W),640)
    ix=balanced_nested_indices(wid,ry,seed=HAND_SEED);allidx=ix[100];orig=hybrid(raw);aug=hand_aug(raw,allidx,HAND_SEED+100);ay=ry[allidx]
    if name.startswith('KNN'):
        q=ix[25];r=np.setdiff1d(ix[50],ix[25],assume_unique=True);assert len(q)==len(r)==len(ry)//4
        assert np.array_equal(allidx,np.arange(len(ry)))
        a=np.concatenate([a,orig,aug,orig[q],aug[r]]);y=np.concatenate([y,ry,ay,ry[q],ay[r]])
        more={'handmade_original':len(ry),'handmade_augmented':len(ry),'handmade_original_repeat':len(q),'handmade_augmented_repeat':len(r)}
    else:
        rng=np.random.default_rng(TARGET_SEED);target=[]
        for w in W:
            for c in TARGET_CLASSES:
                pool=np.flatnonzero((wid==w)&(ry==c));assert len(pool)==40;target.append(rng.permutation(pool)[:20])
        target=np.sort(np.concatenate(target));assert len(target)==160
        tg=hand_aug(raw,target,TARGET_SEED+1000);a=np.concatenate([a,orig,aug,tg]);y=np.concatenate([y,ry,ay,ry[target]])
        more={'handmade_original':len(ry),'handmade_augmented':len(ry),'handmade_targeted':len(target)}
    return a,y,more
def main(name,out):
    assert name in MODELS and sklearn.__version__=='1.8.0'
    out.mkdir(parents=True,exist_ok=False)
    with threadpool_limits(limits=2):
        off,wr=read_training();model=make_model(name)
        t=time.perf_counter();a,y,meta=train_features(name,off,wr);prep=time.perf_counter()-t
        assert a.shape==(COUNTS[name],784 if name.startswith('MLP') else 1108) and y.shape==(COUNTS[name],)
        print('FIT START',name,a.shape,flush=True);t=time.perf_counter()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always');model.fit(a,y)
        fit=time.perf_counter()-t;del a,y;gc.collect()
        path=out/'model.joblib';probe=feature(name,off['X_train'][:16]);ref=model.predict(probe)
        joblib.dump(model,path,compress=3);del model;gc.collect();m=joblib.load(path)
        assert np.array_equal(ref,m.predict(probe)) and np.array_equal(m.classes_,np.arange(16))
        manifest={'model':name,'training_rows':COUNTS[name],'training_prepare_s':prep,'fit_s':fit,'training_time_s':prep+fit,'checkpoint_sha256':digest(path),'checkpoint_bytes':path.stat().st_size,'config':m.get_params(deep=True),'augmentation':meta,'environment':environment(),'warnings':[str(w.message) for w in seen],'validation_loaded_during_fit':False,'original_test_used':False,'professor_test_used':False}
        write_json(out/'fit_manifest.json',manifest)
        pred=m.predict(feature(name,off['X_val']));manifest['official_dev_accuracy']=float(np.mean(pred==off['y_val']));manifest['official_dev_correct']=int(np.sum(pred==off['y_val']))
        reference={'MLP_Normal':18551,'SVC_Normal':18983,'KNN_Normal':18851}
        if name in reference:manifest['reference_correct']=reference[name];manifest['reference_match']=manifest['official_dev_correct']==reference[name]
        write_json(out/'fit_manifest.json',manifest);print('FIT COMPLETE',name,manifest['training_time_s'],manifest['official_dev_accuracy'],flush=True)
if __name__=='__main__':main(sys.argv[1],Path(sys.argv[2]))
