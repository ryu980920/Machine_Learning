
import os,sys,time,json,csv,hashlib,traceback
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='2'
import numpy as np,joblib,pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;RT=ROOT/'runtime';OUT=ROOT/'02_SVC'/'full_runs';OUT.mkdir(exist_ok=True)
L='0123456789ABCDEF';base=json.loads((ROOT/'EXPERIMENT_PROTOCOL.json').read_text())['baseline']['SVC']
baseline=np.load(ROOT/'00_baseline/SVC_Combined/reproduced_pred.npy')
x=np.load(RT/'svc_X.npy',mmap_mode='r');y=np.load(RT/'svc_y.npy')
v=np.load(RT/'dev_X.npy');vy=np.load(RT/'dev_y.npy');o=np.load(RT/'official_X.npy');oy=np.load(RT/'official_y.npy')
assert x.shape==(82080,1108) and len(vy)==1280 and len(oy)==19200
def fast_svc_ovo_predict(model, data, chunk=128):
    sv=np.asarray(model.support_vectors_,dtype=np.float64)
    norms=np.einsum('ij,ij->i',sv,sv)
    starts=np.r_[0,np.cumsum(model.n_support_)]
    ncl=len(model.classes_)
    ret=[];min_abs=np.inf
    for lo in range(0,len(data),chunk):
        z0=np.asarray(data[lo:lo+chunk],dtype=np.float64)
        sq=np.einsum('ij,ij->i',z0,z0)[:,None]+norms[None,:]-2*z0@sv.T
        np.maximum(sq,0,out=sq);sq*=-model._gamma;np.exp(sq,out=sq)
        votes=np.zeros((len(z0),ncl),np.int16);idx=0
        for i in range(ncl):
            si=slice(starts[i],starts[i+1])
            for j in range(i+1,ncl):
                sj=slice(starts[j],starts[j+1])
                dv0=sq[:,si]@model.dual_coef_[j-1,si]+sq[:,sj]@model.dual_coef_[i,sj]+model.intercept_[idx]
                min_abs=min(min_abs,float(np.min(np.abs(dv0))))
                votes[:,i]+=dv0>0;votes[:,j]+=dv0<=0
                idx+=1
        ret.append(model.classes_[votes.argmax(axis=1)])
    return np.concatenate(ret),min_abs


def run(name,C=1.,gm=1.,hw=1.,tw=1.):
    target=OUT/name;target.mkdir(exist_ok=True)
    if (target/'metrics.json').exists():return json.loads((target/'metrics.json').read_text())
    params={'condition':name,'C':C,'gamma_multiplier':gm,'handmade_weight':hw,'target_weight':tw,'training_rows':len(y),'seed':42,'official_train_only':True,'new_validation_used_in_fit':False}
    (target/'config.json').write_text(json.dumps(params,indent=2))
    weights=np.ones(len(y));weights[76800:]=hw;weights[76800:][np.isin(y[76800:],[11,13,12])]*=tw
    gamma=float(gm/(x.shape[1]*x.var()))
    m=SVC(C=C,gamma=gamma,cache_size=200)
    t=time.perf_counter();print('START',name,flush=True)
    m.fit(x,y,sample_weight=weights)
    fit=time.perf_counter()-t
    print('FIT_DONE',name,fit,flush=True)
    joblib.dump(m,target/'model.joblib',compress=3)
    pv=m.predict(v)
    np.save(target/'pred_dev.npy',pv)
    pp,rr,ff,_=precision_recall_fscore_support(vy,pv,labels=np.arange(16),zero_division=0)
    met={**params,'gamma':gamma,'fit_s':fit,'correct':int((pv==vy).sum()),'accuracy':float(np.mean(pv==vy)),'macro_precision':float(pp.mean()),'macro_f1':float(ff.mean()),'corrected':int(((baseline!=vy)&(pv==vy)).sum()),'regressed':int(((baseline==vy)&(pv!=vy)).sum()),'recall':dict(zip(L,rr.tolist())),'timing_note':'fit time only; simultaneous separate-core experiment may influence time, not final matched benchmark'}
    (target/'dev_metrics.json').write_text(json.dumps(met,indent=2))
    print('DEV_DONE',name,met['correct'],met['macro_f1'],flush=True)
    # Gate on development accuracy and precision, F1 before expensive official evaluation
    if met['correct']>base['correct'] and met['macro_precision']>=base['precision'] and met['macro_f1']>base['f1']:
        op,margin=fast_svc_ovo_predict(m,o);np.save(target/'pred_official.npy',op);met['official_correct']=int((op==oy).sum())
        native_check=m.predict(o[np.arange(0,len(o),30)])
        assert np.array_equal(native_check,op[np.arange(0,len(o),30)])
        check_dev,_=fast_svc_ovo_predict(m,v);assert np.array_equal(check_dev,pv)
        met['official_evaluation_method']='RBF OVO direct formula full 19200, native subset 640 and all 1280 dev predictions matched'
        met['min_abs_pairwise_margin']=margin
        met['native_official_full_completed']=False
        met['official_accuracy']=float(np.mean(op==oy))
        pp0,rr0,ff0,_=precision_recall_fscore_support(vy,baseline,labels=np.arange(16),zero_division=0)
        met['protected_pass']=all(rr[L.index(c)]>=rr0[L.index(c)]-2/80-1e-12 for c in '086')
        met['target_pass']=float(np.mean(rr[[L.index(c) for c in 'BDC']]))>float(np.mean(rr0[[L.index(c) for c in 'BDC']]))
        met['accuracy_guardrails_pass']=met['official_correct']>=base['official_correct']-19 and met['protected_pass'] and met['target_pass']
    else:
        met['official_correct']=None;met['accuracy_guardrails_pass']=False;met['official_not_run_reason']='failed development accuracy/precision/F1 gate'
    met['final_adoption']=False
    with (target/'model.joblib').open('rb') as f:met['checkpoint_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    (target/'metrics.json').write_text(json.dumps(met,indent=2))
    print('DONE',name,met['correct'],met['official_correct'],met['accuracy_guardrails_pass'],flush=True)
    return met
if __name__=='__main__':
    with threadpool_limits(2):
        for spec in [('SW1.5',1.,1.,1.5,1.)]:
            try:run(*spec)
            except Exception:
                traceback.print_exc();raise
