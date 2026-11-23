from pathlib import Path
import sys,time,json,numpy as np,pandas as pd
sys.argv=['x','none']; exec(open(Path(__file__).resolve().parent/'run_all.py').read().split("stage=sys.argv[1]")[0])
Ftr=np.load(OUT/'Ftr.npy'); Fva=np.load(OUT/'Fva.npy'); Fm={int(k):v for k,v in np.load(OUT/'Fm.npz').items()}; Fhm=np.vstack([Fm[w] for w in W])
keep=np.r_[0:784,840:1164]; A,B,C=Ftr[:,keep],Fva[:,keep],Fhm[:,keep]
m=hgb(); t=time.time(); m.fit(A,ytr); fs=time.time()-t
t=time.time(); pv=m.predict(B); pvs=time.time()-t; t=time.time(); ph=m.predict(C); phs=time.time()-t
t=time.time(); m.predict(B[:1]); one=time.time()-t
it=np.arange(len(m.train_score_))
av=[accuracy_score(yva,p) for p in m.staged_predict(B)]; ah=[accuracy_score(yhm,p) for p in m.staged_predict(C)]
pd.DataFrame({'iteration':it,'train_loss':-m.train_score_,'val_loss':-m.validation_score_,'official_val_acc':[np.nan]+av[:len(it)-1],'handmade_acc':[np.nan]+ah[:len(it)-1]}).to_csv(OUT/'lc7.csv',index=False)
json.dump(dict(fit_s=round(fs),iters=int(m.n_iter_),pred_val_s=round(pvs,3),pred_hm_s=round(phs,3),ms_per_img=round(1000*pvs/len(yva),4),single_ms=round(1000*one,2),hm=accuracy_score(yhm,ph),val=accuracy_score(yva,pv)),open(OUT/'lc7.json','w'))
print('ok',fs,m.n_iter_,flush=True)
