from pathlib import Path
import sys,time,json,numpy as np,pandas as pd
sys.argv=['x','none']; exec(open(Path(__file__).resolve().parent/'run_all.py').read().split("stage=sys.argv[1]")[0])
Ftr=np.load(OUT/'Ftr.npy'); Fva=np.load(OUT/'Fva.npy'); Fm={int(k):v for k,v in np.load(OUT/'Fm.npz').items()}
keep=np.r_[0:784,840:1164]; w=44; o=[41,42,43]
X=np.vstack([Ftr[:,keep]]+[Fm[x][:,keep] for x in o]); y=np.concatenate([ytr]+[wr[x]['y'] for x in o])
m=hgb(); t=time.time(); m.fit(X,y); fs=time.time()-t
t=time.time(); p=m.predict(Fm[w][:,keep]); ps=time.time()-t
t=time.time(); pv=m.predict(Fva[:,keep]); pvs=time.time()-t
it=np.arange(len(m.train_score_))
ah=[accuracy_score(wr[w]['y'],q) for q in m.staged_predict(Fm[w][:,keep])]
av=[accuracy_score(yva,q) for q in m.staged_predict(Fva[:,keep])]
pd.DataFrame({'iteration':it,'train_loss':-m.train_score_,'val_loss':-m.validation_score_,'official_val_acc':[np.nan]+av[:len(it)-1],'heldout_acc':[np.nan]+ah[:len(it)-1]}).to_csv(OUT/'lc_c3_44.csv',index=False)
json.dump(dict(fit_s=round(fs),iters=int(m.n_iter_),acc=accuracy_score(wr[w]['y'],p),pred_hm_s=round(ps,3),pred_val_s=round(pvs,3),val=accuracy_score(yva,pv)),open(OUT/'lc_c3_44.json','w')); print('ok',flush=True)
