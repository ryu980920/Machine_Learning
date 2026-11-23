from pathlib import Path
import sys,time,json,numpy as np,pandas as pd
kind=sys.argv[1]; sys.argv=['x','none']; exec(open(Path(__file__).resolve().parent/'run_all.py').read().split("stage=sys.argv[1]")[0])
from sklearn.ensemble import HistGradientBoostingClassifier
Ftr=np.load(OUT/'Ftr.npy'); Fva=np.load(OUT/'Fva.npy'); Fm={int(k):v for k,v in np.load(OUT/'Fm.npz').items()}; keep=np.r_[0:784,840:1164]
m=HistGradientBoostingClassifier(max_iter=300,early_stopping=True,validation_fraction=0.1,n_iter_no_change=10,l2_regularization=1.0,random_state=42)
if kind=='n9':
    X,y=Ftr[:,keep],ytr; E=np.vstack([Fm[w] for w in W])[:,keep]; ye=yhm
else:
    o=[41,42,43]; X=np.vstack([Ftr[:,keep]]+[Fm[x][:,keep] for x in o]); y=np.concatenate([ytr]+[wr[x]['y'] for x in o]); E=Fm[44][:,keep]; ye=wr[44]['y']
t=time.time(); m.fit(X,y); fs=time.time()-t
B=Fva[:,keep]; t=time.time(); pv=m.predict(B); pvs=time.time()-t; t=time.time(); pe=m.predict(E); pes=time.time()-t
it=np.arange(len(m.train_score_))
av=[accuracy_score(yva,p) for p in m.staged_predict(B)]; ae=[accuracy_score(ye,p) for p in m.staged_predict(E)]
mx=[max(float(np.abs(t_.nodes['value'][t_.nodes['is_leaf']==1]).max()) for t_ in i_) for i_ in m._predictors]
pd.DataFrame({'iteration':it,'train_loss':-m.train_score_,'val_loss':-m.validation_score_,'official_val_acc':[np.nan]+av[:len(it)-1],'eval_acc':[np.nan]+ae[:len(it)-1],'max_leaf':[np.nan]+mx[:len(it)-1]}).to_csv(OUT/f'lc_{kind}.csv',index=False)
json.dump(dict(fit_s=round(fs),iters=int(m.n_iter_),val=accuracy_score(yva,pv),eval=accuracy_score(ye,pe),pred_val_s=round(pvs,3),pred_eval_s=round(pes,3),ms_per_img=round(1000*pvs/len(yva),4)),open(OUT/f'lc_{kind}.json','w')); print('ok',flush=True)
