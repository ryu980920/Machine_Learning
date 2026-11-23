from pathlib import Path
import sys,time,json,numpy as np
stage=sys.argv[1]; sys.argv=['x','none']; exec(open(Path(__file__).resolve().parent/'run_all.py').read().split("stage=sys.argv[1]")[0])
from sklearn.ensemble import HistGradientBoostingClassifier
def H(seed): return HistGradientBoostingClassifier(max_iter=300,early_stopping=True,validation_fraction=0.1,n_iter_no_change=10,random_state=seed)
Ftr=np.load(OUT/'Ftr.npy'); Fva=np.load(OUT/'Fva.npy'); Fm={int(k):v for k,v in np.load(OUT/'Fm.npz').items()}; Fhm=np.vstack([Fm[w] for w in W])
cols={'pix':slice(0,784),'proj':slice(784,840),'hog':slice(840,1164)}
def sel(X,parts): return np.hstack([X[:,cols[p]] for p in parts])
kind,seed=stage.split('_'); seed=int(seed)
if kind=='n3': A,B,C=Xtr,Xva,Xhm
elif kind=='n5': A,B,C=Ftr,Fva,Fhm
elif kind=='proj': A,B,C=[sel(X,['pix','proj']) for X in (Ftr,Fva,Fhm)]
elif kind=='hog': A,B,C=[sel(X,['pix','hog']) for X in (Ftr,Fva,Fhm)]
m=H(seed); t=time.time(); m.fit(A,ytr); fs=time.time()-t
pv=m.predict(B); ph=m.predict(C)
r=dict(name=f'{kind}_s{seed}',n_feat=A.shape[1],**sc(yva,pv,'val'),**sc(yhm,ph,'hm'),**det(ph),fit_s=round(fs),iters=int(m.n_iter_))
json.dump(r,open(OUT/f'more_{stage}.json','w')); np.save(OUT/f'pred_{stage}.npy',ph); print(r,flush=True)
