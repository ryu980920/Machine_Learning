from pathlib import Path
import sys,time,json,numpy as np
kind=sys.argv[1]; seed=int(sys.argv[2]); w=int(sys.argv[3]) if len(sys.argv)>3 else 0; sys.argv=['x','none']
exec(open(Path(__file__).resolve().parent/'run_all.py').read().split("stage=sys.argv[1]")[0])
from sklearn.ensemble import HistGradientBoostingClassifier
Ftr=np.load(OUT/'Ftr.npy'); Fva=np.load(OUT/'Fva.npy'); Fm={int(k):v for k,v in np.load(OUT/'Fm.npz').items()}; keep=np.r_[0:784,840:1164]
def M(): return HistGradientBoostingClassifier(max_iter=300,early_stopping=True,validation_fraction=0.1,n_iter_no_change=10,l2_regularization=1.0,random_state=seed)
if kind=='n9':
    Fhm=np.vstack([Fm[x] for x in W]); m=M(); t=time.time(); m.fit(Ftr[:,keep],ytr); fs=time.time()-t
    pv=m.predict(Fva[:,keep]); ph=m.predict(Fhm[:,keep])
    r=dict(name=f'N9_s{seed}',**sc(yva,pv,'val'),**sc(yhm,ph,'hm'),**det(ph),fit_s=round(fs),iters=int(m.n_iter_))
    np.save(OUT/f'pred_n9_s{seed}.npy',ph); json.dump(r,open(OUT/f'n9_s{seed}.json','w')); print(r,flush=True)
else:
    o=[x for x in W if x!=w]; X=np.vstack([Ftr[:,keep]]+[Fm[x][:,keep] for x in o]); y=np.concatenate([ytr]+[wr[x]['y'] for x in o])
    m=M(); t=time.time(); m.fit(X,y); fs=time.time()-t; p=m.predict(Fm[w][:,keep]); va=accuracy_score(yva,m.predict(Fva[:,keep]))
    np.save(OUT/f'c6_s{seed}_{w}.npy',p); r=dict(seed=seed,writer=w,acc=accuracy_score(wr[w]['y'],p),val=va,fit_s=round(fs),iters=int(m.n_iter_))
    json.dump(r,open(OUT/f'c6_s{seed}_{w}.json','w')); print(r,flush=True)
