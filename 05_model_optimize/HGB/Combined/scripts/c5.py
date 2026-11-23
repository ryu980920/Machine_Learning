from pathlib import Path
import sys,time,json,numpy as np
w=int(sys.argv[1]); seed=int(sys.argv[2]); sys.argv=['x','none']; exec(open(Path(__file__).resolve().parent/'run_all.py').read().split("stage=sys.argv[1]")[0])
from sklearn.ensemble import HistGradientBoostingClassifier
Ftr=np.load(OUT/'Ftr.npy'); Fva=np.load(OUT/'Fva.npy'); Fm={int(k):v for k,v in np.load(OUT/'Fm.npz').items()}
keep=np.r_[0:784,840:1164]; o=[x for x in W if x!=w]
X=np.vstack([Ftr[:,keep]]+[Fm[x][:,keep] for x in o]); y=np.concatenate([ytr]+[wr[x]['y'] for x in o])
m=HistGradientBoostingClassifier(max_iter=300,early_stopping=True,validation_fraction=0.1,n_iter_no_change=30,random_state=seed)
t=time.time(); m.fit(X,y); fs=time.time()-t
p=m.predict(Fm[w][:,keep]); va=accuracy_score(yva,m.predict(Fva[:,keep]))
np.save(OUT/f'c5_s{seed}_{w}.npy',p)
r=dict(seed=seed,writer=w,acc=accuracy_score(wr[w]['y'],p),val=va,fit_s=round(fs),iters=int(m.n_iter_),best_iter=int(np.argmax(m.validation_score_)))
json.dump(r,open(OUT/f'c5_s{seed}_{w}.json','w')); print(r,flush=True)
