from pathlib import Path
import sys,time,json,numpy as np
w=int(sys.argv[1]); sys.argv=['x','none']; exec(open(Path(__file__).resolve().parent/'run_all.py').read().split("stage=sys.argv[1]")[0])
Ftr=np.load(OUT/'Ftr.npy'); Fm={int(k):v for k,v in np.load(OUT/'Fm.npz').items()}
keep=np.r_[0:784,840:1164]
o=[x for x in W if x!=w]; X=np.vstack([Ftr[:,keep]]+[Fm[x][:,keep] for x in o]); y=np.concatenate([ytr]+[wr[x]['y'] for x in o])
m=hgb(); t=time.time(); m.fit(X,y); fs=time.time()-t; p=m.predict(Fm[w][:,keep])
np.save(OUT/f'c3_{w}.npy',p); json.dump(dict(writer=w,acc=accuracy_score(wr[w]['y'],p),fit_s=round(fs),iters=int(m.n_iter_)),open(OUT/f'c3_{w}.json','w')); print(w,accuracy_score(wr[w]['y'],p),m.n_iter_,flush=True)
