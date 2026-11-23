from pathlib import Path
import sys,time,json,numpy as np
seed=int(sys.argv[1]); sys.argv=['x','none']; exec(open(Path(__file__).resolve().parent/'run_all.py').read().split("stage=sys.argv[1]")[0])
from sklearn.ensemble import HistGradientBoostingClassifier
Ftr=np.load(OUT/'Ftr.npy'); Fva=np.load(OUT/'Fva.npy'); Fm={int(k):v for k,v in np.load(OUT/'Fm.npz').items()}; Fhm=np.vstack([Fm[w] for w in W])
def build(F):
    pr=F[:,784:840]; tot=pr[:,:28].sum(1,keepdims=True)+1e-6
    return np.hstack([F[:,:784],F[:,840:1164],pr/tot]).astype(np.float32)
A,B,C=build(Ftr),build(Fva),build(Fhm)
m=HistGradientBoostingClassifier(max_iter=300,early_stopping=True,validation_fraction=0.1,n_iter_no_change=10,random_state=seed)
t=time.time(); m.fit(A,ytr); fs=time.time()-t
pv=m.predict(B); ph=m.predict(C)
r=dict(name=f'N8_s{seed}',**sc(yva,pv,'val'),**sc(yhm,ph,'hm'),**det(ph),fit_s=round(fs),iters=int(m.n_iter_))
json.dump(r,open(OUT/f'n8_s{seed}.json','w')); np.save(OUT/f'pred_n8_s{seed}.npy',ph); print(r,flush=True)
