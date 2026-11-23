from pathlib import Path
import sys,time,json,numpy as np
kind=sys.argv[1]; seed=int(sys.argv[2]); w=int(sys.argv[3]) if len(sys.argv)>3 else 0; sys.argv=['x','none']
exec(open(Path(__file__).resolve().parent/'run_all.py').read().split("stage=sys.argv[1]")[0])
from sklearn.ensemble import HistGradientBoostingClassifier
from skimage.feature import hog
Ftr=np.load(OUT/'Ftr.npy'); Fva=np.load(OUT/'Fva.npy'); Fm={int(k):v for k,v in np.load(OUT/'Fm.npz').items()}; keep=np.r_[0:784,840:1164]
def fine(F):
    X=F[:,:784].reshape(-1,28,28).astype(np.float32)
    return np.stack([hog(i,orientations=9,pixels_per_cell=(4,4),cells_per_block=(1,1)) for i in X]).astype(np.float32)
cache=OUT/'fine_hog.npz'
def feats(F,key):
    base=F[:,keep]
    if not kind.startswith('n11') and not kind.startswith('c8'): return base
    if cache.exists():
        d=np.load(cache)
        if key in d.files: return np.hstack([base,d[key]])
    return np.hstack([base,fine(F)])
if kind in ('n11','c8') and not cache.exists():
    np.savez(cache,tr=fine(Ftr),va=fine(Fva),**{f'w{x}':fine(Fm[x]) for x in W})
mi=500 if kind in ('n10','c7') else 300
def M(): return HistGradientBoostingClassifier(max_iter=mi,early_stopping=True,validation_fraction=0.1,n_iter_no_change=10,l2_regularization=1.0,random_state=seed)
A=feats(Ftr,'tr'); B=feats(Fva,'va'); Hm={x:feats(Fm[x],f'w{x}') for x in W}
if kind in ('n10','n11'):
    m=M(); t=time.time(); m.fit(A,ytr); fs=time.time()-t
    pv=m.predict(B); ph=m.predict(np.vstack([Hm[x] for x in W]))
    r=dict(name=f'{kind}_s{seed}',**sc(yva,pv,'val'),**sc(yhm,ph,'hm'),**det(ph),fit_s=round(fs),iters=int(m.n_iter_),n_feat=A.shape[1])
    np.save(OUT/f'pred_{kind}_s{seed}.npy',ph); json.dump(r,open(OUT/f'{kind}_s{seed}.json','w')); print(r,flush=True)
else:
    o=[x for x in W if x!=w]; X=np.vstack([A]+[Hm[x] for x in o]); y=np.concatenate([ytr]+[wr[x]['y'] for x in o])
    m=M(); t=time.time(); m.fit(X,y); fs=time.time()-t; p=m.predict(Hm[w]); va=accuracy_score(yva,m.predict(B))
    np.save(OUT/f'{kind}_s{seed}_{w}.npy',p); r=dict(seed=seed,writer=w,acc=accuracy_score(wr[w]['y'],p),val=va,fit_s=round(fs),iters=int(m.n_iter_))
    json.dump(r,open(OUT/f'{kind}_s{seed}_{w}.json','w')); print(r,flush=True)
