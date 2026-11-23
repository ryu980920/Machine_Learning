from pathlib import Path
import sys,time,json,numpy as np
kind=sys.argv[1]; w=int(sys.argv[2]); seed=int(sys.argv[3]); sys.argv=['x','none']
exec(open(Path(__file__).resolve().parent/'run_all.py').read().split("stage=sys.argv[1]")[0])
from exp015_scale_stroke_grid import nested_subset
from exp016_high_fraction_grid import scale_base,stroke_jitter
from exp014_balanced_augmentation import balanced_indices
from skimage.feature import hog
from sklearn.ensemble import HistGradientBoostingClassifier
def hogf(X): return np.stack([hog(i,orientations=9,pixels_per_cell=(7,7),cells_per_block=(2,2)) for i in X.reshape(-1,28,28).astype(np.float32)]).astype(np.float32)
Ftr=np.load(OUT/'Ftr.npy'); Fva=np.load(OUT/'Fva.npy'); Fm={int(k):v for k,v in np.load(OUT/'Fm.npz').items()}
keep=np.r_[0:784,840:1164]; Ftr=Ftr[:,keep]; Fva=Fva[:,keep]
o=[x for x in W if x!=w]
if kind=='c3':
    Hm=np.vstack([Fm[x][:,keep] for x in o]); aug_info={}
else:
    raw=np.vstack([wr[x]['X'] for x in o]); yy=np.concatenate([wr[x]['y'] for x in o])
    rng=np.random.default_rng(20261116); sel=[]; tmap=np.zeros(len(yy),np.int32)
    for c in range(16):
        idx=rng.permutation(np.flatnonzero(yy==c))[:35]; tmap[idx]=np.repeat([16,18,20,22,24],7); sel+=idx.tolist()
    ids=np.array(sorted(sel),dtype=np.int32); half,_=balanced_indices(yy,seed=20261116)
    a=scale_base(raw,ids,tmap); a,st=stroke_jitter(a,0.28,ids,half)
    Xh=process(raw,'com',20); Xh[ids]=process(a,'com',20)
    Hm=np.hstack([Xh,hogf(Xh)]); aug_info=dict(n_aug=int(len(ids)),**{k:int(v) for k,v in st.items() if isinstance(v,(int,np.integer))})
X=np.vstack([Ftr,Hm]); y=np.concatenate([ytr]+[wr[x]['y'] for x in o])
m=HistGradientBoostingClassifier(max_iter=300,early_stopping=True,validation_fraction=0.1,n_iter_no_change=10,random_state=seed)
t=time.time(); m.fit(X,y); fs=time.time()-t
p=m.predict(Fm[w][:,keep]); va=accuracy_score(yva,m.predict(Fva))
np.save(OUT/f'{kind}_s{seed}_{w}.npy',p)
r=dict(kind=kind,seed=seed,writer=w,acc=accuracy_score(wr[w]['y'],p),val=va,fit_s=round(fs),iters=int(m.n_iter_),**aug_info)
json.dump(r,open(OUT/f'{kind}_s{seed}_{w}.json','w')); print(r,flush=True)
