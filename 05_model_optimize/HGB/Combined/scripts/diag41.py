from pathlib import Path
import sys,time,json,numpy as np
l2=float(sys.argv[1]); seed=int(sys.argv[2]); pat=int(sys.argv[3]); sys.argv=['x','none']; exec(open(Path(__file__).resolve().parent/'run_all.py').read().split("stage=sys.argv[1]")[0])
from sklearn.ensemble import HistGradientBoostingClassifier
Ftr=np.load(OUT/'Ftr.npy'); Fm={int(k):v for k,v in np.load(OUT/'Fm.npz').items()}
keep=np.r_[0:784,840:1164]; w=41; o=[42,43,44]
X=np.vstack([Ftr[:,keep]]+[Fm[x][:,keep] for x in o]); y=np.concatenate([ytr]+[wr[x]['y'] for x in o])
m=HistGradientBoostingClassifier(max_iter=300,early_stopping=True,validation_fraction=0.1,n_iter_no_change=pat,l2_regularization=l2,random_state=seed)
t=time.time(); m.fit(X,y); fs=time.time()-t
acc=[accuracy_score(wr[w]['y'],p) for p in m.staged_predict(Fm[w][:,keep])]
# max abs leaf value per iteration
mx=[max(float(np.abs(t_.nodes['value'][t_.nodes['is_leaf']==1]).max()) for t_ in it) for it in m._predictors]
json.dump(dict(l2=l2,seed=seed,pat=pat,fit_s=round(fs),iters=int(m.n_iter_),val_loss=(-m.validation_score_).tolist(),train_loss=(-m.train_score_).tolist(),acc41=acc,max_leaf=mx),open(OUT/f'diag41_l2{l2}_s{seed}_p{pat}.json','w'))
print('l2',l2,'iters',m.n_iter_,'final41',acc[-1],'best val iter',int(np.argmax(m.validation_score_)),flush=True)
