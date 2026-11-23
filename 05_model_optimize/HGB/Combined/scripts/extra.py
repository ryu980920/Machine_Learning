from pathlib import Path
import sys,time,numpy as np
stage=sys.argv[1]; sys.argv=['x','none']; exec(open(Path(__file__).resolve().parent/'run_all.py').read().split("stage=sys.argv[1]")[0])
if stage=='n1':
    Xr=off['X_train']; m=hgb(); m.fit(Xr,ytr)
    ph=m.predict(np.vstack([wr[w]['X'] for w in W])); np.save(OUT/'ph1.npy',ph); print('N1',accuracy_score(yhm,ph),m.n_iter_,flush=True)
if stage.startswith('c1_'):
    w=int(stage[3:]); o=[x for x in W if x!=w]
    X=np.vstack([Xtr]+[tw[x] for x in o]); y=np.concatenate([ytr]+[wr[x]['y'] for x in o])
    m=hgb(); m.fit(X,y); p=m.predict(tw[w]); np.save(OUT/f'c1_{w}.npy',p); print('C1',w,accuracy_score(wr[w]['y'],p),m.n_iter_,flush=True)
