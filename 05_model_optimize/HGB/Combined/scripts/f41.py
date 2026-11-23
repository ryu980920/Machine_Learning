from pathlib import Path
import sys,time,json,numpy as np
sys.argv=['x','none']; exec(open(Path(__file__).resolve().parent/'run_all.py').read().split("stage=sys.argv[1]")[0])
from sklearn.ensemble import HistGradientBoostingClassifier
Ftr=np.load(OUT/'Ftr.npy'); Fm={int(k):v for k,v in np.load(OUT/'Fm.npz').items()}
o=[42,43,44]; X=np.vstack([Ftr]+[Fm[x] for x in o]); y=np.concatenate([ytr]+[wr[x]['y'] for x in o])
m=HistGradientBoostingClassifier(max_iter=140,early_stopping=False,random_state=42)
t=time.time(); m.fit(X,y); fs=time.time()-t
acc=[accuracy_score(wr[41]['y'],p) for p in m.staged_predict(Fm[41])]
json.dump(dict(fit_s=round(fs),acc_by_iter=acc),open(OUT/'f41_fixed140.json','w')); print('done',acc[52],acc[-1],max(acc),int(np.argmax(acc))+1)
