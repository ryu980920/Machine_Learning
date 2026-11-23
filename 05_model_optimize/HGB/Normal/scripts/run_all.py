from pathlib import Path
import sys, time, json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[4]; OUT=Path(__file__).resolve().parent/'out'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(REPO/'05_model_optimize/MLP/Combined/scripts')); sys.path.insert(0,str(REPO/'05_model_optimize/MLP/Normal/scripts'))
from combined_mlp_writer_cv import load_official, load_writers
from exp006_size_com_comparison import process
import exp_cmb005_balanced_scale_stroke as L
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,ConfusionMatrixDisplay
from skimage.feature import hog
import sklearn; print('sklearn',sklearn.__version__,flush=True)
off=load_official(REPO/'02_data/datasets/MNIST_EMNIST_official_split_112000.npz'); wr=load_writers(REPO/'02_data/Hand_made')
W=list(L.WRITERS); LAB='0123456789ABCDEF'; ytr,yva=off['y_train'],off['y_val']
yhm=np.concatenate([wr[w]['y'] for w in W]); whm=np.concatenate([[w]*640 for w in W])
Xtr,tw,_,_=L.prepareraw(off,wr,'scale30_stroke28'); Xva=process(off['X_val'],'com',20); Xhm=np.vstack([tw[w] for w in W])
PAIRS=[('D','0'),('0','D'),('B','8'),('8','B'),('5','F'),('7','9')]
def hgb(**k): return HistGradientBoostingClassifier(max_iter=300,early_stopping=True,validation_fraction=0.1,n_iter_no_change=10,random_state=42,**k)
def sc(y,p,pre): return {f'{pre}_acc':accuracy_score(y,p),f'{pre}_precision':precision_score(y,p,average='macro',zero_division=0),f'{pre}_recall':recall_score(y,p,average='macro',zero_division=0),f'{pre}_f1':f1_score(y,p,average='macro')}
def det(p):
    d=yhm<10; r=dict(hm_digit=accuracy_score(yhm[d],p[d]),hm_letter=accuracy_score(yhm[~d],p[~d]))
    for w in W: r[f'w{w}']=accuracy_score(yhm[whm==w],p[whm==w])
    for a,b in PAIRS: r[f'{a}->{b}']=int(np.sum((yhm==LAB.index(a))&(p==LAB.index(b))))
    return r
res=[]
def save(r): res.append(r); pd.DataFrame(res).to_csv(OUT/'results.csv',index=False); print(r,flush=True)
stage=sys.argv[1]
if stage=='n3':
    m=hgb(); t=time.time(); m.fit(Xtr,ytr); fs=time.time()-t
    pv=m.predict(Xva); ph=m.predict(Xhm)
    save(dict(name='N3',**sc(yva,pv,'val'),**sc(yhm,ph,'hm'),**det(ph),fit_s=round(fs),iters=m.n_iter_))
    np.save(OUT/'ph3.npy',ph)
    def gm(cls=None):
        g=np.zeros(784)
        for it in m._predictors:
            for tr in (it if cls is None else [it[cls]]):
                n=tr.nodes[tr.nodes['is_leaf']==0]; np.add.at(g,n['feature_idx'],n['gain'])
        return g.reshape(28,28)
    maps={'All':gm()}|{c:gm(LAB.index(c)) for c in 'D0B8'}
    np.savez(OUT/'gain.npz',**maps)
    fig,ax=plt.subplots(1,5,figsize=(20,4.5))
    for a,(k,g) in zip(ax,maps.items()): a.imshow(g,cmap='hot'); a.set_title(f'{k} trees'); a.axis('off')
    plt.tight_layout(); plt.savefig(OUT/'D1_gain_heatmap.png',dpi=130)
    st={}
    for k,g in maps.items():
        s=np.sort(g.ravel())[::-1]; st[k]=dict(used=int((g>0).sum()),top20=float(s[:20].sum()/s.sum()))
    json.dump(st,open(OUT/'D1_stats.json','w')); print(st,flush=True)
if stage=='n5':
    def sf(X):
        X=X.reshape(-1,28,28).astype(np.float32); proj=np.hstack([X.sum(1),X.sum(2)])/255.
        return np.hstack([proj,np.stack([hog(i,orientations=9,pixels_per_cell=(7,7),cells_per_block=(2,2)) for i in X])]).astype(np.float32)
    t=time.time(); Ftr=np.hstack([Xtr,sf(Xtr)]); Fva=np.hstack([Xva,sf(Xva)]); Fm={w:np.hstack([tw[w],sf(tw[w])]) for w in W}; Fhm=np.vstack([Fm[w] for w in W]); fe=time.time()-t
    np.save(OUT/'Ftr.npy',Ftr); np.save(OUT/'Fva.npy',Fva); np.savez(OUT/'Fm.npz',**{str(k):v for k,v in Fm.items()})
    m=hgb(); t=time.time(); m.fit(Ftr,ytr); fs=time.time()-t
    t=time.time(); pv=m.predict(Fva); pvs=time.time()-t; t=time.time(); ph=m.predict(Fhm); phs=time.time()-t
    t=time.time(); m.predict(Fva[:1]); one=time.time()-t
    save(dict(name='N5',**sc(yva,pv,'val'),**sc(yhm,ph,'hm'),**det(ph),fit_s=round(fs),iters=m.n_iter_,feature_s=round(fe,1),
              predict_val_s=round(pvs,3),predict_hm_s=round(phs,3),ms_per_img=round(1000*pvs/len(yva),4),single_img_ms=round(1000*one,2)))
    np.save(OUT/'ph5.npy',ph)
    it=np.arange(len(m.train_score_))
    av=[accuracy_score(yva,p) for p in m.staged_predict(Fva)]; ah=[accuracy_score(yhm,p) for p in m.staged_predict(Fhm)]
    lc=pd.DataFrame({'iteration':it,'train_loss':-m.train_score_,'val_loss':-m.validation_score_,'official_val_acc':[np.nan]+av[:len(it)-1],'handmade_acc':[np.nan]+ah[:len(it)-1]})
    lc.to_csv(OUT/'learning_curve.csv',index=False); print(lc.iloc[[1,10,50,100,-1]],flush=True)
if stage.startswith('c2_'):
    w=int(stage[3:]); Ftr=np.load(OUT/'Ftr.npy'); Fm={int(k):v for k,v in np.load(OUT/'Fm.npz').items()}
    o=[x for x in W if x!=w]; X=np.vstack([Ftr]+[Fm[x] for x in o]); y=np.concatenate([ytr]+[wr[x]['y'] for x in o])
    m=hgb(); t=time.time(); m.fit(X,y); fs=time.time()-t; t=time.time(); p=m.predict(Fm[w]); ps=time.time()-t
    np.save(OUT/f'c2_{w}.npy',p); json.dump(dict(writer=w,train_n=len(y),acc=accuracy_score(wr[w]['y'],p),fit_s=round(fs),iters=int(m.n_iter_),predict_s=round(ps,3)),open(OUT/f'c2_{w}.json','w'))
    print('fold',w,accuracy_score(wr[w]['y'],p),round(fs),m.n_iter_,flush=True)
