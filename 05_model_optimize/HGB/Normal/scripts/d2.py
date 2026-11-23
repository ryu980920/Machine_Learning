from pathlib import Path
import sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[4]; OUT=Path('out')
sys.path.insert(0,str(REPO/'05_model_optimize/MLP/Combined/scripts')); sys.path.insert(0,str(REPO/'05_model_optimize/MLP/Normal/scripts'))
from combined_mlp_writer_cv import load_official, load_writers
from exp006_size_com_comparison import process
import exp_cmb005_balanced_scale_stroke as L
off=load_official(REPO/'02_data/datasets/MNIST_EMNIST_official_split_112000.npz'); wr=load_writers(REPO/'02_data/Hand_made')
W=list(L.WRITERS); LAB='0123456789ABCDEF'; yva=off['y_val']
yhm=np.concatenate([wr[w]['y'] for w in W]); whm=np.concatenate([[w]*640 for w in W])
Xhm=np.vstack([process(wr[w]['X'],'com',20) for w in W]); Xva=process(off['X_val'],'com',20)
ph=np.load(OUT/'ph3.npy')
for tc,wc in [('D','0'),('B','8')]:
    t,w=LAB.index(tc),LAB.index(wc); bad=np.where((yhm==t)&(ph==w))[0]; good=np.where((yhm==t)&(ph==t))[0]
    n=8; fig,ax=plt.subplots(3,n+1,figsize=(2*(n+1),6.6))
    rng=np.random.default_rng(0)
    for r,(idx,ti) in enumerate([(bad,f'{tc}->{wc} wrong (n={len(bad)})'),(good,f'{tc} correct (n={len(good)})')]):
        ax[r,0].imshow(Xhm[idx].mean(0).reshape(28,28),cmap='gray'); ax[r,0].set_title('mean\n'+ti,fontsize=8)
        pick=rng.choice(idx,min(n,len(idx)),replace=False)
        for j,i in enumerate(pick): ax[r,j+1].imshow(Xhm[i].reshape(28,28),cmap='gray'); ax[r,j+1].set_title(f'writer {whm[i]}',fontsize=8)
    d=Xhm[bad].mean(0)-Xhm[good].mean(0)
    ax[2,0].imshow(d.reshape(28,28),cmap='bwr',vmin=-120,vmax=120); ax[2,0].set_title('wrong - correct\n(red: more ink in wrong)',fontsize=8)
    ax[2,1].imshow(Xva[yva==w].mean(0).reshape(28,28),cmap='gray'); ax[2,1].set_title(f'official {wc} mean',fontsize=8)
    ax[2,2].imshow(Xva[yva==t].mean(0).reshape(28,28),cmap='gray'); ax[2,2].set_title(f'official {tc} mean',fontsize=8)
    for a in ax.ravel(): a.axis('off')
    plt.tight_layout(); plt.savefig(OUT/f'D2_{tc}_to_{wc}.png',dpi=110)
    print(tc,wc,len(bad),{x:int(np.sum(whm[bad]==x)) for x in W}, 'good',{x:int(np.sum(whm[good]==x)) for x in W})
