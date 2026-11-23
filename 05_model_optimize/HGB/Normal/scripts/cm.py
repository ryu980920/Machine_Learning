from pathlib import Path
import sys,numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
LAB=list('0123456789ABCDEF')
def plot(y,p,title,out):
    cm=confusion_matrix(y,p,labels=range(16)).astype(float); cm=100*cm/cm.sum(1,keepdims=True)
    fig,ax=plt.subplots(figsize=(8.6,7.6))
    im=ax.imshow(cm,cmap='viridis',vmin=0,vmax=100)
    for i in range(16):
        for j in range(16):
            v=cm[i,j]
            if v>0: ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=6.5,color='black' if v>60 else 'white')
    ax.set_xticks(range(16)); ax.set_xticklabels(LAB); ax.set_yticks(range(16)); ax.set_yticklabels(LAB)
    ax.set_xlabel('Predicted label'); ax.set_ylabel('True label'); ax.set_title(title,fontweight='bold',fontsize=11)
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cb.set_label('Percentage of true class (%)')
    plt.tight_layout(); plt.savefig(out,dpi=150); plt.close()
