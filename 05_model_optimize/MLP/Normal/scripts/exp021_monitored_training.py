#!/usr/bin/env python3
"""EXPO21: observed epoch-by-epoch OfficialVal loss/accuracy. OfficialVal is monitoring only (not used for fit/ES internal decision); no original/teacher Test."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import argparse, pathlib, warnings, csv, json,time,sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from pixel_transform import PixelScaler255

class MonitoredMLP(MLPClassifier):
 def _update_no_improvement_count(self, early_stopping, X, y, sample_weight):
  super()._update_no_improvement_count(early_stopping, X, y, sample_weight)
  i=len(self.loss_curve_)
  pp=self.predict_proba(self._monitor_x)
  p=np.clip(pp[np.arange(len(self._monitor_y)),self._monitor_y],1e-14,1.)
  val_nll=float(-np.mean(np.log(p)))
  val_acc=float(np.mean(pp.argmax(1)==self._monitor_y))
  internal=float(self.validation_scores_[-1]) if early_stopping else None
  info=dict(epoch=i,train_loss=float(self.loss_curve_[-1]),officialval_nll=val_nll,officialval_acc=val_acc,
   internal_val_acc=internal,loss_no_improvement=self._no_improvement_count)
  self._history.append(info)
  self._writer.writerow(info);self._logfile.flush()
  if i==1 or i%10==0:
   print('EPOCH',self.random_state,'ES',int(self.early_stopping),i,'train_loss',round(info['train_loss'],5),'val_nll',round(val_nll,5),'val_acc',round(val_acc,5),'internal',round(internal,5) if internal is not None else '-',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True,choices=[42,43,44]);p.add_argument('--early',action='store_true');a=p.parse_args()
 mode='ES' if a.early else 'NOES'
 xp=np.load(ROOT/'X_train_base.npy',mmap_mode='r')
 yt=np.load(ROOT/'y_train.npy')
 xv=np.load(ROOT/'X_val_com.npy',mmap_mode='r')
 yv=np.load(ROOT/'y_val.npy')
 X=np.asarray(xp,dtype=np.float32).copy()
 Xval=np.asarray(xv,dtype=np.float32).copy()/255.
 assert X.shape==(76800,784) and Xval.shape==(19200,784) and len(yv)==19200
 OUT=pathlib.Path('/tmp/exp021_curves');OUT.mkdir(exist_ok=True,parents=True)
 out=OUT/f'curve_{mode}_seed{a.seed}.csv';summary=OUT/f'fit_{mode}_seed{a.seed}.json'
 m=MonitoredMLP(hidden_layer_sizes=(256,128),alpha=.03,learning_rate_init=.001,activation='relu',solver='adam',max_iter=200,early_stopping=a.early,random_state=a.seed,
    n_iter_no_change=15 if a.early else 10, validation_fraction=.1)
 m._monitor_x=Xval;m._monitor_y=yv;m._history=[]
 with open(out,'w',encoding='utf-8',newline='') as f:
  m._logfile=f;m._writer=csv.DictWriter(f,fieldnames=['epoch','train_loss','officialval_nll','officialval_acc','internal_val_acc','loss_no_improvement']);m._writer.writeheader();f.flush()
  t=time.monotonic();print('START',mode,a.seed,flush=True)
  with warnings.catch_warnings(record=True) as notes:
   warnings.simplefilter('always');model=Pipeline([('scale',PixelScaler255()),('model',m)]);model.fit(X,yt)
  runtime=time.monotonic()-t
 final_pp=m.predict_proba(Xval)
 final_pred=np.asarray(final_pp.argmax(1))
 final_acc=float(np.mean(final_pred==yv))
 final_nll=float(-np.log(np.clip(final_pp[np.arange(len(yv)),yv],1e-14,1)).mean())
 final_7to9=int(((yv==7)&(final_pred==9)).sum())
 final_9to7=int(((yv==9)&(final_pred==7)).sum())
 del m._monitor_x,m._monitor_y,m._writer,m._logfile
 H=m._history
 ixacc=int(np.argmax([r['officialval_acc'] for r in H]));ixloss=int(np.argmin([r['officialval_nll'] for r in H]));
 rec=dict(mode=mode,seed=a.seed,final_restored_acc=final_acc,final_restored_nll=final_nll,final_restored_7to9=final_7to9,final_restored_9to7=final_9to7,n_iter=m.n_iter_,total_seconds=round(runtime,2),best_officialval_acc_epoch=ixacc+1,best_officialval_accuracy=H[ixacc]['officialval_acc'],best_officialval_loss_epoch=ixloss+1,best_officialval_nll=H[ixloss]['officialval_nll'],last_officialval_accuracy=H[-1]['officialval_acc'],last_officialval_nll=H[-1]['officialval_nll'],internal_best_epoch=int(np.argmax(m.validation_scores_))+1 if a.early else None,internal_best_score=m.best_validation_score_ if a.early else None,notes=[str(n.message) for n in notes],officialval_used_in_fit=False,original_test_used=False,professor_test_used=False,source='exact EXP017 80%size+14%stroke CoM; Official v3 76800/19200')
 summary.write_text(json.dumps(rec,ensure_ascii=False,indent=2));print('FINISH',json.dumps(rec),flush=True)
