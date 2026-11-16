#!/usr/bin/env python3
"""EXP019 (local-only): Official-v3 train-only upper/lower +/-1px stroke warp.

Reproduce EXP017 80%-size, 14%-stroke + CoM training inputs before adding B1/B2.
Hold out OfficialVal; unchanged official Hand-made 2560 and new438 only evaluated.
No GitHub write, no original Test or professor Test access.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse, csv, hashlib, json, sys, time, warnings
from pathlib import Path
import numpy as np
import cv2, joblib
from scipy.ndimage import affine_transform, map_coordinates
from skimage.measure import euler_number
from scipy.ndimage import label
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

ROOT=Path('/mnt/data/EXP019_local_stroke_aug')
V3=ROOT/'MNIST_EMNIST_official_split_112000.npz'
WRITER_DIR=Path('/mnt/data/exp018_3method_evaluation/official_github')
MODEL_DIR=Path('/mnt/data/EXP018_cause_analysis/models/final')
CAS=Path('/mnt/data/EXP018_cause_analysis')
sys.path.insert(0,str(CAS))
from pixel_transform import PixelScaler255
EXPECTED_V3='0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879'
WRITER_HASH={41:'0a829172a72787c770139c73de34cc8b751181829ff6fe08ec260247b9be5a2f',42:'f364fb453d05b4fe2a705c2b8a80683b404608c7c257d664e3ec7ac86932d4c5',43:'10db20997c727c1bc1f1299f8e89626fadd3ec9618468d6759c86b0d2ddc4a66'}
WRITER_HASH[44]='0e266a36e7b29edfcca0514836b5c383d13939948b4e335d70f61bb40bdc084d'
EXPECTED_MODEL={42:'e08eab163cc2db86b03338185f1b0b4f1416a074572597f2b692adec85a0d190',43:'f47e4cc2a97af98e18e5bfdca9b2621acac0f9c303bcd1458fe270fb58e9f931',44:'120cdf9f48691d3e75016379fb62a3cbf3f66d57ffcb7c4c59902a9639795413'}
KERNEL=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
TARGETS=(16,18,20,22,24)
AUG_SEED=20261116


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def load_official():
 assert sha(V3)==EXPECTED_V3
 with np.load(V3,allow_pickle=False) as z:
  x=z['X_train']; y=z['y_train'].astype(np.int32)
  src=z['train_source_index'].astype(np.int64)
  spl=z['train_source_split'].astype(np.int64)
  aug=z['train_is_augmented']
 assert x.shape==(96000,28,28) and int(aug.sum())==5349
 groups=y.astype(np.int64)*1_000_000_000+spl*100_000_000+src
 tr,va=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(x,y,groups))
 assert (len(tr),len(va))==(76800,19200)
 assert len(np.intersect1d(groups[tr],groups[va]))==0
 return x[tr].astype(np.float32).reshape(-1,784),y[tr],x[va].astype(np.float32).reshape(-1,784),y[va]

def load_handmade():
 X=[];y=[];writers=[]
 for writer in (41,42,43,44):
  p=WRITER_DIR/f'ml26_{writer}.npz';assert sha(p)==WRITER_HASH[writer],writer
  with np.load(p) as z:
   x=z['img']; lab=z['label_id'];labels=z['label']
  assert x.shape==(640,28,28) and (np.bincount(lab,minlength=16)==40).all()
  assert np.array_equal(labels,np.array(list('0123456789ABCDEF'))[lab])
  X.append(x.astype(np.float32).reshape(-1,784));y.append(lab.astype(np.int32));writers.extend([writer]*640)
 return np.concatenate(X),np.concatenate(y),np.array(writers)

def geo(x):
 a=np.asarray(x,dtype=np.float32).reshape(-1,28,28)
 fg=a>=32;assert fg.any((1,2)).all()
 sx=fg.any(axis=1);sy=fg.any(axis=2)
 xmin=sx.argmax(axis=1);xmax=27-sx[:,::-1].argmax(axis=1)
 ymin=sy.argmax(axis=1);ymax=27-sy[:,::-1].argmax(axis=1)
 w=xmax-xmin+1;h=ymax-ymin+1
 return np.maximum(w,h),(xmax+xmin)/2.0,(ymax+ymin)/2.0

def resize_by_targets(raw,ids,target_map):
 from scipy.ndimage import affine_transform
 a=np.asarray(raw,dtype=np.float32).reshape(-1,28,28)
 long,bx,by=geo(a)
 resized=np.empty((len(ids),784),dtype=np.float32)
 for desired in TARGETS:
  at=np.flatnonzero(target_map[ids]==desired)
  for pos in at:
   ix=ids[pos]; inv=float(long[ix])/desired
   center=(by[ix],bx[ix]);offset=(center[0]*(1-inv),center[1]*(1-inv))
   im=affine_transform(a[ix],np.eye(2)*inv,offset=offset,output_shape=(28,28),order=1,mode='constant',cval=0,prefilter=False)
   resized[pos]=np.clip(im,0,255).reshape(-1)
 return resized

def com(X):
 a=np.asarray(X,dtype=np.float32).reshape(-1,28,28)
 mask=a>=32;valid=mask.any((1,2));assert valid.all()
 wt=np.where(mask,a,0).astype(np.float64,copy=False)
 mass=wt.sum(axis=(1,2))
 cx=(wt*np.arange(28,dtype=float)[None,None,:]).sum(axis=(1,2))/mass
 cy=(wt*np.arange(28,dtype=float)[None,:,None]).sum(axis=(1,2))/mass
 dx=np.rint(13.5-cx).astype(np.int32);dy=np.rint(13.5-cy).astype(np.int32)
 out=np.zeros_like(a)
 for shiftx,shifty in np.unique(np.column_stack([dx,dy]),axis=0):
  maskidx=np.flatnonzero((dx==shiftx)&(dy==shifty));sx=int(shiftx);sy=int(shifty)
  ya,yb=max(0,-sy),min(28,28-sy);xa,xb=max(0,-sx),min(28,28-sx)
  if ya<yb and xa<xb:out[maskidx,ya+sy:yb+sy,xa+sx:xb+sx]=a[maskidx,ya:yb,xa:xb]
 return out.reshape(-1,784)

def balanced_indices(y,seed=AUG_SEED):
 rng=np.random.default_rng(seed); selections=[];targets=np.zeros(len(y),np.int32)
 for cl in range(16):
  orig=np.flatnonzero(y==cl); n_aug=(len(orig)//10)*5
  sel=np.sort(rng.choice(orig,size=n_aug,replace=False))
  choices=np.tile(np.array(TARGETS,dtype=np.int32),len(sel)//5);rng.shuffle(choices)
  targets[sel]=choices;selections.extend(sel.tolist())
 ids=np.array(sorted(selections),dtype=np.int32)
 return ids,targets

def nested_subset(y,ratio=.8):
 assert ratio==.8
 half,half_target=balanced_indices(y,AUG_SEED)
 targets=half_target.copy();selected=list(half.tolist())
 for klass in range(16):
  allcls=np.flatnonzero(y==klass);prev=half[y[half]==klass]
  rem=np.setdiff1d(allcls,prev,assume_unique=True)
  rng=np.random.default_rng(AUG_SEED+791*klass+29);rem=rng.permutation(rem)
  offset=(klass*3)%5
  extra_targets=np.array([TARGETS[(j+offset)%5] for j in range(len(rem))],dtype=np.int32)
  targets[rem]=extra_targets
  desired=int(np.rint(len(allcls)*ratio));howmany=desired-len(prev)
  selected.extend(rem[:howmany].tolist())
 ids=np.array(sorted(selected),dtype=np.int32)
 assert len(ids)==61440
 for klass in range(16):
  counts=[int(np.sum((y[ids]==klass)&(targets[ids]==size))) for size in TARGETS]
  assert max(counts)-min(counts)<=1,(klass,counts)
 return ids,targets

def topology_stroke(img,threshold=128):
 mask=np.asarray(img)>=threshold
 return int(label(mask,structure=np.ones((3,3)))[1]),int(euler_number(mask,connectivity=2))

def stroke_jitter(images,strength,ids,ids_half):
 x=np.asarray(images,dtype=np.float32).reshape(-1,28,28).copy()
 ids_half=np.asarray(ids_half,dtype=np.int32)
 half_rank=np.full(76800,-1,dtype=np.int32);half_rank[ids_half]=np.arange(len(ids_half),dtype=np.int32)
 parity=np.where(half_rank[ids]>=0,half_rank[ids]%2,ids%2)
 changed=0;rejected=0
 for j,img in enumerate(x):
  proposed=cv2.erode(img,KERNEL) if parity[j]==0 else cv2.dilate(img,KERNEL)
  mixed=np.clip((1-strength)*img+strength*proposed,0,255)
  if topology_stroke(mixed)==topology_stroke(img):x[j]=mixed;changed+=1
  else:rejected+=1
 return x.reshape(-1,784),dict(modified=changed,rejected=rejected)

def build_base_train(x,y):
 t=time.time();print('BUILD_START',len(x),flush=True)
 base=com(x)
 ids,tmap=nested_subset(y,.8)
 ids_half,_=balanced_indices(y,AUG_SEED)
 resized=resize_by_targets(x,ids,tmap);print('RESIZED',time.time()-t,flush=True)
 changed,stats=stroke_jitter(resized,.14,ids,ids_half);print('STROKE',time.time()-t,stats,flush=True)
 base[ids]=com(changed)
 long,_,_=geo(base)
 print('BASE_PREPARED',time.time()-t,'train_n',len(base),'aug_n',len(ids),'mass',int(base.sum()),'label_7_n',(y==7).sum(),flush=True)
 return base,dict(prep_seconds=time.time()-t,base_ink_mass=float(base.sum()),size_stroke_stats=stats,aug_n=len(ids),base_longside_median=float(np.median(long)))

YY,XX=np.meshgrid(np.arange(28,dtype=np.float32),np.arange(28,dtype=np.float32),indexing='ij')
def warp(img,mode,dx):
 im=img.reshape(28,28);yz,xz=np.where(im>=32)
 if not len(yz):return im.copy()
 top,bottom=yz.min(),yz.max();pos=(YY-top)/max(bottom-top,1)
 if mode=='B1_top':wt=np.clip((.55-pos)/.3,0,1)
 elif mode=='B2_bottom':wt=np.clip((pos-.4)/.35,0,1)
 else:raise ValueError(mode)
 target_x=XX-dx*wt
 out=map_coordinates(im,[YY,target_x],order=1,mode='constant',cval=0,prefilter=False).astype(np.float32)
 if out.sum()>0:out=np.clip(out*(im.sum()/out.sum()),0,255)
 return out

def topo_96(img):
 mask=img>=96
 return (int(label(mask)[1]),int(euler_number(mask,connectivity=2)))

def make_augmented_train(base,y,mode,fraction=.20):
 t=time.time();out=base.copy();rec=[];accepted=[]
 for cl in range(16):
  orig=np.flatnonzero(y==cl)
  rand=np.random.default_rng(20261117+cl*147)
  order=rand.permutation(orig)
  per_dir=int(round(len(orig)*fraction/2))
  used=set()
  for direction in (-1,1):
   wanted=per_dir;ok=0;attempt=0
   for ix in order:
    ix=int(ix)
    if ix in used:continue
    attempt+=1
    im=base[ix].reshape(28,28)
    trial=warp(im,mode,direction)
    if topo_96(im)!=topo_96(trial):continue
    # also preserve original EXP017 stroke topology definition at 128
    if topology_stroke(im)!=topology_stroke(trial):continue
    # require at least small real edit and coherent local shape
    if np.mean(np.abs(im-trial))<.05:continue
    out[ix]=com(trial[None,:])[0];used.add(ix);accepted.append(ix);ok+=1
    if ok>=wanted:break
   rec.append(dict(method=mode,label=cl,direction=direction,wanted=wanted,accepted=ok,attempted=attempt,accept_rate=ok/max(attempt,1)))
  print('WARP_PREP',mode,'class',cl,'accepted',len(used),'target',per_dir*2,'elapsed',round(time.time()-t,1),flush=True)
 assert len(accepted)==len(set(accepted))
 print('AUG_DONE',mode,'n_modified',len(accepted),'fraction',len(accepted)/len(y),'prepsec',round(time.time()-t,1),flush=True)
 return out,rec,accepted

def eval_predictions(model,X,y):
 pred=model.predict(X).astype(np.int32)
 groups={'all':np.ones(len(y),bool),'digits':y<10,'letters':y>=10,'7':y==7,'9':y==9}
 d={}
 for key,mask in groups.items():d[key]={'n':int(mask.sum()),'correct':int((pred[mask]==y[mask]).sum()),'accuracy':float((pred[mask]==y[mask]).mean())}
 d['7to9']=int(((y==7)&(pred==9)).sum());d['9to7']=int(((y==9)&(pred==7)).sum())
 d['7to_other']=int(((y==7)&(pred!=7)&(pred!=9)).sum());d['9to_other']=int(((y==9)&(pred!=9)&(pred!=7)).sum())
 return d,pred

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--model',choices=['baseline','B1_top','B2_bottom']);ap.add_argument('--seed',type=int,choices=[42,43,44]);args=ap.parse_args()
 ROOT.mkdir(exist_ok=True,parents=True)
 if args.prepare:
  x,y,xval,yval=load_official()
  Xbase,meta=build_base_train(x,y)
  np.save(ROOT/'X_base80_14_com.npy',Xbase)
  np.save(ROOT/'y_train.npy',y)
  np.save(ROOT/'X_officialval_com.npy',com(xval))
  np.save(ROOT/'y_val.npy',yval)
  original_x,original_y,w=load_handmade()
  np.save(ROOT/'X_handmade_com.npy',com(original_x));np.save(ROOT/'y_handmade.npy',original_y)
  np.save(ROOT/'handmade_writer.npy',w)
  # new 438 diagnostic: do not use to tune
  with np.load('/mnt/data/handwritten_0-9_A-F_28x28_grayscale(2).npz') as z:nx=z['X'];ny=z['y']
  with np.load('/mnt/data/EXP018_품질판정반영_438장/EXP018_user_screened_438.npz') as z:kept=z['original_index']
  np.save(ROOT/'X_new438_com.npy',com(nx[kept]));np.save(ROOT/'y_new438.npy',ny[kept])
  meta.update({'official_v3_sha256':EXPECTED_V3,'handmade_verified_sha256':WRITER_HASH,'new438':'exploratory only, repeated usage','training_performed':False})
  (ROOT/'prep_manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
  print('PREP_SUCCESS',flush=True)
  return
 assert args.model is not None and args.seed is not None
 train=np.load(ROOT/'X_base80_14_com.npy',mmap_mode='r')
 y=np.load(ROOT/'y_train.npy')
 dataset={k:(np.load(ROOT/f'X_{k}_com.npy',mmap_mode='r'),np.load(ROOT/f'y_{k if k!="handmade" else "handmade"}.npy')) for k in []}
 test=[('officialval',np.load(ROOT/'X_officialval_com.npy',mmap_mode='r'),np.load(ROOT/'y_val.npy')),
       ('handmade2560',np.load(ROOT/'X_handmade_com.npy',mmap_mode='r'),np.load(ROOT/'y_handmade.npy')),
       ('new438',np.load(ROOT/'X_new438_com.npy',mmap_mode='r'),np.load(ROOT/'y_new438.npy'))]
 if args.model=='baseline':
  path=MODEL_DIR/f'seed{args.seed}.joblib';assert sha(path)==EXPECTED_MODEL[args.seed]
  model=joblib.load(path);meta={'status':'frozen_exact','model_hash':sha(path),'seed':args.seed,'variant':'A_baseline'}
 else:
  warped,records,accepted=make_augmented_train(train,y,args.model,.20)
  # save once per mode, deterministically selection independent of seed
  manifest=ROOT/f'{args.model}_augment_manifest.json';
  if not manifest.exists():manifest.write_text(json.dumps({'mode':args.model,'records':records,'accepted_count':len(accepted),'accepted_index_sha':hashlib.sha256(np.array(accepted,np.int32).tobytes()).hexdigest(),'actual_applied_fraction':len(accepted)/len(y)},ensure_ascii=False,indent=2))
  model=Pipeline([('scale_0_1',PixelScaler255()),('model',MLPClassifier(hidden_layer_sizes=(256,128),alpha=.03,learning_rate_init=.001,activation='relu',solver='adam',max_iter=200,early_stopping=False,random_state=args.seed))])
  t=time.time()
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always');model.fit(warped,y)
  print('FIT_DONE',args.model,args.seed,'seconds',round(time.time()-t,1),'n_iter',model.named_steps['model'].n_iter_,flush=True)
  outdir=ROOT/'models'/args.model;outdir.mkdir(parents=True,exist_ok=True)
  p=outdir/f'seed{args.seed}.joblib';joblib.dump(model,p,compress=3)
  meta={'status':'train_completed','variant':args.model,'seed':args.seed,'n_iter':int(model.named_steps['model'].n_iter_), 'fit_seconds':time.time()-t,'model_hash':sha(p),'training_source_SHA':EXPECTED_V3,'n_train':len(y),'variant_accepted_n':len(accepted),'warnings':[str(w.message) for w in ws]}
 results=[]
 for dataset_name,X,yt in test:
  d,pred=eval_predictions(model,X,yt)
  print('SCORE',args.model,args.seed,dataset_name,'accuracy',round(100*d['all']['accuracy'],5),'7to9',d['7to9'],'9to7',d['9to7'],flush=True)
  for group in ['all','digits','letters','7','9']:
   results.append({'variant':args.model,'seed':args.seed,'dataset':dataset_name,'group':group,**d[group], '7to9':d['7to9'] if group=='7' else '', '9to7':d['9to7'] if group=='9' else ''})
  if args.model!='baseline':
   outdir=ROOT/'predictions';outdir.mkdir(exist_ok=True)
   np.save(outdir/f'{args.model}_seed{args.seed}_{dataset_name}.npy',pred.astype(np.uint8))
 with (ROOT/f'metrics_{args.model}_{args.seed}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 (ROOT/f'model_{args.model}_{args.seed}_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 print('EXPERIMENT_COMPLETE',args.model,args.seed,flush=True)
if __name__=='__main__':main()
