#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,csv,time
from pathlib import Path
import numpy as np,pandas as pd,joblib
from scipy.ndimage import affine_transform
R=Path('/mnt/data/EXP019_local_stroke_aug')
sys.path.insert(0,str(R));sys.path.insert(0,'/mnt/data/EXP018_cause_analysis')
from train_local_warp import load_official,com,geo,EXPECTED_MODEL,EXPECTED_V3,MODEL_DIR,sha
YV=np.load(R/'y_val.npy');YH=np.load(R/'y_handmade.npy');YN=np.load(R/'y_new438.npy')
with np.load('/mnt/data/handwritten_0-9_A-F_28x28_grayscale(2).npz') as z:
 new480=z['X'].astype(np.float32).reshape(-1,784);y480=z['y'].astype(int)
 new480=com(new480)
Xbase={'officialval':np.load(R/'X_officialval_com.npy',mmap_mode='r'),
       'handmade2560':np.load(R/'X_handmade_com.npy',mmap_mode='r'),
       'new438':np.load(R/'X_new438_com.npy',mmap_mode='r'),
       'new480':new480}
YS={'officialval':YV,'handmade2560':YH,'new438':YN,'new480':y480}
models={}
for variant in ('baseline','B1_top','B2_bottom'):
 for seed in (42,43,44):
  path=(MODEL_DIR/f'seed{seed}.joblib') if variant=='baseline' else (R/'models'/variant/f'seed{seed}.joblib')
  if not path.exists():print('MISSING',variant,seed,path);continue
  if variant=='baseline':assert sha(path)==EXPECTED_MODEL[seed]
  models[variant,seed]=joblib.load(path)
print('MODELS_READY',list(models),flush=True)
results=[];pairs=[];classrows=[];preds={}
for (variant,seed),m in models.items():
 for ds,X in Xbase.items():
  y=YS[ds]
  p=m.predict(X).astype(int)
  preds[variant,seed,ds]=p
  g={'all':np.ones(len(y),bool),'digits':y<10,'letters':y>=10,'7':y==7,'9':y==9}
  for nm,ids in g.items():
   results.append({'variant':variant,'seed':seed,'dataset':ds,'group':nm,'n':int(ids.sum()),'correct':int((p[ids]==y[ids]).sum()),'accuracy':float((p[ids]==y[ids]).mean())})
  for t,pred in [(7,9),(9,7),(7,7),(9,9)]:
   mt=y==t;pairs.append({'variant':variant,'seed':seed,'dataset':ds,'truth':t,'prediction':pred,'n_true':int(mt.sum()),'n_pred':int(((y==t)&(p==pred)).sum())})
  for t in range(16):
   mt=y==t
   classrows.append({'variant':variant,'seed':seed,'dataset':ds,'class':str(t) if t<10 else 'ABCDEF'[t-10],'n':int(mt.sum()),'correct':int(((y==t)&(p==t)).sum()),'accuracy':float(((y==t)&(p==t)).sum()/mt.sum())})

pd.DataFrame(results).to_csv(R/'all_3seeds_dataset_group_metrics.csv',index=False)
pd.DataFrame(pairs).to_csv(R/'all_3seeds_7to9_9to7_pairs.csv',index=False)
pd.DataFrame(classrows).to_csv(R/'all_3seeds_classwise_metrics.csv',index=False)
Rdf=pd.DataFrame(results);P=pd.DataFrame(pairs)
print('\nAVERAGE_ACC',flush=True)
print((Rdf.groupby(['dataset','variant','group'])['accuracy'].mean().mul(100).round(4)).to_string(),flush=True)
print('\nPAIR_COUNTS_MEAN',flush=True)
print((P.groupby(['dataset','variant','truth','prediction'])['n_pred'].mean().round(3)).to_string(),flush=True)

# Paired image transitions, seed-wise, compare to original baseline predictions
transition=[]
for v in ('B1_top','B2_bottom'):
 for ds,y in YS.items():
  for s in (42,43,44):
   if (v,s) not in models:continue
   p0=preds['baseline',s,ds];p1=preds[v,s,ds]
   for group,sel in [('all',np.ones(len(y),bool)),('7',y==7),('9',y==9),('digits',y<10),('letters',y>=10)]:
    a=p0[sel]==y[sel];b=p1[sel]==y[sel]
    transition.append({'dataset':ds,'variant':v,'seed':s,'group':group,'n':int(sel.sum()),'repaired':int((~a & b).sum()),'induced_error':int((a & ~b).sum()),'net':int(b.sum()-a.sum()),'flip_rate':float((p0[sel]!=p1[sel]).mean())})
pd.DataFrame(transition).to_csv(R/'all_3seeds_paired_transitions.csv',index=False)

# Reproduce official validation 0.8 / 1.2 scale response gate from original GitHub EXP017.
print('PREP_VAL_SCALES',flush=True)
xtr,ytr,raw,yval=load_official()
assert np.array_equal(yval,YV)
xraw=raw.reshape(-1,28,28)
long,bx,by=geo(xraw)
scale_sets={}
for scale in (.8,1.2):
 xscaled=np.empty_like(xraw)
 inv=1/scale
 for j,im in enumerate(xraw):
  xscaled[j]=np.clip(affine_transform(im,np.eye(2)*inv,offset=(by[j]*(1-inv),bx[j]*(1-inv)),output_shape=(28,28),order=1,mode='constant',cval=0,prefilter=False),0,255)
 scale_sets[scale]=com(xscaled)
 print('SCALE_PREP',scale,'ready',flush=True)

gaterows=[]
for (variant,seed),model in models.items():
 base=model.predict_proba(Xbase['officialval']).astype(np.float64)
 orig_fam=base[:,10:].sum(axis=1)>=.5
 for scale,arr in scale_sets.items():
  prob=model.predict_proba(arr).astype(np.float64)
  family=prob[:,10:].sum(axis=1)>=.5
  p_let=prob[:,10:].sum(axis=1);base_let=base[:,10:].sum(axis=1)
  ids={'all':np.ones(len(yval),bool),'digits':yval<10,'letters':yval>=10,'8':yval==8,'B':yval==11,'D':yval==13}
  ids.update({str(i) if i<10 else 'ABCDEF'[i-10]:yval==i for i in range(16)})
  for key,mask in ids.items():
   gaterows.append({'variant':variant,'seed':seed,'scale':scale,'group':key,'n':int(mask.sum()),'family_flip':float((family[mask]!=orig_fam[mask]).mean()),'mean_abs_delta_p_letter':float(np.abs(p_let[mask]-base_let[mask]).mean()),'accuracy':float((prob.argmax(1)[mask]==yval[mask]).mean())})
 pd.DataFrame(gaterows).to_csv(R/'officialval_08_12_scale_family_flip_gate.csv',index=False)
 print('GATE_COMPLETE',variant,seed,flush=True)

gates=pd.DataFrame(gaterows)
print('GATE_KEY',flush=True)
print(gates[gates.group.isin(['8','B','D','digits','letters'])].groupby(['variant','group','scale'])['family_flip'].mean().round(4).to_string(),flush=True)

report=[]
report.append('# EXP유019 (local-only) — 상단·하단 국소 획 ±1px 증강 검증')
report.append('')
report.append('- Official Train v3 SHA256: `'+EXPECTED_V3+'`')
report.append('- Existing EXP유017 pipeline 80% size (16·18·20·22·24px), stroke 14%, CoM, MLP (256,128), alpha=0.03, LR=0.001, ReLU/Adam, max_iter=200, early_stopping=False')
report.append('- B1: train-only 20% top x-shift ±1px; B2: same 20% bottom x-shift ±1px; label balanced 16 classes, left/right balanced per class, preserve topology thresholds 96/128, no new train rows')
report.append('- Validation and HM never augmented; repeated HM2560/new438 used as diagnostic, not independent test or for tuning.')
report.append('- Baseline = original EXP017 frozen checkpoint, not retrained; B1/B2 freshly fit same 76,800 train rows and same sklearn hyperparameters and seeds.')
report.append('')
report.append('## 3-seed mean accuracies (%)')
report.append('| Dataset | Baseline | B1 top | B2 bottom |')
report.append('|---|---:|---:|---:|')
for ds in ('officialval','handmade2560','new480','new438'):
 vals=[]
 for v in ('baseline','B1_top','B2_bottom'):
  vals.append(Rdf[(Rdf.dataset==ds)&(Rdf.variant==v)&(Rdf.group=='all')].accuracy.mean()*100)
 report.append('| '+ds+' | '+' | '.join(f'{x:.4f}%' for x in vals)+' |')
report.append('')
report.append('## 7→9 and 9→7 (3-seed total, correlated across same original images)')
report.append('| Dataset | Variant | True 7 count | 7→9 / 3-seed | True 9 count | 9→7 / 3-seed |')
report.append('|---|---|---:|---:|---:|---:|')
for ds in ('officialval','handmade2560','new480','new438'):
 for v in ('baseline','B1_top','B2_bottom'):
  sub=P[(P.dataset==ds)&(P.variant==v)]; n7=int(sub[(sub.truth==7)].n_true.iloc[0]);n9=int(sub[(sub.truth==9)].n_true.iloc[0]);
  a=int(sub[(sub.truth==7)&(sub.prediction==9)].n_pred.sum());b=int(sub[(sub.truth==9)&(sub.prediction==7)].n_pred.sum());
  report.append(f'| {ds} | {v} | {n7} | {a}/{3*n7} | {n9} | {b}/{3*n9} |')
report.append('')
report.append('## OfficialVal 0.8/1.2 scale family-flip gate')
for v in ('baseline','B1_top','B2_bottom'):
 if (v,42) not in models:continue
 val=Rdf[(Rdf.dataset=='officialval')&(Rdf.group=='all')&(Rdf.variant==v)].accuracy.mean()*100
 hm=Rdf[(Rdf.dataset=='handmade2560')&(Rdf.group=='all')&(Rdf.variant==v)].accuracy.mean()*100
 p_7_9=P[(P.dataset=='officialval')&(P.variant==v)&(P.truth==7)&(P.prediction==9)].n_pred.sum()
 p_9_7=P[(P.dataset=='officialval')&(P.variant==v)&(P.truth==9)&(P.prediction==7)].n_pred.sum()
 fail=[]
 if val < 96.5365-.3:fail.append('OfficialVal accuracy <-0.3pp')
 if hm < 92.0703-.5:fail.append('HM2560 accuracy <-0.5pp')
 if v!='baseline' and p_9_7>P[(P.dataset=='officialval')&(P.variant=='baseline')&(P.truth==9)&(P.prediction==7)].n_pred.sum(): fail.append('OfficialVal 9→7 increased')
 for cl in ('B','8','D'):
  k=gates[(gates.variant==v)&(gates.group==cl)].groupby('scale').family_flip.mean().max()
  if k>.20:fail.append(f'{cl} scale family flip >20%')
  each=gates[(gates.variant==v)&(gates.group==cl)].family_flip.max()
  if each>.30:fail.append(f'{cl} seed worst family flip >30%')
 for cl,scale in [('digits',1.2),('letters',.8)]:
  part=gates[(gates.variant==v)&(gates.group==cl)&(gates.scale==scale)]
  if part.family_flip.mean()>.10:fail.append(f'{cl} family flip >10%')
  if part.mean_abs_delta_p_letter.mean()>.12:fail.append(f'{cl} Pletter abs change >.12')
 report.append(f'- **{v}**: OfficialVal {val:.4f}%, HM2560 {hm:.4f}%, OfficialVal 7→9 {p_7_9}, 9→7 {p_9_7}; **'+('gates passed' if not fail else 'rejected: '+', '.join(fail))+'**')
report.append('')
report.append('## Conclusion boundaries')
report.append('This is a retrospective controlled training-data intervention. Model performance may vary by seed; all three seeds and pair errors must be reviewed together. This study does not establish the causal shortcut nor demonstrate independent writer generalization. It does not alter GitHub main or release models.')
(R/'EXP019_LocalStrokeAug_3Seed_Study.md').write_text('\n'.join(report)+'\n',encoding='utf8')
print('\n'.join(report[-12:]),flush=True)
print('ALL_SUMMARY_DONE',flush=True)
