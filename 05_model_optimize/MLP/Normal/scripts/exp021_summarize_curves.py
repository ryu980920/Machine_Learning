#!/usr/bin/env python3
"""Aggregate a fully completed 3-seed EXP021 run. No new model fitting.

Monitoring external OfficialVal each epoch is diagnostic. Early stopping uses
sklearn's internal stratified 10% of pre-augmented OfficialTrain, not OfficialVal.
"""
import os
for env in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[env]='1'
import csv, json,sys,pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

root=pathlib.Path('/mnt/data/EXP021_early_stopping_diagnostic')
tmp=pathlib.Path('/tmp/exp021_curves')
out=root/'final';out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root))
from pixel_transform import PixelScaler255

allrows=[];summ=[];source={}
for seed in (42,43,44):
    for mode in ('NOES','ES'):
        f=tmp/f'curve_{mode}_seed{seed}.csv'
        summfile=tmp/f'fit_{mode}_seed{seed}.json'
        if not f.exists() or not summfile.exists():
            raise FileNotFoundError(f'Incomplete run: {mode}/{seed}: {f}, {summfile}')
        df=pd.read_csv(f)
        rec=json.loads(summfile.read_text())
        assert len(df)==rec['n_iter'], (seed,mode,len(df),rec['n_iter'])
        assert df['epoch'].tolist()==list(range(1,len(df)+1))
        df.insert(0,'mode',mode);df.insert(1,'seed',seed)
        allrows.append(df)
        source[(seed,mode)]=(df,rec)
        best_loss=df.loc[df['officialval_nll'].idxmin()]
        best_acc=df.loc[df['officialval_acc'].idxmax()]
        if mode=='ES':
            restored_epoch=rec['internal_best_epoch']
            restored_acc=rec['final_restored_acc']
            restored_nll=rec['final_restored_nll']
            errors_7to9=rec['final_restored_7to9']
            errors_9to7=rec['final_restored_9to7']
        else:
            restored_epoch=rec['n_iter']
            restored_acc=rec.get('final_restored_acc',rec['last_officialval_accuracy'])
            restored_nll=rec.get('final_restored_nll',rec['last_officialval_nll'])
            errors_7to9=rec.get('final_restored_7to9',np.nan)
            errors_9to7=rec.get('final_restored_9to7',np.nan)
        summ.append(dict(seed=seed,mode=mode,train_rows=(69120 if mode=='ES' else 76800),internal_val_rows=(7680 if mode=='ES' else 0),actual_stopping_epoch=rec['n_iter'],restored_epoch=restored_epoch,reported_officialval_acc=restored_acc,reported_officialval_nll=restored_nll,raw_last_acc=rec['last_officialval_accuracy'],raw_last_nll=rec['last_officialval_nll'],best_officialval_acc_epoch=int(best_acc['epoch']),best_officialval_acc=float(best_acc['officialval_acc']),best_officialval_nll_epoch=int(best_loss['epoch']),best_officialval_nll=float(best_loss['officialval_nll']),internal_best_epoch=rec.get('internal_best_epoch'),internal_best_score=rec.get('internal_best_score'),err_7to9=errors_7to9,err_9to7=errors_9to7,training_time_seconds=rec['total_seconds']))

pd.concat(allrows,ignore_index=True).to_csv(out/'EXP021_epochwise_3seed_learning_curves.csv',index=False)
summary=pd.DataFrame(summ).sort_values(['seed','mode'])
summary.to_csv(out/'EXP021_comparison_by_seed.csv',index=False)

# Grounded comparison to the original frozen EXP017 model weights.
xval=np.load(root/'X_val_com.npy',mmap_mode='r')
yval=np.load(root/'y_val.npy')
verified=[]
for seed in (42,43,44):
    p=root/'frozen_models'/f'seed{seed}.joblib'
    m=joblib.load(p)
    probs=m.predict_proba(xval)
    pred=probs.argmax(1)
    verified.append(dict(seed=seed,officialval_acc=float(np.mean(pred==yval)),officialval_nll=float(-np.log(np.clip(probs[np.arange(len(yval)),yval],1e-14,1)).mean()),err_7to9=int(((yval==7)&(pred==9)).sum()),err_9to7=int(((yval==9)&(pred==7)).sum()),n_iter=m.named_steps['model'].n_iter_))
pd.DataFrame(verified).to_csv(out/'EXP021_original_frozen_model_check.csv',index=False)

fig,ax=plt.subplots(3,3,figsize=(17.4,11.7),sharex='col',constrained_layout=True)
for j,seed in enumerate((42,43,44)):
    for mode,label in (('NOES','No Early Stopping'),('ES','Early Stopping')):
        df,rec=source[(seed,mode)]
        ax[0,j].plot(df.epoch,df.train_loss,label=label,lw=1.6)
        ax[1,j].plot(df.epoch,df.officialval_nll,label=label,lw=1.6)
        ax[2,j].plot(df.epoch,100*df.officialval_acc,label=label,lw=1.6)
        if mode=='ES':
            ax[0,j].axvline(rec['internal_best_epoch'],linestyle='--',alpha=.5)
            ax[1,j].axvline(rec['internal_best_epoch'],linestyle='--',alpha=.5)
            ax[2,j].axvline(rec['internal_best_epoch'],linestyle='--',alpha=.5)
    ax[0,j].set_title(f'Seed {seed}',fontsize=13)
    ax[2,j].set_xlabel('Epoch')
    for i in range(3):ax[i,j].grid(alpha=.2)
ax[0,0].set_ylabel('Training loss\n(includes L2 term)')
ax[1,0].set_ylabel('OfficialVal log loss\n(no L2 term)')
ax[2,0].set_ylabel('OfficialVal accuracy (%)')
ax[0,2].legend(loc='upper right')
fig.suptitle('EXP021 | Epoch-by-epoch Learning Curves (alpha=0.03, 80%-Size + 14%-Stroke + CoM)\nDashed: ES internal best accuracy epoch, restored after stopping',fontsize=16)
fig.savefig(out/'EXP021_Learning_Curves_3Seeds.png',dpi=180,bbox_inches='tight')
plt.close(fig)

# Zoom to post-warmup plateaus, where stopping decisions matter.
fig,ax=plt.subplots(2,3,figsize=(17.4,8.2),constrained_layout=True)
for j,seed in enumerate((42,43,44)):
    for mode,label in (('NOES','Early Stopping OFF'),('ES','Early Stopping ON')):
        df,rec=source[(seed,mode)]
        cutoff=df[df.epoch>=10]
        ax[0,j].plot(cutoff.epoch,cutoff.officialval_nll,label=label)
        ax[1,j].plot(cutoff.epoch,100*cutoff.officialval_acc,label=label)
        if mode=='ES':
            ax[0,j].axvline(rec['internal_best_epoch'],linestyle='--',alpha=.5)
            ax[1,j].axvline(rec['internal_best_epoch'],linestyle='--',alpha=.5)
    noesdf,_=source[(seed,'NOES')]
    ax[0,j].axvline(int(noesdf.loc[noesdf.officialval_nll.idxmin(),'epoch']),linestyle=':',alpha=.55)
    ax[1,j].axvline(int(noesdf.loc[noesdf.officialval_acc.idxmax(),'epoch']),linestyle=':',alpha=.55)
    ax[0,j].set_title(f'Seed {seed}')
    ax[1,j].set_xlabel('Epoch (zoom: >=10)')
    for i in range(2):ax[i,j].grid(alpha=.22)
ax[0,0].set_ylabel('OfficialVal log loss')
ax[1,0].set_ylabel('OfficialVal accuracy (%)')
ax[1,-1].legend(fontsize=9)
fig.suptitle('EXP021 | Validation plateau and stopping point (post-warmup)',fontsize=15)
fig.savefig(out/'EXP021_Validation_Plateau_Zoom_3Seeds.png',dpi=180,bbox_inches='tight')
plt.close(fig)

fig,ax=plt.subplots(1,3,figsize=(16.5,4.8),constrained_layout=True,sharey=True)
for j,seed in enumerate((42,43,44)):
    df,rec=source[(seed,'ES')]
    ax[j].plot(df.epoch,100*df.internal_val_acc,label='Internal validation (used for ES)')
    ax[j].plot(df.epoch,100*df.officialval_acc,label='Official validation (diagnostic)')
    ax[j].axvline(rec['internal_best_epoch'],linestyle='--',alpha=.6)
    ax[j].set_title(f'Seed {seed}: restored epoch {rec["internal_best_epoch"]}')
    ax[j].set_xlabel('Epoch')
    ax[j].grid(alpha=.2)
ax[0].set_ylabel('Accuracy (%)')
ax[-1].legend(fontsize=9)
fig.suptitle('EXP021 | Early-stopping criterion vs external OfficialVal',fontsize=14)
fig.savefig(out/'EXP021_EarlyStopping_Internal_vs_Official_Accuracy.png',dpi=180,bbox_inches='tight')
plt.close(fig)

stats=summary.groupby('mode').agg(officialval_acc_mean=('reported_officialval_acc','mean'),officialval_acc_std=('reported_officialval_acc','std'),niter_mean=('actual_stopping_epoch','mean'),restored_epoch_mean=('restored_epoch','mean')).reset_index()
stats.to_csv(out/'EXP021_3Seed_aggregate.csv',index=False)
readme=[]
readme.append('# EXP021 — Early Stopping Learning Curves (실측 3-Seed)')
readme.append('')
readme.append('## 목적 및 모델 조건')
readme.append('')
readme.append('기존 EXP017 최종 MLP `(256,128)` / alpha=0.03 / learning rate=0.001 / Size80%-Stroke14%+CoM에 대해 Seed42/43/44에서 epoch별 Training Loss, 외부 OfficialVal Log Loss, 외부 OfficialVal Accuracy를 재측정하였다. 기존 frozen model weights는 변경하지 않았다.')
readme.append('')
readme.append('## 실험의 데이터 분할 차이')
readme.append('')
readme.append('- **NOES**: OfficialTrain 76,800 전체로 학습, `early_stopping=False`, `n_iter_no_change=10`, 최대 200 epoch, training-loss 기반 종료.')
readme.append('- **ES**: 동일 공식 Train에 대해 scikit-learn 내부에서 10%(7,680장)를 별도 검증으로 분리, 실제 가중치 학습 69,120장, `early_stopping=True`, `n_iter_no_change=15`, 최대 200 epoch, 내부 검증 **accuracy** 기반 중단 및 최고 score epoch 가중치 복원.')
readme.append('- 외부 OfficialVal 19,200장은 두 모델의 학습/내부 ES 판단에 사용되지 않았지만 **곡선 진단에 매 epoch 조회**하므로, external validation에서 후선정한 최고 epoch는 독립 Test 점수로 사용하거나 바로 배포하지 않는다.')
readme.append('- Original Test/교수님 Test/사용자 손글씨를 학습과 조기 종료 판단에 사용하지 않았다.')
readme.append('')
readme.append('## 관찰 결과')
readme.append('')
readme.append('| Seed | Mode | 실제 종료 | 복원 epoch | OfficialVal(최종) | OfficialVal loss 최저 epoch | OfficialVal accuracy 최고 epoch |')
readme.append('|---|---|---:|---:|---:|---:|---:|')
for x in summ:
    readme.append(f'| {x["seed"]} | {x["mode"]} | {x["actual_stopping_epoch"]} | {x["restored_epoch"]} | {x["reported_officialval_acc"]*100:.4f}% | {x["best_officialval_nll_epoch"]} | {x["best_officialval_acc_epoch"]} |')
readme.append('')
for row in stats.to_dict('records'):
    readme.append(f'- {row["mode"]}: 3 Seed 평균 OfficialVal 정확도 **{row["officialval_acc_mean"]*100:.4f}%**, 종료 epoch 평균 {row["niter_mean"]:.1f}, 복원 epoch 평균 {row["restored_epoch_mean"]:.1f}.')
readme.append('')
readme.append('**중요한 설계 한계:** ES와 NOES는 학습 데이터 수(69,120 vs 76,800)와 patience(15 vs 10)가 달라서, 성능 차이를 **Early Stopping의 독립 인과 효과**라고 단정할 수 없다. ES의 평가 상 이득이 있더라도 Train 분할 영향이 섞여 있다. 보다 엄격한 인과 비교에는 동일한 69,120 Train으로 같은 optimizer trajectory를 실행하여 체크포인트만 다른 epoch에서 선택해야 한다.')
readme.append('')
readme.append('## 실측 모델 선택에 필요한 추가 절차')
readme.append('')
readme.append('현재 ES가 내부 Validation Accuracy 기준으로 중단한 위치가 외부 Validation Loss/Accuracy에 비해 이르거나 늦었는지 곡선에서 확인하되, 조기 종료 정책 변경은 동일 분할/동일 학습 경로 비교와 크기 Shortcut Gate 검증을 마친 뒤에 결정한다.')
readme.append('')
readme.append('## 원본 파일')
readme.append('')
readme.append('- `EXP021_epochwise_3seed_learning_curves.csv`: 6개 실험의 매 epoch 수치')
readme.append('- `EXP021_comparison_by_seed.csv`, `EXP021_3Seed_aggregate.csv`: epoch 및 정확도 비교')
readme.append('- `EXP021_original_frozen_model_check.csv`: 원본 동결 모델과 버전 확인')
readme.append('- `EXP021_Learning_Curves_3Seeds.png`: 3×3 학습 곡선')
readme.append('- `EXP021_EarlyStopping_Internal_vs_Official_Accuracy.png`: 내부 검증 vs 외부 검증')
readme.append('- `EXP021_Validation_Plateau_Zoom_3Seeds.png`: 10 epoch 이후 검증 구간 확대')
(out/'EXP021_Study_Report.md').write_text('\n'.join(readme)+'\n',encoding='utf8')
print('OUTPUTS',*[p.name for p in out.iterdir()],sep='\n')
print(stats.to_string(index=False))
