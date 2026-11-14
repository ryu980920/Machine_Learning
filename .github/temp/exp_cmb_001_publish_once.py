from pathlib import Path
import os, subprocess
from zipfile import ZipFile, ZIP_DEFLATED
import csv, io, json, hashlib, shutil, re, sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SRC=Path(os.environ['EXP_CMB_SOURCE_ZIP'])
ROOT=Path(os.environ['GITHUB_WORKSPACE'])/'05_model_optimize'/'MLP'/'Combined'
OUT=ROOT
RESULT=ROOT/'results'/'EXP_CMB_001'
SCRIPT=ROOT/'scripts'
LOG=ROOT/'logs'
for d in [RESULT/'figures',SCRIPT,LOG]:d.mkdir(parents=True,exist_ok=True)

def shasum(content):return hashlib.sha256(content).hexdigest()
def ztext(z,n):return z.read('exp_cmb_001_results/'+n).decode('utf-8-sig')
def parse_csv(z,n):return list(csv.DictReader(io.StringIO(ztext(z,n))))
def fromzip(z,n):return z.read('exp_cmb_001_results/'+n)

with ZipFile(SRC) as z:
  oof=parse_csv(z,'OOF_comparison.csv')
  folds=parse_csv(z,'fold_comparison.csv')
  official=parse_csv(z,'official_development_validation.csv')
  fits=parse_csv(z,'fit_times_and_iterations.csv')
  preds=parse_csv(z,'OOF_predictions.csv')
  src_manifest=json.loads(ztext(z,'training_validation_manifest.json'))
  src_hashes=json.loads(ztext(z,'input_hashes.json'))
  src_summary=json.loads(ztext(z,'run_summary.json'))
  match_fit={(i['model'],i['held_out_writer']):i for i in fits}
  bymodel={i['model']:i for i in oof}
  assert len(preds)==2560
  labels='0123456789ABCDEF'
  tru=np.asarray([labels.index(r['true_label']) for r in preds]); normals=np.asarray([labels.index(r['normal_pred']) for r in preds]); combos=np.asarray([labels.index(r['combined_pred']) for r in preds]); writers=np.asarray([int(r['writer']) for r in preds])
  assert all(np.sum(writers==w)==640 for w in [41,42,43,44]); assert np.array_equal(np.bincount(tru,minlength=16),np.full(16,160))
  for name,pred in [('Normal',normals),('Combined',combos)]:
    assert abs(accuracy_score(tru,pred)-float(bymodel[name]['accuracy']))<1e-12
    assert abs(f1_score(tru,pred,average='macro')-float(bymodel[name]['macro_f1']))<1e-12
    cm=confusion_matrix(tru,pred,labels=list(range(16)))
    cmlist=list(csv.reader(io.StringIO(ztext(z,name.lower()+'_OOF_confusion_matrix.csv'))))
    raw=np.array([[int(x) for x in row[1:]] for row in cmlist[1:]])
    assert np.array_equal(cm,raw)
  n2c=int(np.sum((normals!=tru)&(combos==tru)));c2n=int(np.sum((normals==tru)&(combos!=tru))); assert (n2c,c2n)==(540,62)

  # Controlled summary: 2 OOF + 8 writer-holdout + 5 official-validation rows
  head=['dataset','evaluation','model','held_out_writer','train_writers','n_training_per_fit','n_evaluated','accuracy','macro_f1','accuracy_delta_pp_vs_normal','fit_seconds','n_iterations','final_training_loss','predict_seconds']
  rows=[]
  def emit(dataset,evaluation,model,held,trainwriters,ntr,n,acc,f1,delta='',predtime=''):
    mf=match_fit.get((model,str(held) if held!='' else 'all')) or (match_fit.get((model,'all')) if model=='Normal' else None)
    if mf is None:raise KeyError((model,held))
    rows.append(dict(zip(head,[dataset,evaluation,model,held,trainwriters,ntr,n,acc,f1,delta,mf['fit_seconds'],mf['n_iter'],mf['final_loss'],predtime])))
  for model in ['Normal','Combined']:
    x=bymodel[model]
    mf=match_fit[('Normal','all')] if model=='Normal' else None
    rows.append(dict(zip(head,['handmade','pooled_OOF',model,'','per-fold 3 other writers' if model=='Combined' else 'none',76800 if model=='Normal' else 78720,x['n'],x['accuracy'],x['macro_f1'],100*(float(x['accuracy'])-float(bymodel['Normal']['accuracy'])) if model=='Combined' else 0,mf['fit_seconds'] if mf else '',mf['n_iter'] if mf else '',mf['final_loss'] if mf else '',''])))
  for f in folds:
    held=int(f['held_out_writer']);tw=','.join(f['training_writer_'+str(i)] for i in [1,2,3])
    emit('handmade','writer_holdout','Normal',held,'none',76800,640,f['normal_acc'],f['normal_macro_f1'],0,f['normal_predict_seconds'])
    emit('handmade','writer_holdout','Combined',held,tw,78720,640,f['combined_acc'],f['combined_macro_f1'],100*float(f['combined_minus_normal_accuracy']),f['combined_predict_seconds'])
  for f in official:
    model=f['model'];held= f['held_out_writer'] if model=='Combined' else '';tr= 78720 if model=='Combined' else 76800
    keyheld=held if model=='Combined' else 'all'
    mf=match_fit[(model,str(keyheld))]
    tw=next((',' .join(row['training_writer_'+str(i)] for i in [1,2,3]) for row in folds if row['held_out_writer']==str(held)),'') if model=='Combined' else 'none'
    delta=100*(float(f['accuracy'])-float(official[0]['accuracy'])) if model=='Combined' else 0
    rows.append(dict(zip(head,['official_dev','fixed_validation',model,held,tw,tr,f['n'],f['accuracy'],f['macro_f1'],delta,mf['fit_seconds'],mf['n_iter'],mf['final_loss'],''])))
  assert len(rows)==15
  with (RESULT/'metrics_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=head);writer.writeheader();writer.writerows(rows)
  for name in ['OOF_predictions.csv','loss_curves.csv']:(RESULT/name).write_bytes(fromzip(z,name))
  (RESULT/'official_split_indices.npz').write_bytes(fromzip(z,'official_split_indices.npz'))

  # Models: archive the exact five joblib payloads, with no extra duplicates.
  modelnames=[i.filename for i in z.infolist() if i.filename.startswith('exp_cmb_001_results/trained_models/') and i.filename.endswith('.joblib')]
  assert len(modelnames)==5
  hashes={}
  with ZipFile(RESULT/'trained_models.zip','w',ZIP_DEFLATED,compresslevel=6,allowZip64=True) as zout:
    for n in sorted(modelnames):
      b=z.read(n);stem=n.rsplit('/',1)[-1];hashes[stem]=shasum(b)
      zout.writestr(stem,b,compress_type=ZIP_DEFLATED,compresslevel=6)
  with ZipFile(RESULT/'trained_models.zip') as m:
    assert len(m.namelist())==5
    for k,d in hashes.items(): assert shasum(m.read(k))==d

  # Keep exactly the four approved Viridis PNGs and original loss PNG.
  subprocess.run([sys.executable,str(SCRIPT/'render_cm_viridis.py'),'--predictions',str(RESULT/'OOF_predictions.csv'),'--output-dir',str(RESULT/'figures')],check=True)
  (RESULT/'figures'/'MLP_loss_curves.png').write_bytes(fromzip(z,'MLP_loss_curves.png'))

  with np.load(RESULT/'official_split_indices.npz',allow_pickle=False) as p:
    tr=p['train_idx'];va=p['val_idx'];gi=p['train_group_ids'];gv=p['val_group_ids']
    assert len(tr)==76800 and len(va)==19200
    assert len(np.intersect1d(tr,va))==0 and len(np.intersect1d(gi,gv))==0
    assert np.array_equal(np.sort(np.r_[tr,va]),np.arange(96000))

  # Consolidate input hashes, split manifest, execution summary, render settings, warnings.
  render={'source_matrices':{name:{'sha256':shasum(fromzip(z,name.lower()+'_OOF_confusion_matrix.csv'))} for name in ('Normal','Combined')}}
  ex={
    'experiment_id':'EXP-CMB-001','experiment_date_kst':'2026-11-15',
    'run_status':'success',
    'source_repository':'ryu980920/Machine_Learning','source_branch':'exp/mlp-combined-cv-001',
    'github_actions':{'run_id':43622998640,'job_id':132532593279,'run_url':'https://github.com/ryu980920/Machine_Learning/actions/runs/43622998640','source_artifact_id':14124782428,'artifact_sha256':'53de75b62b17ffd5906d784f2ec6f5b4a0b8b95b07ec9d61456af3b1237f9041'},
    'dataset_sha256':src_hashes,
    'input_data_paths':{'official':'02_data/datasets/MNIST_EMNIST_official_split_112000.npz','handmade_directory':'02_data/Hand_made','handmade_files':[f'ml26_{x}.npz' for x in (41,42,43,44)]},
    'model':src_manifest['model'],'pipeline':'PixelScaler255 (X/255.0) -> MLPClassifier(random_state=42)',
    'mlp_hyperparameters':src_manifest['mlp_hyperparameters'],
    'official_split':src_manifest['official_train_split'],
    'official_split_file':'official_split_indices.npz',
    'handmade_cv':src_manifest['handmade_cv'],
    'folds':src_manifest['folds'],
    'runtime':{k:src_summary[k] for k in ('python_version','numpy_version','scikit_learn_version')},
    'model_fit_warnings':{'Normal':src_summary['normal_warnings'],**{f'Combined heldout {w}':json.loads(ztext(z,f'fold_{w}_warnings.json')) for w in (41,42,43,44)}},
    'models_zip_members_sha256':hashes,
    'evaluations':{'normal_handmade_oof':src_summary['normal_handmade_oof'],'combined_handmade_oof':src_summary['combined_handmade_oof'],'combined_minus_normal_accuracy_pp':src_summary['combined_minus_normal_accuracy_pp'],'normal_official_dev_validation':src_summary['normal_official_dev_validation']},
    'prediction_reconciliation':{'normal_wrong_combined_correct':n2c,'normal_correct_combined_wrong':c2n,'net_additional_correct':n2c-c2n},
    'visualization':{'type':'Viridis count and row-normalized confusion matrices','class_order':labels,'n_per_class':160,'count_scale':[0,160],'normalized_scale_percent':[0,100],'zero_annotations_hidden':True,'input_cm_csv_sha256':{k:v['sha256'] for k,v in render['source_matrices'].items()},'original_blues_and_comparison_images_excluded':True,'plots_generated_without_retraining':True},
    'tests':{'original_test_used':False,'professor_test_used':False,'official_train_validation_row_intersection':0,'official_source_group_intersection':0,'handmade_train_validation_writer_intersection':0,'postprocessing_changed_predictions':False},
    'notes':['Hand-made OOF is development cross-validation, not a final independent Test.','Combined pooled OOF consists of four independent fitted models, not one final combined model.','Historical W02 Normal baseline weights were not reused: a new Normal was fitted under matching configuration in the same controlled experiment.','metrics_summary.csv is a long-form table merging prior metrics CSVs; omitted derived confusion matrices and per-class metrics are reproducible from OOF_predictions.csv.']
  }
  (RESULT/'experiment_manifest.json').write_text(json.dumps(ex,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

  # Auditor validation for metrics_summary vs predictions for pooled and each writer, and no changes to the source prediction CSV.
  with (RESULT/'metrics_summary.csv').open(encoding='utf-8-sig',newline='') as f: st=list(csv.DictReader(f))
  for row in st:
    if row['dataset']!='handmade':continue
    sel=np.ones(len(tru),dtype=bool) if row['evaluation']=='pooled_OOF' else writers==int(row['held_out_writer'])
    pred=normals if row['model']=='Normal' else combos
    assert abs(accuracy_score(tru[sel],pred[sel])-float(row['accuracy']))<1e-12
    assert abs(f1_score(tru[sel],pred[sel],average='macro')-float(row['macro_f1']))<1e-12

# README for this single experiment (not extra analysis CSVs)
(RESULT/'README.md').write_text('''# EXP-CMB-001 — MLP Combined 4-Fold CV 결과

이 폴더는 **2026-11-15에 실제 실행 완료된** 16-class MLP Normal/Combined 통제 비교의 재현 산출물입니다.

- **작성자 보류 OOF**: Normal Accuracy **62.5781%**, Macro F1 **0.623046**; Combined Accuracy **81.25%**, Macro F1 **0.812043** (총 2,560장).
- Combined는 서로 다른 작성자를 제외하고 학습한 모델 4개의 OOF 예측을 합한 **개발 평가**입니다. 단일 최종 Combined 모델 또는 외부 Test 결과가 아닙니다.
- 공식 v3 학습 데이터: 96,000장 중 Group-aware Train 76,800장 / Development Validation 19,200장. Fold별 Combined Train은 여기에 Hand-made 학습 1,920장을 더한 **78,720장**, 작성자 보류 Validation 640장입니다.
- 공식 Original Test와 교수님 Test는 사용하지 않았습니다.

| 파일 | 사용 목적 |
|---|---|
| `metrics_summary.csv` | 전체·작성자별·공식 개발 Validation 지표와 학습 시간·반복 수를 15개 행으로 통합 (행마다 데이터셋·평가 방식·모델·보류 작성자 명시) |
| `OOF_predictions.csv` | 모든 2,560장의 정답, 작성자, 원본 행 인덱스, Normal/Combined 예측 및 정오 여부. CM·클래스별 F1·오분류 목록 재생성의 근거 |
| `loss_curves.csv` | Normal + Combined 4-Fold의 iteration별 Training Loss. **Validation Loss가 아님** |
| `experiment_manifest.json` | 데이터 해시·분할·Fold 목록·환경·모델 설정·실행 URL·시각화 조건·검사 결과 통합 |
| `official_split_indices.npz` | 공식 데이터 Train/Validation 원본 행 인덱스 및 출처 그룹 ID. 정확한 분할 복원 |
| `trained_models.zip` | 실제 학습·보존한 Normal 1개, Combined 4개 `.joblib` 파일 (압축 내부 이름 유지; 해시는 manifest 참고) |
| `figures/` | 최종 Viridis Count CM 2개 / Row-normalized CM 2개 / MLP Training Loss PNG 1개 |

## 재현 방법

실행 코드는 [`../../scripts/`](../../scripts/)에서 확인할 수 있습니다. 새로운 학습 없이 CM만 재생성하려면 이 저장소 루트에서 다음 명령을 실행합니다.

```bash
python 05_model_optimize/MLP/Combined/scripts/render_cm_viridis.py
```

`OOF_predictions.csv`의 `writer`, `index_in_writer`는 기존 원본 `02_data/Hand_made/ml26_{writer}.npz`의 **0 기반 순서**입니다. 행별 예측에서 Accuracy, Macro F1 및 Confusion Matrix를 다시 계산할 수 있습니다.

**주의:** 모델 역직렬화(`joblib.load`)는 신뢰할 수 있는 파일에 대해서만 수행합니다. 모델 5개는 CV 평가 및 검증용이지, 별도 최종 모델 선정·재학습을 의미하지 않습니다.

GitHub Actions 원본 기록: [Run #43622998640](https://github.com/ryu980920/Machine_Learning/actions/runs/43622998640).
''',encoding='utf-8')

# Log based on actual results. No speculative augmented policy.
LOGTXT='''# EXP유001 | Combined MLP Baseline 및 작성자별 4-Fold Cross-Validation

> **실험 ID:** EXP-CMB-001 · **2026-11-15** (KST) · **담당자:** 유용성 · **상태:** 완료
>
> **질문:** 공식 데이터만 학습한 Normal MLP에 Hand-made Training 데이터를 추가하면, 학습에 참여하지 않은 작성자의 16-class 손글씨 인식 성능이 좋아지는가?
>
> **결론:** 동일 MLP 조건에서 Hand-made OOF Accuracy **62.5781% → 81.2500% (+18.6719%p)**, Macro F1 **0.623046 → 0.812043**. 단, `0 ↔ D` 혼동은 개선되지 않았으며 일부 Fold의 공식 Validation 성능이 감소했다.

| 항목 | 기록 |
|---|---|
| 구분 | MLP / Combined 초기 Baseline, Normal 대조군 |
| 실험 근거 | [GitHub Actions Run #43622998640](https://github.com/ryu980920/Machine_Learning/actions/runs/43622998640), 최종 `success` |
| 모델·전처리 | `PixelScaler255(X/255.0) → MLPClassifier(random_state=42)` |
| 모델 설정 | MLP 기본 설정 (은닉층 100개 노드, ReLU, Adam 등); 정확한 값은 [`experiment_manifest.json`](../results/EXP_CMB_001/experiment_manifest.json) |
| 재현 코드 | [`combined_mlp_writer_cv.py`](../scripts/combined_mlp_writer_cv.py) / [`test_reproducibility.py`](../scripts/test_reproducibility.py) |
| 수치 근거 | [`metrics_summary.csv`](../results/EXP_CMB_001/metrics_summary.csv) / [`OOF_predictions.csv`](../results/EXP_CMB_001/OOF_predictions.csv) |

## 1. 문제 발견 및 실험 목적

기존 Normal MLP는 공식 MNIST·EMNIST 개발 Validation에 비해 Hand-made 데이터 인식 성능이 낮았다. 학습에 포함되지 않은 필체에 대한 일반화 문제를 확인하기 위해 **MLP 구조와 하이퍼파라미터를 변경하지 않은 상태**에서 Hand-made 학습 데이터를 추가했다.

이번 실험의 핵심 비교 변수는 **Hand-made 데이터를 학습에 포함하는지 여부**다. 다음 세 가지를 확인한다.

1. 학습하지 않은 작성자의 Hand-made Accuracy·Macro F1 개선 여부
2. Confusion Matrix에서 개선·악화된 클래스와 작성자별 편차
3. 공식 MNIST·EMNIST 개발 Validation 정확도 유지 여부

## 2. 원인 가설 및 방법 선정

- **가설:** 공식 데이터로만 학습한 Normal MLP는 실제 작성자의 필체·획 형태 다양성을 충분히 경험하지 못했다. Hand-made Training을 추가하면 미학습 작성자에 대한 오분류가 감소할 것으로 예상했다.
- **방법:** 작성자 ID를 기준으로 한 **Leave-One-Writer-Out 4-Fold Cross-Validation(CV)**. 각 Fold에서 작성자 3명의 Hand-made 이미지를 학습에 추가하고 나머지 1명은 검증 전용으로 보류했다.
- **통제:** Normal은 공식 Training만으로 새로 1회 학습하고 네 작성자에게 평가했다. Combined는 같은 모델 설정으로 Fold마다 새로 학습했다. 추가 회전·반전·이진화·증강이나 하이퍼파라미터 튜닝은 시행하지 않았다.

## 3. 데이터·분할·수행 조건

| 항목 | Normal | Combined (각 Fold) |
|---|---:|---:|
| 공식 MNIST·EMNIST Train | 76,800 | 76,800 |
| 추가 Hand-made Train | 0 | 1,920 (작성자 3명) |
| 총 학습 이미지 | 76,800 | **78,720** |
| 공식 개발 Validation | 19,200 | 19,200 |
| 보류 작성자 Validation | 640/작성자 | 640/작성자 |
| 모델 설정 | `MLPClassifier(random_state=42)` | 동일 |
| 픽셀 전처리 | `X/255.0` | 동일 |
| 학습 재사용 | 공식 Train으로 1회 새 학습 | Fold마다 새 학습 |

- 공식 v3 NPZ: `02_data/datasets/MNIST_EMNIST_official_split_112000.npz`, SHA-256 `0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879`.
- 공식 96,000 Training의 내부 분할: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` 첫 Fold 사용. 출처 그룹은 `label·원본 split·원본 index`를 합성해 원본 및 증강본의 Training/Validation 분리 보장.
- Hand-made: `02_data/Hand_made/ml26_41.npz`~`ml26_44.npz` 각 640장, 작성자별 16클래스×40장, 총 2,560장. **현재 Grayscale 버전**, 기존 보정 방향 유지, 새 42번 NPZ 사용. 원본 SHA-256은 manifest에 수록.

| Fold | Combined 학습 작성자 | 보류 Validation 작성자 | 학습 수 | Validation 수 |
|---|---|---|---:|---:|
| 1 | 42·43·44 | **41** | 78,720 | 640 |
| 2 | 41·43·44 | **42** | 78,720 | 640 |
| 3 | 41·42·44 | **43** | 78,720 | 640 |
| 4 | 41·42·43 | **44** | 78,720 | 640 |

**실제 실행 순서:** SHA-256 확인 → 공식 Train/Validation 인덱스 생성 및 그룹 중복 검사 → Normal 1회 학습 → Combined Fold별 1회씩 학습 → 동일 작성자 Hand-made 추론 → 2,560장 OOF 예측 취합 → 공식 개발 Validation 평가 → CM·손실 곡선 및 학습 모델 저장 → CSV 수치 재검산.

**재현 자료:** [`official_split_indices.npz`](../results/EXP_CMB_001/official_split_indices.npz) (Train/Validation 실제 원본 행 인덱스), [`experiment_manifest.json`](../results/EXP_CMB_001/experiment_manifest.json) (Fold 작성자·이미지 범위·모델·환경·SHA), [`trained_models.zip`](../results/EXP_CMB_001/trained_models.zip) (Normal 1개 및 Combined 4개 모델).

## 4. 실측 결과

### 4.1 Hand-made OOF 전체 비교

| 항목 | Normal | Combined | 변화 |
|---|---:|---:|---:|
| Accuracy | 62.5781% | **81.2500%** | **+18.6719%p** |
| Macro F1 | 0.623046 | **0.812043** | +0.188997 |
| 맞힌 이미지 / 2,560 | 1,602 | **2,080** | +478 |
| 틀린 이미지 / 2,560 | 958 | **480** | -478 |

이미지별 실제 예측 비교 시 **Normal 오답 → Combined 정답 540장**, **Normal 정답 → Combined 오답 62장**, 순개선 478장이다.

### 4.2 작성자별 성능

| 보류 작성자 | Normal Accuracy | Combined Accuracy | 개선폭 | Combined Macro F1 |
|---|---:|---:|---:|---:|
| 41 | 60.3125% | **75.9375%** | +15.6250%p | 0.7554 |
| 42 | 68.4375% | **85.6250%** | +17.1875%p | 0.8535 |
| 43 | 55.4688% | **78.5938%** | +23.1250%p | 0.7737 |
| 44 | 66.0938% | **84.8438%** | +18.7500%p | 0.8474 |

모든 보류 작성자에서 개선이 관찰되었다. 특히 43번은 개선폭이 가장 컸지만, Combined 이후에도 41·43번 작성자의 정확도가 상대적으로 낮았다.

### 4.3 Confusion Matrix 분석

최종 이미지에는 16×16 `viridis` 디자인을 적용했다. **원본 예측값은 변경하지 않았고**, Count형은 색상 범위 0~160장, 행 정규화형은 0~100%로 고정했다.

| 실제 → 예측 | Normal | Combined | 변화 |
|---|---:|---:|---|
| `0 → D` | 44 | 30 | 감소 |
| `D → 0` | 14 | 29 | **증가** |
| `8 → B` | 43 | 27 | 감소 |
| `B → 8` | 50 | 26 | 감소 |
| `5 → F` | 57 | 17 | 감소 |
| `4 → F` | 52 | 7 | 감소 |
| `9 → F` | 45 | 5 | 감소 |
| `7 → F` | 45 | 9 | 감소 |

- `0 ↔ D` 양방향 오류는 **58장 → 59장**으로 개선되지 않았다. 43번 작성자의 `0 → D` 28장과 41·42번 작성자의 `D → 0` 16·11장을 우선 검토한다.
- `8 ↔ B` 양방향 오류는 **93장 → 53장**으로 줄었지만 잔존한다.
- F로의 과다 예측은 감소했으며, F Precision은 **37.14% → 73.13%**로 개선됐다.

**최종 그래프:** [Normal Count](../results/EXP_CMB_001/figures/normal_count_viridis.png) · [Combined Count](../results/EXP_CMB_001/figures/combined_count_viridis.png) · [Normal Row-normalized](../results/EXP_CMB_001/figures/normal_normalized_viridis.png) · [Combined Row-normalized](../results/EXP_CMB_001/figures/combined_normalized_viridis.png).

### 4.4 공식 개발 Validation 성능

| 모델 | Accuracy | Macro F1 |
|---|---:|---:|
| Normal | 97.2240% | 0.9721 |
| Combined (41 보류) | 96.5313% | 0.9651 |
| Combined (42 보류) | 97.2604% | 0.9724 |
| Combined (43 보류) | 97.3594% | 0.9734 |
| Combined (44 보류) | 97.3177% | 0.9730 |

41번 작성자 보류 Fold는 Normal보다 공식 Validation Accuracy가 약 **0.69%p 감소**했다. 공식 데이터 성능 보존도 후속 Combined 최적화에서 함께 관리한다.

### 4.5 학습 반복·시간·Loss

| 모델 | 반복 수 | 학습 시간 | 최종 Training Loss |
|---|---:|---:|---:|
| Normal | 66 | 32.82초 | 0.001386 |
| Combined (41 보류) | 76 | 39.25초 | 0.002767 |
| Combined (42 보류) | 68 | 34.93초 | 0.001592 |
| Combined (43 보류) | 70 | 35.60초 | 0.001548 |
| Combined (44 보류) | 69 | 35.60초 | 0.001524 |

Loss Curve는 **Iteration별 Training Loss**이며 Validation Loss를 측정한 것이 아니다. 5개 모델 모두 학습 경고가 발생하지 않았다. [Loss Curve](../results/EXP_CMB_001/figures/MLP_loss_curves.png) · [원본 수치](../results/EXP_CMB_001/loss_curves.csv).

## 5. 결과 판단 및 다음 실험

**판단:** Hand-made 이미지를 추가한 Combined MLP는 보류 작성자 OOF Accuracy를 **+18.67%p** 개선했다. 동일한 모델과 스케일러에서 비교했으므로 Hand-made 데이터 편입 효과를 확인했다.

**남은 문제:** `0/D`, `8/B` 필체 혼동이 남고, 특정 Fold는 공식 데이터 개발 Validation 정확도가 하락했다. 그 원인이 되는 이미지 형태는 아직 별도로 정량 분석하지 않았다.

**후속 후보(미실시):** `0↔D`, `8↔B` 오분류 이미지를 작성자별로 점검하고, 중심·기울기·크기·획 굵기의 원인을 측정한다. 해당 결과를 바탕으로 자동 전처리 또는 Training-only Augmentation을 단일 변수씩 비교하고, 같은 4-Fold CV 및 공식 개발 Validation으로 판단한다.

**검증 근거:** 공식 원본·증강 그룹 누출 0, Hand-made 작성자 누출 0, SHA-256 확인, 저장된 모델/예측/CM/Loss 수치 일치. 공식 Original Test 16,000장과 교수님 Test는 **미사용**.

**해석 범위:** OOF는 개발 평가이며 최종 독립 Test가 아니다. 네 Combined Fold는 **각각 별도 모델 4개**이므로, 현재 성능을 단일 최종 모델의 성능으로 표시하지 않는다. 기존 W02 Normal 저장 가중치를 재사용하지 않고 이 실험에서 통제용 Normal을 새로 학습했다.
'''
(LOG/'EXP유001_Combined_MLP_4Fold_CV_Baseline.md').write_text(LOGTXT,encoding='utf-8')

# Ensure exactly 11 results data+figs files, excluding README
resultfiles=[p for p in RESULT.rglob('*') if p.is_file() and p.name!='README.md']
assert len(resultfiles)==11, [str(p) for p in resultfiles]
print('BUILD OK',OUT,'\nRESULTS:')
for p in resultfiles:print(str(p.relative_to(ROOT)),p.stat().st_size)
print('LOG size', (LOG/'EXP유001_Combined_MLP_4Fold_CV_Baseline.md').stat().st_size)