#!/usr/bin/env python3
"""Prepare common leakage-free subset for 13 baselines from reviewed ML04 v3.

Source 96k Training only. NO official Test and NO Hand_made at candidate screening.
"""
import hashlib,json,platform,argparse
from pathlib import Path
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Prepare group-separated 12k/3.2k preview dataset from v3.')
parser.add_argument('--official',type=Path,default=Path('02_data/datasets/MNIST_EMNIST_official_split_112000.npz'),
                    help='Path to the ORIGINAL verified official v3 NPZ')
args=parser.parse_args()
ROOT.mkdir(exist_ok=True)
INPUT=args.official
EXPECTED='0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879'
with INPUT.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
assert digest==EXPECTED, f'Not the reviewed v3 file: {digest}'
with np.load(INPUT, allow_pickle=False) as data:
    x=data['X_train']
    y=data['y_train'].astype(np.int64)
    source=data['train_source_index'].astype(np.int64)
    split=data['train_source_split'].astype(np.int64)
    augmented=data['train_is_augmented']
assert x.shape==(96000,28,28) and len(y)==96000 and int(augmented.sum())==5349
groups=y * 1_000_000_000 + split*100_000_000 + source
skf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
tidx,vidx=next(skf.split(x,y,groups))
assert len(tidx)==76800 and len(vidx)==19200
assert not set(groups[tidx]).intersection(groups[vidx])

def choose(indices,n_per_class,seed):
    rng=np.random.default_rng(seed)
    result=[]
    for cls in range(16):
        ids=indices[y[indices]==cls]
        assert len(ids)>=n_per_class,(cls,len(ids))
        result.extend(rng.choice(ids,size=n_per_class,replace=False).tolist())
    return np.asarray(sorted(result),dtype=np.int64)

train_idx=choose(tidx,750,42)
val_idx=choose(vidx,200,43)
assert len(train_idx)==12000 and len(val_idx)==3200
assert not set(groups[train_idx]).intersection(groups[val_idx])
assert all(np.bincount(y[train_idx],minlength=16)==750)
assert all(np.bincount(y[val_idx],minlength=16)==200)
# Small shape-preserving input to minimize memory overhead in evaluation subprocesses.
np.savez_compressed(ROOT/'common_subset.npz',X_train=x[train_idx],y_train=y[train_idx],X_val=x[val_idx],y_val=y[val_idx],train_idx=train_idx,val_idx=val_idx)
meta={'source_path':'02_data/datasets/MNIST_EMNIST_official_split_112000.npz',
      'source_sha256':digest,'source_training_count':96000,'source_official_test_count_NOT_USED':16000,
      'full_training_part':int(len(tidx)),'full_validation_part':int(len(vidx)),
      'screening_train_n':int(len(train_idx)),'screening_val_n':int(len(val_idx)),
      'per_class_train':750,'per_class_validation':200,'validation_group_leakage':0,
      'split':'StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42) first fold',
      'subsample_train_seed':42,'subsample_validation_seed':43,'classes':'0-9,A-F',
      'sample_shape':'(28,28)','sklearn_version':sklearn_version,'numpy_version':np.__version__,
      'python_version':platform.python_version(),
      'disclaimer':'Early 13-model screening only. No official Test or Hand-made data exposed.'}
(ROOT/'common_data_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(meta,indent=2,ensure_ascii=False),flush=True)