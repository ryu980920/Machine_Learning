"""Unit checks using *synthetic* examples only (never report as experiment metrics)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from tempfile import TemporaryDirectory

spec = importlib.util.spec_from_file_location('exp_cmb001',Path(__file__).parent/'combined_mlp_writer_cv.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

model = m.new_mlp()
assert isinstance(model.named_steps['model'],MLPClassifier)
assert model.named_steps['model'].get_params()==MLPClassifier(random_state=42).get_params()
X=np.array([[0,255],[255,127]],dtype=np.float32)
assert np.allclose(model.named_steps['scale_0_1'].transform(X),X/255.)
assert np.allclose(m.metrics(np.array([0,1,2]),np.array([0,1,3]))['accuracy'],2/3)

# Synthetic original/augmentation groups: 16 classes, each 30 unique groups, 2 rows per source.
y=np.repeat(np.arange(16),60)
group=np.repeat(np.arange(16*30),2)
ids=np.arange(len(y))
tr,va=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(ids,y,group))
assert len(set(group[tr]) & set(group[va]))==0
assert np.array_equal(np.sort(np.r_[tr,va]),ids)

with TemporaryDirectory() as td:
    d=Path(td)
    labels=np.arange(16)
    pred=labels.copy(); pred[7]=8
    target=labels.copy()
    cm=m.save_cm(d/'confusion.csv',target,pred)
    m.save_per_class(d/'perclass.csv',target,pred)
    assert (d/'confusion.csv').exists() and cm.shape==(16,16)
    assert int(cm.sum())==16
    assert (d/'perclass.csv').exists()
    y=np.arange(16,dtype=int)
    m.make_plots(d,y,y,np.roll(y,1),{'Normal':[1.,.8,.5], 'Combined held-out 41':[1.,.9,.6]})
    assert (d/'MLP_loss_curves.png').exists()
print('PASS: model definitions, scaler, synthetic group split, metrics, CSV, plots')