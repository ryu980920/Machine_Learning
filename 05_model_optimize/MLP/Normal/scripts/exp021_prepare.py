import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,sys,time
import numpy as np
root=Path('/mnt/data/EXP021_early_stopping_diagnostic')
sys.path.insert(0,str(root))
import train_local_warp as old
old.ROOT=root
old.V3=root/'MNIST_EMNIST_official_split_112000.npz'
t=time.monotonic();print('VERIFY_OFFICIAL',flush=True)
x,y,xval,yval=old.load_official()
train,report=old.build_base_train(x,y)
assert report['base_ink_mass']==2133060736.0,report
np.save(root/'X_train_base.npy',train);np.save(root/'y_train.npy',y)
np.save(root/'X_val_com.npy',old.com(xval));np.save(root/'y_val.npy',yval)
(root/'prep_manifest.json').write_text(json.dumps(report,indent=2))
print('PREP_DONE',round(time.monotonic()-t,2),report,flush=True)
