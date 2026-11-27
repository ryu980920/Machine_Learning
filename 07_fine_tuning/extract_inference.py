#!/usr/bin/env python3
"""Run frozen SVC/KNN checkpoint inference using original CoM -> Deskew -> Pixel/HOG.
No data is used for training. Input npz must contain X or img (28x28, grayscale
0..255 in the original dataset convention). Model must be a trusted joblib
checkpoint from this experiment. Do not unpickle untrusted model files.
"""
import argparse,os,sys,zipfile,tempfile,hashlib,csv
from pathlib import Path
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="2"
import numpy as np,joblib
from threadpoolctl import threadpool_limits
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--kit",type=Path,required=True)
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--model-sha256",required=True)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    with a.model.open("rb") as f: assert hashlib.file_digest(f,"sha256").hexdigest()==a.model_sha256
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        with zipfile.ZipFile(a.kit) as z:
            for member in z.namelist():
                if member=="final_benchmark.py" or member.startswith("sources/"):z.extract(member,tmp)
        sys.path.insert(0,str(tmp))
        import final_benchmark as fb
        with np.load(a.input,allow_pickle=False) as z:
            key="X" if "X" in z.files else "img"
            x=z[key];assert x.ndim in (2,3) and x.size%784==0
            assert np.isfinite(x).all() and x.min()>=0 and x.max()<=255
            x=x.reshape(-1,784)
            ids=z["sample_index"] if "sample_index" in z.files else np.arange(len(x))
            collectors=z["collector_id"] if "collector_id" in z.files else np.full(len(x),-1)
        with threadpool_limits(2):
            model=joblib.load(a.model)
            assert np.array_equal(model.classes_,np.arange(16))
            pred=model.predict(fb.hybrid(x))
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["collector","index","prediction","label_id"])
            w.writerows((int(c),int(i),fb.LABELS[int(j)],int(j)) for c,i,j in zip(collectors,ids,pred))
        print("Predictions saved:",len(pred),a.out)
if __name__=="__main__":main()
