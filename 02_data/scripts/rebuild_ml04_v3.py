#!/usr/bin/env python3
"""Rebuild and verify the reviewed v3 NPZ from the official 112,000 image baseline.

Requires Python 3.13, NumPy 2.3.5, SciPy 1.17.0.
Stops without modifying the original unless the resulting file exactly matches
the reviewed v3 SHA-256. This is a one-time migration, not a training pipeline.
"""
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, rotate

DATASET = Path("02_data/datasets/MNIST_EMNIST_official_split_112000.npz")
ORIGINAL_HASH = "b52ce7873abb21182627841fb22b5c87fd6f54ac5cb925242deab44499271ec1"
REVIEWED_HASH = "0e9570d0eb04c0c1da99d043b85bfdab7f3b639eebe4b10335b05b57ad671879"
SEED = 20261107

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def imghash(im):
    return hashlib.blake2b(np.ascontiguousarray(im).tobytes(), digest_size=16).digest()

def duplicates(train, test):
    a, b = Counter(map(imghash, train)), Counter(map(imghash, test))
    return sum(n*(n-1)//2 for n in a.values()), sum(n*b.get(h,0) for h,n in a.items())

def border_max(x):
    return int(max(np.max(x[0,:]), np.max(x[-1,:]), np.max(x[:,0]), np.max(x[:,-1])))

def transform(original, method, rng):
    a = original.astype(np.float32)
    if method == 1:
        strength = float(rng.choice([-1.,1.]))*float(rng.uniform(3.0,9.0))
        image = rotate(a,strength,reshape=False,order=1,mode="constant",cval=0.,prefilter=False)
    elif method == 2:
        strength = float(rng.uniform(0.22,0.45))
        image = (1-strength)*a + strength*maximum_filter(a,size=3,mode="constant",cval=0)
    elif method == 3:
        strength = float(rng.uniform(0.20,0.42))
        image = (1-strength)*a + strength*minimum_filter(a,size=3,mode="constant",cval=0)
    else: raise ValueError(method)
    return np.rint(np.clip(image,0,255)).astype(np.uint8),strength

def quality_ok(orig,new,method):
    before,after=int(orig.sum()),int(new.sum())
    if before==0 or after==0 or np.array_equal(orig,new):return False
    low,high={1:(0.90,1.08),2:(1.05,1.53),3:(0.57,0.97)}[method]
    if not low<=after/before<=high:return False
    if border_max(orig)<35 and border_max(new)>=100:return False
    return True

def main():
    actual=sha256(DATASET)
    if actual == REVIEWED_HASH:
        print("Already v3; no changes")
        return
    if actual != ORIGINAL_HASH:
        raise RuntimeError(f"Unknown NPZ version: {actual}")
    with np.load(DATASET,allow_pickle=False) as f:
        d={k:f[k].copy() for k in f.files}
    tr,te=d["X_train"],d["X_test"]
    ytr,yte=d["y_train"],d["y_test"]
    aug=d["train_is_augmented"]
    assert tr.shape==(96000,28,28) and te.shape==(16000,28,28)
    assert int(aug.sum())==5348 and duplicates(tr,te)[1]==1
    assert int(ytr[78001])==int(ytr[79394])==int(yte[13962])==13
    assert np.array_equal(tr[79394],te[13962])
    assert not aug[78001] and not aug[79394]

    # v2: swap the D original at train[78001] with test[13962].
    for tk,ek in [("X_train","X_test"),("train_source_index","test_source_index"),
                  ("train_source_split","test_source_split")]:
        tmp=d[tk][78001].copy()
        d[tk][78001]=d[ek][13962]
        d[ek][13962]=tmp

    source=d["train_source_index"]
    split=d["train_source_split"]
    lookup={(int(ytr[i]),int(split[i]),int(source[i])):int(i)
            for i in np.flatnonzero(~aug)}
    assert len(lookup)==int((~aug).sum())
    rng=np.random.default_rng(SEED)
    types=np.zeros(len(tr),dtype=np.uint8)
    strengths=np.zeros(len(tr),dtype=np.float32)
    aug_ids=np.flatnonzero(aug)
    for label in (11,13,14):
        ids=aug_ids[ytr[aug_ids]==label]
        counts=[round(len(ids)*0.40),round(len(ids)*0.30)]
        counts += [len(ids)-sum(counts)]
        recipe=np.repeat([1,2,3],counts)
        rng.shuffle(recipe)
        types[ids]=recipe
    assert np.all(types[aug_ids]!=0)

    seen={imghash(im) for im in tr[~aug]}
    seen.update(imghash(im) for im in te)
    for i in aug_ids:
        key=(int(ytr[i]),int(split[i]),int(source[i]))
        parent=tr[lookup[key]]
        requested=int(types[i])
        for attempt in range(100):
            candidate,strength=transform(parent,requested,rng)
            if quality_ok(parent,candidate,requested) and imghash(candidate) not in seen:
                break
        else:raise RuntimeError(f"Augmentation failed at {i}")
        tr[i]=candidate
        strengths[i]=strength
        seen.add(imghash(candidate))
    d["train_augmentation_type"]=types
    d["train_augmentation_strength"]=strengths
    d["augmentation_seed"]=np.array(SEED,dtype=np.int64)
    d["dataset_revision"]=np.array("v2: split_duplicate_swap+rotation+stroke")
    assert duplicates(tr,te)==(1,0)

    # v3: correct the remaining duplicate inside Training only.
    i,j=78001,79394
    assert np.array_equal(tr[i],tr[j]) and not aug[i]
    image=rotate(tr[j].astype(np.float32),5.0,reshape=False,order=1,mode="constant",cval=0.0)
    tr[i]=np.rint(np.clip(image,0,255)).astype(np.uint8)
    aug[i]=True
    d["train_source_split"][i]=d["train_source_split"][j]
    d["train_source_index"][i]=d["train_source_index"][j]
    types[i]=1
    strengths[i]=5.0
    d["dataset_revision"]=np.asarray("v3: single_D_train_dup_aug_replacement")
    assert duplicates(tr,te)==(0,0)
    assert int(aug.sum())==5349
    assert Counter(types[aug].tolist())==Counter({1:2141,2:1604,3:1604})
    for label,expected in [(11,2474),(13,1660),(14,1215)]:
        assert int(((ytr==label)&aug).sum())==expected
    tmp=DATASET.with_name(DATASET.stem+"_tmp.npz")
    np.savez_compressed(tmp,**d)
    checksum=sha256(tmp)
    if checksum!=REVIEWED_HASH:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"v3 hash mismatch: {checksum}")
    tmp.replace(DATASET)
    print(f"Updated {DATASET}: SHA256={sha256(DATASET)}, 96000/16000; augmented=5349; exact duplicates=0")

if __name__=="__main__":main()
