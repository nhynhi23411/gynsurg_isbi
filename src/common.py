import json,hashlib,csv,random
from pathlib import Path
import numpy as np
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def approved(root,test=False):
    d=json.loads((Path(root)/'configs/review_status.json').read_text())
    key='approved_for_test' if test else 'approved_for_training'
    if not d.get(key) or not d.get('reviewer') or not d.get('protocol_sha256'):
        raise RuntimeError('Review required: complete configs/review_status.json after user approval')
def fresh_out(p):
    p=Path(p)
    if p.exists() and any(p.iterdir()):raise FileExistsError('Use a new empty run directory')
    p.mkdir(parents=True,exist_ok=True);return p
def write_csv(path,rows):
    if not rows:raise ValueError('No eligible rows')
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def paired_bootstrap(a,b,seed=20261016,n=2000):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=b.shape or not a.size or not np.isfinite(a-b).all():raise ValueError('Invalid paired cases')
    rng=np.random.default_rng(seed);d=a-b
    boot=d[rng.integers(0,len(d),(n,len(d)))].mean(1)
    return dict(delta=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n_groups=len(d))
def overlap(pred,gold):
    p=np.asarray(pred,bool);g=np.asarray(gold,bool);inter=np.logical_and(p,g).sum();total=p.sum()+g.sum();union=np.logical_or(p,g).sum()
    return dict(dice=float(2*inter/total) if total else 1.,iou=float(inter/union) if union else 1.)
