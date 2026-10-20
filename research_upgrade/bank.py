"""Finite, target-factorized replay for TRAIN-only counterfactual simulation.

Paid history is enforced for every observable lookup. Scoring labels are never
passed to selection or feature extraction. Existing SAM2 replay remains frozen.
"""
import csv,hashlib,itertools,json,time
from pathlib import Path
import numpy as np

QUERY=(0,24,49)
SCORE=(74,99,124)
FRAMES=QUERY+SCORE
BUDGETS=(0,1,2,5)
AUC_WEIGHTS=(.1,.2,.4,.3)

def popcount(n):return int(n).bit_count()
def subsets(paid):return [s for s in range(8) if s & ~paid == 0]
def entropy(p):
    p=np.clip(np.asarray(p,np.float32),1e-6,1-1e-6)
    return float(-(p*np.log(p)+(1-p)*np.log(1-p)).mean()/np.log(2))
def overlap(pred,gold):
    p=np.asarray(pred,bool);g=np.asarray(gold,bool);inter=int((p&g).sum());a=int(p.sum());b=int(g.sum())
    return (2*inter/(a+b) if a+b else 1.,inter/(a+b-inter) if a+b-inter else 1.)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_csv(path,rows):
    rows=list(rows)
    if not rows:raise ValueError('Empty table')
    with Path(path).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class PaidView:
    def __init__(self,states,anchors,head,paid):
        if paid not in range(8):raise ValueError('Invalid paid history')
        self.states=states;self.anchors=anchors;self.head=int(head);self.paid=int(paid)
    def probability(self,subset,frame):
        if subset & ~self.paid:raise ValueError('Unpurchased prompt state accessed')
        if frame not in FRAMES:raise ValueError('Snapshot unavailable')
        return self.states[self.head,subset][FRAMES.index(frame)]
    def mask(self,index):
        if not self.paid & (1<<index):raise ValueError('Unpurchased mask accessed')
        return self.anchors[self.head,index]

def observable_features(view,subset,frame):
    """No scored references, target-presence labels, or evaluation metrics input."""
    p=view.probability(subset,frame);base=view.probability(0,frame)
    m=p>.5;b=base>.5;roi=m|b
    idx=[j for j in range(3) if view.paid & (1<<j)]
    anchors=[];weights=[]
    # Single-anchor support is explicitly unavailable, never self-scored.
    if len(idx)>=2:
        for j in idx:
            reconstruction=view.probability(subset & ~(1<<j),QUERY[j])
            anchors.append(overlap(reconstruction>.5,view.mask(j))[0])
            weights.append(np.exp(-abs(frame-QUERY[j])/50.))
    agreement=float(np.average(anchors,weights=weights)) if anchors else 0.
    q=[QUERY[j] for j in idx]
    previous=FRAMES[max(0,FRAMES.index(frame)-1)]
    temporal=overlap(m,view.probability(subset,previous)>.5)[0]
    # Numeric features only. Missing support is represented explicitly.
    return dict(target_uterus=int(view.head==0),paid_count=len(idx),output_prompt_count=popcount(subset),
        output_is_raw=int(subset==view.paid),frame_seconds=frame/25.,
        nearest_paid_seconds=min([abs(frame-t)/25. for t in q],default=10.),
        entropy=entropy(p),baseline_entropy=entropy(base),roi_entropy=entropy(p[roi]) if roi.any() else 0.,
        foreground_fraction=float(m.mean()),baseline_foreground_fraction=float(b.mean()),
        probability_disagreement=float(np.abs(p-base).mean()),mask_disagreement=float((m!=b).mean()),
        temporal_mask_consistency=temporal,anchor_support=int(len(idx)>=2),
        anchor_reconstruction_mean=agreement,anchor_reconstruction_min=min(anchors) if anchors else 0.)

def trajectory(selector,policy,seed=20261017):
    """Selector contains entropy only; no gold/quality table is accessible here."""
    rng=np.random.default_rng(seed);paid=[0,0];history=[];paths={0:tuple(paid)}
    for n in range(1,6):
        eligible=[(j,k) for j in range(3) for k in range(2) if not paid[k]&(1<<j)]
        if policy=='random':j,k=eligible[int(rng.integers(len(eligible)))]
        elif policy=='uniform':
            least=min(popcount(paid[k]) for _,k in eligible)
            pool=[(j,k) for j,k in eligible if popcount(paid[k])==least]
            def spread(item):
                j,k=item;same=[QUERY[a] for a in range(3) if paid[k]&(1<<a)]
                distance=min(abs(QUERY[j]-a) for a in same) if same else min(QUERY[j]+1,125-QUERY[j])
                return distance,-QUERY[j],-k
            j,k=max(pool,key=spread)
        elif policy=='uncertainty':j,k=max(eligible,key=lambda z:(selector[z[1],paid[z[1]],QUERY[z[0]]],-QUERY[z[0]],-z[1]))
        else:raise ValueError('Unsupported selector')
        paid[k]|=1<<j;history.append((QUERY[j],k));paths[n]=tuple(paid)
    return paths,history

def score_paths(paths,labels,policy,seed,clip,source):
    rows=[]
    for n in BUDGETS:
        cells=[labels[k,paths[n][k],t] for k in range(2) for t in SCORE]
        rows.append(dict(clip_id=clip,procedure_id=source,policy=policy,seed=seed,n_queries=n,actual_density=.8*n,
            dice=float(np.mean([x['dice'] for x in cells])),iou=float(np.mean([x['iou'] for x in cells]))))
    return rows

def build_bank(d,cfg,out,engine_factory=None):
    from src.sam_engine import SAM2Engine
    from src.surgery import validate_sequence
    start=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p,g,f,qi,si=validate_sequence(d)
    if tuple(map(int,qi))!=QUERY or tuple(map(int,si))!=SCORE or p.shape[:2]!=(125,2):raise ValueError('Wrong official-only scope')
    anchors={(k,j):g[t,k].copy() for k in range(2) for j,t in enumerate(QUERY)}
    states={(k,0):p[list(FRAMES),k].copy() for k in range(2)}
    engine=(engine_factory or SAM2Engine)(cfg,str(d['jpeg_dir']),p)
    timings=[]
    for k in range(2):
        for subset in range(1,8):
            prompts=[(QUERY[j],k,anchors[k,j]) for j in range(3) if subset&(1<<j)]
            engine.prompts=[(t,h,mask.copy()) for t,h,mask in prompts[:-1]]
            before=time.perf_counter();pred=engine.correct(*prompts[-1]);elapsed=time.perf_counter()-before
            states[k,subset]=pred[list(FRAMES),k].copy()
            timings.append(dict(head=k,subset=subset,prompts=popcount(subset),propagation_seconds=elapsed))
            del pred
    del engine
    clip=str(d['clip_id']);source=str(d['procedure_id'])
    selector={};label_lookup={};labels=[];features=[];selectors=[]
    for k in range(2):
        for subset in range(8):
            for t in QUERY:
                e=entropy(states[k,subset][FRAMES.index(t)]);selector[k,subset,t]=e
                selectors.append(dict(clip_id=clip,procedure_id=source,head=k,subset=subset,frame=t,entropy=e))
            for t in SCORE:
                prediction=states[k,subset][FRAMES.index(t)]>.5;reference=g[t,k]
                dice,iou=overlap(prediction,reference)
                row=dict(clip_id=clip,procedure_id=source,head=k,subset=subset,frame=t,dice=dice,iou=iou,gold_present=bool(reference.any()),prediction_present=bool(prediction.any()))
                labels.append(row);label_lookup[k,subset,t]=row
        for paid in range(8):
            view=PaidView(states,anchors,k,paid)
            for subset in subsets(paid):
                for t in SCORE:
                    features.append(dict(clip_id=clip,procedure_id=source,head=k,paid=paid,subset=subset,frame=t,**observable_features(view,subset,t)))
    curves=[];queryrows=[]
    for policy,seeds in [('uncertainty',[20261017]),('uniform',[20261017]),('random',list(range(100,110)))]:
        for seed in seeds:
            paths,history=trajectory(selector,policy,seed)
            curves.extend(score_paths(paths,label_lookup,policy,seed,clip,source))
            queryrows.extend(dict(clip_id=clip,procedure_id=source,policy=policy,seed=seed,query=n,frame=t,head=k) for n,(t,k) in enumerate(history,1))
    for name,rows in [('features',features),('evaluation_labels',labels),('selector_observables',selectors),('policy_curves',curves),('policy_queries',queryrows),('timings',timings)]:write_csv(out/(name+'.csv'),rows)
    record=dict(clip_id=clip,procedure_id=source,propagations=14,feature_rows=len(features),label_rows=len(labels),
        selection_uses_scoring_gold=False,cache_scope='training counterfactual simulation only',all_gold_prompts_charged_in_simulated_history=True,
        total_seconds=time.perf_counter()-start,files={x.name:sha(x) for x in out.glob('*.csv')})
    (out/'bank.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    return record
