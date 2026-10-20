"""Fixed one-head external controls. Selector never receives evaluation labels."""
import itertools,json,time
from pathlib import Path
import numpy as np
from research_upgrade.external_scope import QUERY,SCORE,BUDGETS,AUC_WEIGHTS,decode_release_mask,valid_overlap

def entropy(p):
    p=np.clip(np.asarray(p,np.float32),1e-6,1-1e-6)
    return float(-(p*np.log(p)+(1-p)*np.log(1-p)).mean()/np.log(2))

class PaidEntropy:
    def __init__(self,table):self.table=table;self.paid=0
    def read(self,state,j):
        if state!=self.paid:raise ValueError('Only current paid state is observable')
        if self.paid&(1<<j):raise ValueError('Already purchased')
        return self.table[state,j]
    def buy(self,j):
        if j not in range(3) or self.paid&(1<<j):raise ValueError('Invalid purchase')
        self.paid|=1<<j

def trajectory(entropies,policy,seed=100):
    view=PaidEntropy(entropies);rng=np.random.default_rng(seed);paths=[0];history=[]
    for _ in range(3):
        eligible=[j for j in range(3) if not view.paid&(1<<j)]
        if policy=='uncertainty':j=max(eligible,key=lambda j:(view.read(view.paid,j),-QUERY[j]))
        elif policy=='random':j=eligible[int(rng.integers(len(eligible)))]
        elif policy=='uniform':
            prior=[QUERY[j] for j in range(3) if view.paid&(1<<j)]
            j=max(eligible,key=lambda j:(min(abs(QUERY[j]-t) for t in prior) if prior else min(QUERY[j]+1,80-QUERY[j]),-QUERY[j]))
        else:raise ValueError('Unknown fixed policy')
        view.buy(j);paths.append(view.paid);history.append(QUERY[j])
    return paths,history

def score_trajectory(paths,labels):
    rows=[]
    for n,state in enumerate(paths):
        cells=[labels[state,t] for t in SCORE];available=[c for c in cells if c['dice'] is not None]
        rows.append(dict(n_queries=n,actual_density=1.25*n,
            dice=float(np.mean([c['dice'] for c in available])) if available else None,
            iou=float(np.mean([c['iou'] for c in available])) if available else None,
            scored_frames=len(available),missing_frames=len(cells)-len(available)))
    return rows

def build_controls(baseline,query_masks,score_masks,valid_maps,cfg,jpeg_dir,engine_factory=None):
    from src.sam_engine import SAM2Engine
    factory=engine_factory or SAM2Engine
    if baseline.shape[:2]!=(80,1):raise ValueError('One head / 80 raw frames required')
    if set(query_masks)!=set(QUERY) or set(score_masks)!=set(SCORE) or set(valid_maps)!=set(SCORE):raise ValueError('Fixed disjoint reference pools required')
    table={};labels={};present=[];durations=[]
    engine=factory(cfg,jpeg_dir,baseline)
    for state in range(8):
        started=time.perf_counter()
        if state:
            paid=[(t,0,query_masks[t]) for j,t in enumerate(QUERY) if state&(1<<j)]
            # Every supplied mask belongs to this charged subset. One fresh replay
            # per subset; no uncharged gold initialization and no redundant prefixes.
            engine.prompts=paid[:-1]
            p=engine.correct(*paid[-1])
        else:p=baseline
        durations.append(time.perf_counter()-started)
        for j,t in enumerate(QUERY):table[state,j]=entropy(p[t,0])
        for t in SCORE:
            labels[state,t]=valid_overlap(p[t,0]>.5,score_masks[t],valid_maps[t])
    traces=[];curves=[]
    for policy,seeds in [('uncertainty',[100]),('uniform',[100]),('random',range(100,110))]:
        for seed in seeds:
            paths,history=trajectory(table,policy,seed)
            traces.append(dict(policy=policy,seed=seed,paths=paths,query_frames=history,charged_masks=3))
            for row in score_trajectory(paths,labels):curves.append(dict(policy=policy,seed=seed,**row))
    evaluation=[dict(state=s,frame=t,**c) for (s,t),c in labels.items()]
    observable=[dict(state=s,query_frame=QUERY[j],entropy=value) for (s,j),value in table.items()]
    return dict(curves=curves,traces=traces,evaluation=evaluation,observable=observable,
        propagation_calls=7,state_seconds=durations,engine_hardware=getattr(engine,'hardware',None),
        scoring_reference_summary=[dict(frame=t,valid_pixels=int(valid_maps[t].sum()),foreground_pixels=int((score_masks[t]&valid_maps[t]).sum())) for t in SCORE])
