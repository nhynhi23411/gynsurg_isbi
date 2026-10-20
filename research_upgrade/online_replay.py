"""On-demand OFFLINE replay from charged prompts; no evaluator labels input.

Online here refers to sequential prompt purchases, not causal video streaming.
The video propagator still uses both temporal directions. Returned predictions
cover the frozen scoring snapshots only, not a dense validated deployment.
"""
import time
import numpy as np
from research_upgrade.bank import QUERY,SCORE,FRAMES,BUDGETS,PaidView,observable_features,entropy,popcount
from research_upgrade.acceptance_v2 import paired_features

def select_next(states,paid,policy,rng):
    eligible=[(j,k) for j in range(3) for k in range(2) if not paid[k]&(1<<j)]
    if policy=='random':return eligible[int(rng.integers(len(eligible)))]
    if policy=='uncertainty':return max(eligible,key=lambda z:(entropy(states[z[1],paid[z[1]]][FRAMES.index(QUERY[z[0]])]),-QUERY[z[0]],-z[1]))
    if policy=='uniform':
        least=min(popcount(paid[k]) for _,k in eligible);pool=[(j,k) for j,k in eligible if popcount(paid[k])==least]
        def spread(item):
            j,k=item;same=[QUERY[a] for a in range(3) if paid[k]&(1<<a)]
            return (min(abs(QUERY[j]-a) for a in same) if same else min(QUERY[j]+1,125-QUERY[j]),-QUERY[j],-k)
        return max(pool,key=spread)
    raise ValueError('Unsupported selector')

def run(baseline,engine,purchase_mask,policy='uncertainty',seed=20261017,model=None,feature_names=None,threshold=0.):
    """purchase_mask(frame,head) is invoked only AFTER the query is selected.

    Caller owns reference loading and scoring separately. No precomputed bank,
    evaluator Dice/IoU, unpurchased mask table, or procedure labels are accepted.
    Actual propagation count includes leave-one-prompt-out feature support.
    """
    baseline=np.asarray(baseline)
    if baseline.ndim!=4 or baseline.shape[:2]!=(125,2):raise ValueError('Expected frozen 125-frame two-head scope')
    if not np.isfinite(baseline).all() or baseline.min()<0 or baseline.max()>1:raise ValueError('Baseline must contain finite probabilities')
    if model is not None and not feature_names:raise ValueError('Frozen feature names required')
    start=time.perf_counter();paid=[0,0];anchors={};states={(k,0):baseline[list(FRAMES),k].copy() for k in range(2)}
    accepted={(k,t):0 for k in range(2) for t in SCORE};rng=np.random.default_rng(seed)
    history=[];decisions=[];calls=[];outputs={};purchase_seconds=0.;feature_seconds=0.
    def ensure(head,subset):
        if subset&~paid[head]:raise ValueError('Unpurchased prompt state requested')
        if (head,subset) in states:return
        prompts=[(QUERY[j],head,anchors[head,j]) for j in range(3) if subset&(1<<j)]
        engine.prompts=[(t,k,m.copy()) for t,k,m in prompts[:-1]]
        before=time.perf_counter();pred=engine.correct(*prompts[-1]);seconds=time.perf_counter()-before
        states[head,subset]=pred[list(FRAMES),head].copy()
        calls.append(dict(head=head,subset=subset,paid_at_execution=paid[head],seconds=seconds))
    def features(head,subset,frame):
        ensure(head,subset)
        if popcount(paid[head])>=2:
            for j in range(3):
                if paid[head]&(1<<j):ensure(head,subset&~(1<<j))
        return observable_features(PaidView(states,anchors,head,paid[head]),subset,frame)
    def snapshot(n):
        outputs[n]=np.stack([np.stack([states[k,accepted[k,t]][FRAMES.index(t)] for k in range(2)]) for t in SCORE])
    snapshot(0)
    for n in range(1,6):
        j,k=select_next(states,paid,policy,rng);t=QUERY[j]
        before=time.perf_counter();mask=np.asarray(purchase_mask(t,k));purchase_seconds+=time.perf_counter()-before
        if mask.shape!=baseline.shape[2:] or not np.isin(mask,[0,1]).all():raise ValueError('Purchased mask must be binary and match image size')
        anchors[k,j]=mask.astype(bool,copy=True);paid[k]|=1<<j
        history.append(dict(query=n,frame=t,head=k,paid_subset=paid[k]));ensure(k,paid[k])
        for frame in SCORE:
            old=accepted[k,frame];new=paid[k];value=None;use=True
            if model is not None and popcount(paid[k])>=2 and old!=new:
                before=time.perf_counter();a=features(k,new,frame);b=features(k,old,frame)
                vector=paired_features(a,b,feature_names)
                value=float(model.predict(vector[None,:])[0])
                if not np.isfinite(value):raise ValueError('Nonfinite gate output')
                use=value>threshold;feature_seconds+=time.perf_counter()-before
            if use:accepted[k,frame]=new
            decisions.append(dict(query=n,head=k,frame=frame,old_subset=old,new_subset=new,accepted=bool(use),score=value))
        if n in BUDGETS:snapshot(n)
    return dict(predictions=outputs,queries=history,decisions=decisions,propagations=calls,
        runtime=dict(total_seconds=time.perf_counter()-start,purchase_callback_seconds=purchase_seconds,feature_and_support_seconds=feature_seconds,
                     propagation_seconds=sum(c['seconds'] for c in calls)),
        scored_frame_indices=SCORE,paid_masks=len(history),precomputed_future_prompt_states=False,
        temporal_mode='offline_bidirectional',real_clinician_time_measured=False)
