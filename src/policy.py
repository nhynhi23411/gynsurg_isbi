import numpy as np
from scipy.ndimage import distance_transform_edt
def features(prob,framewise,organ_ids,instrument_ids,sigma=.03,compute_proximity=True):
    p=np.clip(np.asarray(prob,dtype=np.float32),1e-6,1-1e-6)
    entropy=-(p*np.log(p)+(1-p)*np.log(1-p)).mean(axis=(-1,-2))/np.log(2)
    disagreement=np.abs(p-framewise).mean(axis=(-1,-2));prox=np.zeros(p.shape[:2],float);h,w=p.shape[-2:]
    for t in range(len(p)) if compute_proximity else []:
        organs=np.any(p[t,organ_ids]>.5,axis=0) if organ_ids else np.zeros((h,w),bool)
        tools=np.any(p[t,instrument_ids]>.5,axis=0) if instrument_ids else np.zeros((h,w),bool)
        for k in range(p.shape[1]):
            target=tools if k in organ_ids else organs;mask=p[t,k]>.5
            if target.any() and mask.any():
                d=distance_transform_edt(~target)[mask].min()/np.hypot(h,w);prox[t,k]=np.exp(-d/sigma)
    return entropy,disagreement,prox
def choose(prob,framewise,candidates,queries,policy,cfg,rng):
    if policy=='scaled_diversity_v1':
        scores=scaled_candidate_scores(prob,framewise,candidates,queries,cfg)
        best=max(scores,key=lambda row:(row['score'],-row['frame'],-row['structure']))
        return best['frame'],best['structure']
    if policy not in ['random','uniform','uncertainty','error','error_diversity','error_proximity','full']:raise ValueError('Unknown correction policy: '+str(policy))
    eligible=[q for q in candidates if q not in queries]
    if not eligible:raise ValueError('Budget exceeds query pool')
    if policy=='random':return eligible[int(rng.integers(len(eligible)))]
    if policy=='uniform':
        # Temporal max-min coverage separately per structure, interleave classes.
        counts={k:sum(q[1]==k for q in queries) for _,k in eligible};least=min(counts.values());eligible=[q for q in eligible if counts[q[1]]==least]
        def spread(q):
            same=[a for a,k in queries if k==q[1]]
            return min(abs(q[0]-a) for a in same) if same else min(q[0]+1,len(prob)-q[0])
        return max(eligible,key=lambda q:(spread(q),-q[0],-q[1]))
    # Scores are frame-local: evaluate only eligible frames, preserving exact scores.
    if not np.isfinite(cfg['sigma']) or cfg['sigma']<=0 or not np.isfinite(cfg['tau_s']) or cfg['tau_s']<=0:raise ValueError('Positive finite policy scales required')
    frame_ids=sorted({t for t,_ in eligible});lookup={t:i for i,t in enumerate(frame_ids)}
    ent,dis,prox=features(prob[frame_ids],framewise[frame_ids],cfg['organ_ids'],cfg['instrument_ids'],cfg['sigma'],compute_proximity=policy in ['full','error_proximity'])
    e=ent if policy=='uncertainty' else .5*(ent+dis)
    alpha=cfg['alpha'] if policy in ['full','error_proximity'] else 0
    beta=cfg['beta'] if policy in ['full','error_diversity'] else 0
    def score(q):
        t,k=q;same=[a for a,j in queries if j==k];red=max([np.exp(-abs(t-a)/(cfg['tau_s']*cfg['fps'])) for a in same],default=0.)
        return e[lookup[t],k]*(1+alpha*prox[lookup[t],k])-beta*red
    return max(eligible,key=lambda q:(score(q),-q[0],-q[1]))

def scaled_candidate_scores(prob,framewise,candidates,queries,cfg,frame_ids=None):
    lam=cfg['relative_diversity'];tau=cfg['tau_s'];fps=cfg['fps']
    if not np.isfinite(lam) or not 0<=lam<=1 or not np.isfinite(tau) or tau<=0 or not np.isfinite(fps) or fps<=0:raise ValueError('Invalid relative diversity parameters')
    eligible=[q for q in candidates if q not in queries]
    if not eligible:raise ValueError('Budget exceeds query pool')
    ids=sorted({t for t,_ in eligible});lookup={t:i for i,t in enumerate(ids)}
    # v2 disagreement is computed before entropy clipping; legacy features unchanged.
    if frame_ids is None:positions=ids
    else:
        source_index={int(t):i for i,t in enumerate(frame_ids)}
        if len(source_index)!=len(frame_ids):raise ValueError('Duplicate sparse frame IDs')
        positions=[source_index[t] for t in ids]
    raw=np.asarray(prob[positions],dtype=np.float32);f=np.asarray(framewise[positions],dtype=np.float32)
    clipped=np.clip(raw,1e-6,1-1e-6)
    ent=-(clipped*np.log(clipped)+(1-clipped)*np.log(1-clipped)).mean(axis=(-1,-2))/np.log(2)
    dis=np.abs(raw-f).mean(axis=(-1,-2));rows=[]
    for t,k in eligible:
        i=lookup[t];e=float(.5*(ent[i,k]+dis[i,k]));same=[a for a,j in queries if j==k]
        r=max([np.exp(-abs(t-a)/(tau*fps)) for a in same],default=0.)
        penalty=float(e*lam*r)
        rows.append(dict(frame=int(t),structure=int(k),entropy=float(ent[i,k]),disagreement=float(dis[i,k]),proximity=0.,redundancy=float(r),error_proxy=e,proximity_boost=0.,redundancy_penalty=penalty,score=float(e*(1-lam*r))))
    return rows
