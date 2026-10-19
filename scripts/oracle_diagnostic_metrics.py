"""CPU diagnostic primitives; feature API has no gold/reference argument."""
import numpy as np

def dice(a,b):
    a=np.asarray(a,bool);b=np.asarray(b,bool);den=a.sum()+b.sum()
    return float(2*(a&b).sum()/den) if den else 1.

def iou(a,b):
    a=np.asarray(a,bool);b=np.asarray(b,bool);union=(a|b).sum()
    return float((a&b).sum()/union) if union else 1.

def observable(raw,baseline,previous_state,previous_snapshot,query_distance_s):
    raw=np.asarray(raw);baseline=np.asarray(baseline);a=raw>.5;b=baseline>.5;roi=a|b
    if not roi.any():roi=np.ones(a.shape,bool)
    def entropy(x):
        p=np.clip(x[roi].astype(float),1e-6,1-1e-6)
        return float((-(p*np.log(p)+(1-p)*np.log(1-p))).mean()/np.log(2))
    hs,hf=entropy(raw),entropy(baseline)
    return dict(sam_entropy=hs,baseline_entropy=hf,entropy_advantage=hf-hs,roi_disagreement=float(np.abs(raw[roi]-baseline[roi]).mean()),log_area_ratio=float(np.log((a.sum()+1)/(b.sum()+1))),baseline_mask_iou=iou(a,b),state_change_iou=iou(a,np.asarray(previous_state)>.5),sparse_temporal_iou=iou(a,np.asarray(previous_snapshot)>.5),nearest_query_s=query_distance_s,raw_area=int(a.sum()),baseline_area=int(b.sum()))

def outcome(raw,baseline,gold):
    r=dice(np.asarray(raw)>.5,gold);b=dice(np.asarray(baseline)>.5,gold)
    return dict(raw_dice=r,baseline_dice=b,gain=r-b,restricted_oracle_dice=max(r,b),gold_present=bool(np.asarray(gold).any()))
