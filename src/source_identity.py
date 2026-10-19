"""Pixel-only temporal/source identity checks; separate from clinical evaluation."""
import numpy as np


def refine_window(frames, query, offsets=(0,24,49,74,99,124)):
    frames=np.asarray(frames);query=np.asarray(query);offsets=np.asarray(offsets)
    if frames.ndim!=4 or query.shape!=(len(offsets),)+frames.shape[1:]:
        raise ValueError('Incompatible RGB verification arrays')
    if len(frames)<=max(offsets) or min(offsets)<0:
        raise ValueError('Insufficient verification window')
    if frames.dtype!=np.uint8 or query.dtype!=np.uint8:
        raise ValueError('Expected original uint8 RGB samples')
    target=query.astype(np.float32)/255
    best=None
    for shift in range(len(frames)-max(offsets)):
        errors=np.abs(frames[shift+offsets].astype(np.float32)/255-target).mean((1,2,3))
        candidate=(float(errors.mean()),shift,errors)
        if best is None or candidate[0]<best[0]:best=candidate
    return best


def frame_correlations(frames,query):
    a=np.asarray(frames,dtype=np.float64).reshape(len(frames),-1)
    b=np.asarray(query,dtype=np.float64).reshape(len(query),-1)
    if a.shape!=b.shape:raise ValueError('Correlation shape mismatch')
    a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
    denominator=np.sqrt((a*a).sum(1)*(b*b).sum(1))
    return np.divide((a*b).sum(1),denominator,out=np.full(len(a),np.nan),where=denominator>0)

def identity_decision(candidates):
    """Conservative preregistered engineering thresholds, without clinical labels."""
    ranked=sorted(candidates,key=lambda x:x['six_frame_rgb_mae'])
    if not ranked:raise ValueError('No candidates')
    best=ranked[0]
    other=next((x for x in ranked if x['source_video_id']!=best['source_video_id']),None)
    margin=None if other is None else other['six_frame_rgb_mae']-best['six_frame_rgb_mae']
    ratio=None if other is None else other['six_frame_rgb_mae']/max(best['six_frame_rgb_mae'],1e-8)
    correlations=best.get('per_frame_rgb_correlation',[])
    passed=(other is not None and best['six_frame_rgb_mae']<=.02
            and best['max_frame_rgb_mae']<=.03 and margin>=.01 and ratio>=2
            and len(correlations)==6 and np.isfinite(correlations).all() and min(correlations)>=.99)
    return dict(best=best,second_source=other,source_margin=margin,source_ratio=ratio,
                pixel_identity_pass=bool(passed),threshold_version='v2_development_validated')
