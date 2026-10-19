"""Prediction-only heuristic fallback; no reference masks enter trust weights."""
import numpy as np

def trust_weight(sam,framewise,delta):
    if not np.isfinite(delta) or delta<=0:raise ValueError('Positive finite trust delta required')
    s=np.asarray(sam);f=np.asarray(framewise)
    if s.ndim!=2 or s.shape!=f.shape or not s.size:raise ValueError('Trust inputs must be matching nonempty 2D maps')
    if not np.isfinite(s).all() or not np.isfinite(f).all() or s.min()<0 or s.max()>1 or f.min()<0 or f.max()>1:raise ValueError('Invalid trust probabilities')
    roi=(s>.5)|(f>.5)
    if not roi.any():roi=np.ones(s.shape,bool)
    def entropy(p):
        p=np.clip(p[roi].astype(np.float64),1e-6,1-1e-6)
        return float(np.clip((-(p*np.log(p)+(1-p)*np.log(1-p))).mean()/np.log(2),0,1))
    hs,hf=entropy(s),entropy(f);dis=float(np.abs(s[roi]-f[roi]).mean())
    w=float(np.clip((1-hs)*np.exp(-dis*(1-hf)/delta),0,1))
    return w,dict(sam_entropy=hs,framewise_entropy=hf,disagreement=dis,weight=w,roi_pixels=int(roi.sum()))

class ReliabilityFallback:
    def __init__(self,engine,cfg,baseline):
        self.engine=engine;self.baseline=baseline.copy();self.delta=cfg['reliability_delta'];self.prompts=[];self.last_reliability_summary=[]
        if not np.isfinite(self.delta) or self.delta<=0:raise ValueError('Invalid reliability delta')

    def correct(self,t,k,mask):
        # This mask has already been selected and charged by the correction loop.
        self.prompts.append((int(t),int(k),np.asarray(mask,bool).copy()))
        raw=self.engine.correct(t,k,mask);result=self.baseline.copy();self.last_reliability_summary=[]
        if raw.shape!=result.shape:raise ValueError('Propagation output shape changed')
        for structure in sorted({q[1] for q in self.prompts}):
            for frame in range(len(raw)):
                w,summary=trust_weight(raw[frame,structure],self.baseline[frame,structure],self.delta)
                result[frame,structure]=w*raw[frame,structure]+(1-w)*self.baseline[frame,structure]
                charged=any(a==frame and b==structure for a,b,_ in self.prompts)
                self.last_reliability_summary.append(dict(frame=frame,structure=structure,charged_prompt_override=charged,**summary))
        for frame,structure,gold in self.prompts:result[frame,structure]=gold
        return result

def wrap_engine(engine,cfg,baseline):
    mode=cfg.get('propagation_reliability','raw')
    if mode=='raw':return engine
    if mode=='confidence_fallback_v1':return ReliabilityFallback(engine,cfg,baseline)
    raise ValueError('Unknown propagation reliability mode: '+str(mode))
