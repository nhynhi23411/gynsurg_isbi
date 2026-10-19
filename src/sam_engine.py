import numpy as np
import hashlib,json
from contextlib import nullcontext
from pathlib import Path
def _file_sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class SAM2Engine:
    """Replay charged full-mask corrections, with no hidden gold initialization.
    Framewise predictions are the zero-cost baseline. Only corrected structures
    receive SAM2 propagation. Offline, independent structure states avoid mutual
    object suppression. Replay all corrections for deterministic budget comparison.
    """
    def __init__(self,cfg,jpeg_dir,baseline,cache=None,predictor=None):
        self.precision=cfg.get('sam_precision','float32')
        if self.precision not in ['float32','bfloat16']:raise ValueError('Unsupported SAM precision')
        if self.precision=='bfloat16' and not str(cfg['device']).startswith('cuda'):raise ValueError('bfloat16 mode requires CUDA in this adapter')
        if len(list(Path(jpeg_dir).glob('*.jpg')))!=len(baseline):raise ValueError('SAM video frame count differs from baseline')
        from .runtime_contract import validate_hardware
        self.hardware=validate_hardware(cfg)
        if predictor is None:
            from sam2.build_sam import build_sam2_video_predictor
            predictor=build_sam2_video_predictor(cfg['sam_config'],cfg['sam_checkpoint'],device=cfg['device'],apply_postprocessing=cfg.get('sam_postprocessing',False))
        self.predictor=predictor;self.jpeg_dir=jpeg_dir;self.baseline=baseline;self.prompts=[];self.cache=cache
        if cache is not None:
            checkpoint=Path(cfg['sam_checkpoint'])
            context=dict(jpeg_dir=str(Path(jpeg_dir).resolve()),baseline_sha256=hashlib.sha256(np.ascontiguousarray(baseline).tobytes()).hexdigest(),
                shape=list(baseline.shape),dtype=str(baseline.dtype),sam_config=cfg['sam_config'],device=cfg['device'],postprocessing=cfg.get('sam_postprocessing',False),precision=self.precision,
                checkpoint_sha256=_file_sha(checkpoint),engine_source_sha256=_file_sha(__file__),hardware=self.hardware,
                jpeg_hashes=[(p.name,_file_sha(p)) for p in sorted(Path(jpeg_dir).glob('*.jpg'))])
            self.context_key=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    def correct(self,t,k,mask):
        import torch
        self.prompts.append((int(t),int(k),np.asarray(mask,bool).copy()));result=self.baseline.copy()
        with torch.inference_mode(), (torch.autocast('cuda',dtype=torch.bfloat16) if self.precision=='bfloat16' else nullcontext()):
            for key in sorted({q[1] for q in self.prompts}):
                prompts=sorted([q for q in self.prompts if q[1]==key],key=lambda q:q[0])
                cache_key=None
                if self.cache is not None:
                    # Every mask in this key has already been selected and charged.
                    cache_key=(self.context_key,key,tuple((frame,hashlib.sha256(gold.tobytes()).hexdigest()) for frame,_,gold in prompts))
                    cached=self.cache.get(cache_key)
                    if cached is not None:
                        result[:,key]=cached
                        continue
                state=self.predictor.init_state(video_path=self.jpeg_dir,offload_video_to_cpu=True)
                for frame,_,gold in prompts:self.predictor.add_new_mask(state,frame_idx=frame,obj_id=1,mask=gold)
                first=prompts[0][0]
                for reverse in [False,True]:
                    for frame,ids,logits in self.predictor.propagate_in_video(state,start_frame_idx=first,reverse=reverse):
                        result[frame,key]=logits[0,0].sigmoid().float().cpu().numpy()
                # full masks on charged query frames remain exact, although never scored.
                for frame,_,gold in prompts:result[frame,key]=gold
                del state
                if self.cache is not None:self.cache.put(cache_key,result[:,key])
        return result
