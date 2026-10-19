"""Prepare offline correction evaluation from approved video, masks and TorchScript segmenter.

TorchScript must return TENSOR logits B,K,H,W for independent binary structures.
Do not pass a categorical joint-head model without an audited probability adapter.
"""
import argparse,json,csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .common import approved,fresh_out,sha
from .reference_scope import reference_scope_ready
ROOT=Path(__file__).resolve().parents[1]
def validate_metadata(meta):
    for key in ['procedure_id','clip_id','fps','n_structures','input_size','mean','std','query_ids','scoring_ids','group_verified','exposure_clean','permission_status','reference_format','split','clinical_scope_reviewed','human_annotation_complete']:
        if key not in meta:raise ValueError('Missing '+key)
    if not meta['procedure_id'] or not meta['clip_id']:raise ValueError('Missing verified source identifiers')
    if meta['group_verified'] is not True or meta['exposure_clean'] is not True or meta['permission_status']!='verified':raise ValueError('Grouping/exposure/permission unresolved')
    if not reference_scope_ready(meta) or meta['human_annotation_complete'] is not True:raise ValueError('Reference scope/completeness unresolved')
    if meta['split'] not in ['val','test']:raise ValueError('Unsupported evaluation split')
    if meta.get('official_split')=='test' and meta['split']!='test':raise ValueError('Official test cannot be relabelled as development')
    if meta['reference_format']!='binary_per_structure':raise ValueError('Reference masks must be binary and structure-specific')
    if meta['fps']<=0 or meta['n_structures']<=0:raise ValueError('Invalid sequence dimensions')
    for name in ['query_ids','scoring_ids']:
        ids=meta[name]
        if not ids or any(type(t) is not int or t<0 for t in ids) or len(set(ids))!=len(ids):raise ValueError('Invalid '+name)
    if set(meta['query_ids'])&set(meta['scoring_ids']):raise ValueError('Query/scoring overlap')
    if 'frame_count' in meta and (type(meta['frame_count']) is not int or meta['frame_count']<=max(meta['query_ids']+meta['scoring_ids'])):raise ValueError('Invalid declared frame count')

def validate_references(refs,meta):
    required={(t,k) for t in set(meta['query_ids'])|set(meta['scoring_ids']) for k in range(meta['n_structures'])}
    seen=set()
    for r in refs:
        t,k=int(r['frame_index']),int(r['structure'])
        if (t,k) not in required:raise ValueError('Reference outside declared query/scoring structures')
        if (t,k) in seen:raise ValueError('Duplicate reference')
        if r['annotation_complete'].lower()!='true':raise ValueError('Incomplete reference')
        if not r['mask'] or not Path(r['mask']).is_file():raise ValueError('Missing reference mask')
        seen.add((t,k))
    if required!=seen:raise ValueError('Reference masks missing for eligible query/scoring structures')

def main():
    a=argparse.ArgumentParser();a.add_argument('--video',required=True);a.add_argument('--segmenter',required=True);a.add_argument('--metadata',required=True);a.add_argument('--references',required=True);a.add_argument('--out',required=True);a.add_argument('--device',default='cuda');args=a.parse_args()
    meta=json.loads(Path(args.metadata).read_text());validate_metadata(meta)
    approved(ROOT,test=meta['split']=='test')
    with open(args.references,encoding='utf-8-sig') as f:refs=list(csv.DictReader(f))
    validate_references(refs,meta)
    for key in ['procedure_id','clip_id','fps','n_structures','input_size','mean','std','query_ids','scoring_ids','group_verified','exposure_clean','permission_status','reference_format']:
        if key not in meta:raise ValueError('Missing '+key)
    if meta['group_verified'] is not True or meta['exposure_clean'] is not True or meta['permission_status']!='verified':raise ValueError('Grouping/exposure/permission unresolved')
    if meta['reference_format']!='binary_per_structure':raise ValueError('Reference masks must be binary and structure-specific')
    if set(meta['query_ids'])&set(meta['scoring_ids']):raise ValueError('Query/scoring overlap')
    out=fresh_out(args.out);frames=out/'frames';frames.mkdir();model=torch.jit.load(args.segmenter,map_location=args.device).eval()
    import cv2
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened():raise ValueError('Cannot open video')
    if abs(cap.get(cv2.CAP_PROP_FPS)-meta['fps'])>.01:raise ValueError('Video fps differs from metadata')
    probs=[];i=0;mean=torch.tensor(meta['mean'],device=args.device).view(1,3,1,1);std=torch.tensor(meta['std'],device=args.device).view(1,3,1,1)
    with torch.inference_mode():
        while True:
            if 'frame_count' in meta and i>=meta['frame_count']:break
            ok,bgr=cap.read()
            if not ok:break
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);original_h,original_w=rgb.shape[:2]
            scale=min(1.,meta.get('working_max_side',720)/max(original_h,original_w));h,w=max(1,round(original_h*scale)),max(1,round(original_w*scale))
            rgb=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA);Image.fromarray(rgb).save(frames/f'{i:05d}.jpg',quality=95)
            x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float().to(args.device)/255
            x=torch.nn.functional.interpolate(x,size=(meta['input_size'],)*2,mode='bilinear',align_corners=False)
            logits=model((x-mean)/std)
            if not torch.is_tensor(logits) or logits.shape[1]!=meta['n_structures']:raise ValueError('Expected independent binary-channel tensor; adapt model explicitly')
            p=torch.nn.functional.interpolate(logits.sigmoid(),size=(h,w),mode='bilinear',align_corners=False)[0].cpu().numpy().astype(np.float16);probs.append(p);i+=1
    cap.release()
    if not probs:raise ValueError('Empty video')
    if 'frame_count' in meta and len(probs)!=meta['frame_count']:raise ValueError('Video shorter than declared window')
    if max(meta['query_ids']+meta['scoring_ids'])>=len(probs):raise ValueError('Declared reference frame beyond decoded video')
    p=np.stack(probs);gold=np.zeros_like(p,dtype=bool);seen=set()
    for r in refs:
        t,k=int(r['frame_index']),int(r['structure'])
        if r['annotation_complete'].lower()!='true':raise ValueError('Incomplete reference')
        if (t,k) in seen:raise ValueError('Duplicate reference')
        seen.add((t,k))
        with Image.open(r['mask']) as im:
            if im.size!=(original_w,original_h):raise ValueError('Reference original coordinates differ')
            raw=np.asarray(im)
            if raw.ndim!=2 or not set(np.unique(raw).tolist())<={0,1,255}:raise ValueError('Multiclass/color mask cannot be treated as structure-specific binary reference')
            g=np.asarray(im.resize((gold.shape[-1],gold.shape[-2]),Image.Resampling.NEAREST))>0
        gold[t,k]=g
    required={(t,k) for t in set(meta['query_ids'])|set(meta['scoring_ids']) for k in range(meta['n_structures'])}
    if not required<=seen:raise ValueError('Reference masks missing for eligible query/scoring structures')
    scope={key:str(meta.get(key,'')).lower() if isinstance(meta.get(key),bool) else str(meta.get(key,'')) for key in ['clinical_scope_reviewed','reference_review_mode','official_reference_scope_verified','official_annotation_completeness','reference_scope_evidence','reference_scope_evidence_sha256']}
    np.savez_compressed(out/'sequence.npz',probabilities=p,framewise=p,gold=gold,query_ids=meta['query_ids'],scoring_ids=meta['scoring_ids'],fps=meta['fps'],jpeg_dir=str(frames.resolve()),procedure_id=meta['procedure_id'],clip_id=meta['clip_id'],split=meta['split'],group_verified='true',exposure_clean='true',human_annotation_complete='true',permission_status='verified',**scope)
    (out/'provenance.json').write_text(json.dumps(dict(video_sha256=sha(args.video),segmenter_sha256=sha(args.segmenter),metadata=meta,references_sha256=sha(args.references),original_shape=[original_h,original_w],working_shape=list(gold.shape[-2:]),evaluation_coordinates='isotropically_downsampled_working_grid'),indent=2))
if __name__=='__main__':main()
