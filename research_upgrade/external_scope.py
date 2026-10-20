"""Release-scoped CholecSeg8k decoding and window manifests, no inference."""
from pathlib import Path
import csv,hashlib,json
import numpy as np

KNOWN={5,11,12,13,21,22,23,24,25,31,32,33,50}
VOID={0,255}
QUERY=(0,15,31)
SCORE=(47,63,79)
BUDGETS=(0,1,2,3)
AUC_WEIGHTS=(1/6,1/3,1/3,1/6)

def decode_release_mask(mask):
    a=np.asarray(mask)
    if a.ndim==3:
        if a.shape[-1]!=3 or not np.array_equal(a[:,:,0],a[:,:,1]) or not np.array_equal(a[:,:,0],a[:,:,2]):
            raise ValueError('Expected equal-channel watershed RGB, not color visualization')
        a=a[:,:,0]
    if a.ndim!=2 or not np.issubdtype(a.dtype,np.integer):raise ValueError('Expected integer watershed pixels')
    unknown=set(map(int,np.unique(a)))-KNOWN-VOID
    if unknown:raise ValueError('Unregistered release codes: '+str(sorted(unknown)))
    valid=~np.isin(a,list(VOID));foreground=np.isin(a,[31,32])
    return foreground,valid

def valid_overlap(prediction,reference,valid):
    p=np.asarray(prediction,bool);r=np.asarray(reference,bool);v=np.asarray(valid,bool)
    if p.shape!=r.shape or r.shape!=v.shape:raise ValueError('Shape mismatch')
    if not v.any():return {'dice':None,'iou':None,'valid_pixels':0,'reference_present':None}
    p=p&v;r=r&v;inter=int((p&r).sum());total=int(p.sum()+r.sum());union=int((p|r).sum())
    return dict(dice=2*inter/total if total else 1.,iou=inter/union if union else 1.,valid_pixels=int(v.sum()),reference_present=bool(r.any()))

def build_windows(pairs):
    groups={}
    for row in pairs:groups.setdefault((row['source_video'],row['window']),[]).append(row)
    windows=[]
    for (source,window),rows in groups.items():
        rows=sorted(rows,key=lambda r:int(r['source_frame_index']));indices=[int(r['source_frame_index']) for r in rows]
        if len(rows)!=80 or indices!=list(range(indices[0],indices[0]+80)):raise ValueError('Noncontiguous or duplicate window')
        if any(not r['image_member'].startswith(source+'/') or not r['mask_member'].startswith(source+'/') for r in rows):raise ValueError('Source/path mismatch')
        windows.append(dict(source_video=source,window=window,source_start_frame=indices[0],frames=80,
            query_source_indices=[indices[i] for i in QUERY],score_source_indices=[indices[i] for i in SCORE],
            image_members=[r['image_member'] for r in rows],mask_members=[r['mask_member'] for r in rows]))
    return sorted(windows,key=lambda r:(int(r['source_video'].removeprefix('video')),r['source_start_frame']))

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];folder=root/'results/research_upgrade_external_20261020';path=folder/'archive_pairs.csv'
    with path.open(newline='',encoding='utf8') as f:windows=build_windows(csv.DictReader(f))
    if len(windows)!=101 or len({r['source_video'] for r in windows})!=17:raise ValueError('Wrong external cohort')
    result=dict(status='preparation_only_no_inference',pairs_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        query_relative_frames=QUERY,scoring_relative_frames=SCORE,budgets=BUDGETS,auc_weights=AUC_WEIGHTS,
        windows=windows,final_test=False)
    (folder/'window_manifest_draft_v1.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('Prepared 101 windows / 17 sources; no model inference')
