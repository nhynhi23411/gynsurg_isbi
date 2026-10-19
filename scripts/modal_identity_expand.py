"""Expanded six-frame retrieval for failed IDs; original reports remain immutable."""
from pathlib import Path
import json,modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('isbi-gynsurg-identity-expand-v3',tags={'project':'isbi-gynsurg'})
volume=modal.Volume.from_name('isbi-gynsurg-task1-recovery-v1',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.11').apt_install('ffmpeg')
 .pip_install('numpy==1.26.4','opencv-python-headless==4.11.0.86')
 .env({'OPENBLAS_NUM_THREADS':'2'})
 .add_local_file(str(ROOT/'data/manifests/task3_identity_verification_cv2_v2.npz'),'/opt/queries.npz')
 .add_local_file(str(ROOT/'src/source_identity.py'),'/opt/source_identity.py'))

@app.function(image=image,cpu=2,memory=2048,timeout=900,max_containers=8,volumes={'/recovery':volume})
def retrieve(revision,vid,ids):
    import numpy as np,cv2
    folder=Path('/recovery')/revision/vid
    f=np.load(folder/'rgb_2fps_32x18.npy').astype(np.float32).reshape(-1,18*32*3)/255
    q=np.load('/opt/queries.npz')['rgb']
    qq=np.array([[cv2.resize(x,(32,18),interpolation=cv2.INTER_AREA) for x in q[int(cid)-1]] for cid in ids],dtype=np.float32).reshape(len(ids),6,-1)/255
    distance=np.zeros((len(ids),len(f)-10),np.float32)
    # Every annotated sample contributes; no expected source IDs/counts in selection.
    for j,lag in enumerate([0,2,4,6,8,10]):
        a=f[lag:len(f)-10+lag];b=qq[:,j]
        distance+=((b*b).sum(1)[:,None]+(a*a).sum(1)[None]-2*b@a.T)/(6*f.shape[1])
    rows=[];receipt=json.loads((folder/'download_verified.json').read_text())
    for i,cid in enumerate(ids):
        d=distance[i].copy()
        for rank in range(12):
            t=int(d.argmin())
            rows.append(dict(clip_id=cid,source_video_id=vid,coarse_start_s=t/2,
              coarse_rgb_mse=float(d[t]),local_rank=100+rank,mirror_video_sha256=receipt['sha256'],retrieval_version='six_frame_v3'))
            d[max(0,t-4):min(len(d),t+5)]=np.inf
    return dict(video_id=vid,candidates=rows)

@app.function(image=image,cpu=2,memory=3072,timeout=7200,max_containers=8,volumes={'/recovery':volume})
def refine(revision,vid,candidates):
    import numpy as np,cv2,sys,hashlib,time
    sys.path.insert(0,'/opt')
    from source_identity import refine_window,frame_correlations
    folder=Path('/recovery')/revision/vid;receipt=json.loads((folder/'download_verified.json').read_text())
    q=np.load('/opt/queries.npz')['rgb'];offsets=np.array([0,24,49,74,99,124]);rows=[]
    cv2.setNumThreads(1);cap=cv2.VideoCapture(str(folder/(vid+'.mp4')),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
    assert abs(cap.get(cv2.CAP_PROP_FPS)-25)<.001
    evidence=folder/'identity_evidence_v3';evidence.mkdir(exist_ok=True);start=time.perf_counter()
    for c in candidates:
        guess=round(c['coarse_start_s']*25);lo=max(0,guess-50);hi=guess+50
        cap.set(cv2.CAP_PROP_POS_FRAMES,lo);frames=[]
        for _ in range(lo,hi+125):
            ok,bgr=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(cv2.resize(bgr,(128,72),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
        a=np.asarray(frames);target=q[int(c['clip_id'])-1]
        mean,shift,errors=refine_window(a,target,offsets)
        selected=a[shift+offsets];name=f"{c['clip_id']}_{lo+shift:07}.png"
        panel=np.concatenate([np.concatenate(list(target),1),np.concatenate(list(selected),1)],0)
        assert cv2.imwrite(str(evidence/name),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
        # Diagnostic only: test individual temporal jitter without changing decision rules.
        jitter=[]
        for j,off in enumerate(offsets):
            times=range(max(0,shift+int(off)-3),min(len(a),shift+int(off)+4))
            err=[float(np.abs(a[t].astype(np.float32)/255-target[j].astype(np.float32)/255).mean()) for t in times]
            at=int(np.argmin(err));jitter.append(dict(frame=lo+list(times)[at],mae=err[at]))
        rows.append(dict(**c,refined_start_frame=lo+shift,refined_start_s=(lo+shift)/25,
          six_frame_rgb_mae=mean,max_frame_rgb_mae=float(errors.max()),per_frame_rgb_mae=errors.tolist(),
          per_frame_rgb_correlation=frame_correlations(selected,target).tolist(),
          source_video_sha256=receipt['sha256'],evidence_path=str((evidence/name).relative_to('/recovery')),
          verification_offsets=offsets.tolist(),verification_shape=[128,72],jitter_diagnostic=jitter))
        if len(rows)%20==0:print(vid,'expanded refine',len(rows),'/',len(candidates),flush=True)
    cap.release()
    result=dict(video_id=vid,candidates=rows,seconds=time.perf_counter()-start,
       query_sha256=hashlib.file_digest(open('/opt/queries.npz','rb'),'sha256').hexdigest(),
       threshold_changed=False,clinical_segmentation_evaluation=False)
    (folder/'expanded_refinement_v3.json').write_text(json.dumps(result,indent=2));volume.commit();return result

@app.local_entrypoint()
def main():
    import csv
    manifest=json.loads((ROOT/'materials/verification/task1_mirror_manifest.json').read_text())
    rows=list(csv.DictReader((ROOT/'data/manifests/source_identity_recovery_v2/candidate_mapping.csv').open(encoding='utf8')))
    ids=[r['clip_id'] for r in rows if r['pixel_identity_pass']!='True']
    coarse=ROOT/'materials/verification/task1_expanded_retrieval_v3';coarse.mkdir(exist_ok=True)
    output=ROOT/'materials/verification/task1_refinement_v3';output.mkdir(exist_ok=True)
    by_clip={cid:[] for cid in ids}
    for result in retrieve.starmap([(manifest['revision'],f'{i:02}',ids) for i in range(1,22)],order_outputs=False):
        (coarse/(result['video_id']+'.json')).write_text(json.dumps(result,indent=2))
        for c in result['candidates']:by_clip[c['clip_id']].append(c)
    by_source={}
    for cid,items in by_clip.items():
        ranked=sorted(items,key=lambda c:c['coarse_rgb_mse']);chosen=ranked[:18]
        alt=next(c for c in ranked if c['source_video_id']!=ranked[0]['source_video_id'])
        if alt not in chosen:chosen.append(alt)
        for c in chosen:by_source.setdefault(c['source_video_id'],[]).append(c)
    fresh={}
    for result in refine.starmap([(manifest['revision'],vid,items) for vid,items in by_source.items()],order_outputs=False):
        fresh[result['video_id']]=result
        print('EXPANDED_REFINED',result['video_id'],len(result['candidates']),flush=True)
        (output/(result['video_id']+'.expanded.json')).write_text(json.dumps(result,indent=2))
    # Include all original shortlisted candidates for the evaluator completeness gate.
    for i in range(1,22):
        vid=f'{i:02}';old=ROOT/f'materials/verification/task1_refinement_v1/{vid}.json'
        previous=json.loads(old.read_text()) if old.exists() else None
        added=fresh.get(vid)
        if previous or added:
            merged=dict(previous or added);merged['candidates']=(previous['candidates'] if previous else [])+(added['candidates'] if added else [])
            merged['expanded_retrieval_version']='v3';(output/(vid+'.json')).write_text(json.dumps(merged,indent=2))
    print('Expanded reports complete; thresholds unchanged; mapping NOT approved',flush=True)
