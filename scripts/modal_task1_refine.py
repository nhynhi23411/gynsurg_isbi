"""Independent six-frame pixel verification of coarse source identity candidates."""
from pathlib import Path
import json
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('isbi-gynsurg-task1-refine-v1',tags={'project':'isbi-gynsurg'})
volume=modal.Volume.from_name('isbi-gynsurg-task1-recovery-v1',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.11').apt_install('ffmpeg')
       .pip_install('numpy==1.26.4','opencv-python-headless==4.11.0.86')
       .add_local_file(str(ROOT/'data/manifests/task3_identity_verification_cv2_v2.npz'),'/opt/queries.npz')
       .add_local_file(str(ROOT/'src/source_identity.py'),'/opt/source_identity.py'))

@app.function(image=image,cpu=2,memory=2048,timeout=7200,max_containers=8,volumes={'/recovery':volume})
def refine_video(revision:str,vid:str,candidates:list):
    import cv2,numpy as np,time,sys,hashlib
    sys.path.insert(0,'/opt')
    from source_identity import refine_window,frame_correlations
    cv2.setNumThreads(1)
    folder=Path('/recovery')/revision/vid;video=folder/(vid+'.mp4')
    receipt=json.loads((folder/'download_verified.json').read_text())
    q=np.load('/opt/queries.npz')['rgb'];offsets=np.array([0,24,49,74,99,124])
    cap=cv2.VideoCapture(str(video),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);fps=cap.get(cv2.CAP_PROP_FPS)
    assert abs(fps-25)<.001, 'Need explicit timebase alignment for a transcoded mirror'
    start=time.perf_counter();rows=[]
    for candidate in candidates:
        cid=int(candidate['clip_id']);guess=round(candidate['coarse_start_s']*fps)
        lo=max(0,guess-30);hi=guess+30
        cap.set(cv2.CAP_PROP_POS_FRAMES,lo)
        assert abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-lo)<.5
        frames=[]
        for i in range(lo,hi+125):
            ok,bgr=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(cv2.resize(bgr,(128,72),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
        a=np.asarray(frames)
        mean,shift,errors=refine_window(a,q[cid-1],offsets)
        evidence=folder/'identity_evidence';evidence.mkdir(exist_ok=True)
        name=f'{cid:03}_{lo+shift:07}.png'
        panel=np.concatenate([np.concatenate(list(q[cid-1]),axis=1),
                              np.concatenate(list(a[shift+offsets]),axis=1)],axis=0)
        assert cv2.imwrite(str(evidence/name),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
        rows.append(dict(**candidate, refined_start_frame=lo+shift,
            refined_start_s=(lo+shift)/fps, six_frame_rgb_mae=mean,
            max_frame_rgb_mae=float(errors.max()), per_frame_rgb_mae=errors.tolist(),
            per_frame_rgb_correlation=frame_correlations(a[shift+offsets],q[cid-1]).tolist(),
            source_video_sha256=receipt['sha256'],source_coordinate_shape=[1920,1080],
            verification_shape=[128,72],verification_offsets=offsets.tolist(),
            evidence_path=str((evidence/name).relative_to('/recovery'))))
        if len(rows)%25==0:print(vid,'refined',len(rows),'/',len(candidates),flush=True)
    cap.release()
    result=dict(video_id=vid,seconds=time.perf_counter()-start,candidates=rows,
                query_sha256=hashlib.file_digest(open('/opt/queries.npz','rb'),'sha256').hexdigest(),resampling='OpenCV INTER_AREA both sides',
                segmentation_evaluation=False,procedure_mapping_approved=False)
    (folder/'refined_candidates.json').write_text(json.dumps(result,indent=2));volume.commit()
    return result

@app.local_entrypoint()
def main(top_k:int=6,smoke:bool=False):
    manifest=json.loads((ROOT/'materials/verification/task1_mirror_manifest.json').read_text())
    coarse=list((ROOT/'materials/verification/task1_recovery_v1').glob('??.json'))
    if not smoke:assert len(coarse)==21, 'Compare all source videos before candidate pruning'
    if smoke:
        args=[]
        for file in coarse:
            candidates=json.loads(file.read_text())['candidates']
            selected=sorted([c for c in candidates if c['local_rank']==0 and int(c['clip_id'])<=170],key=lambda c:c['coarse_rgb_mse'])[:4]
            args.append((manifest['revision'],file.stem,selected))
        out=ROOT/'materials/verification/task1_refinement_smoke_cv2_v2';out.mkdir(exist_ok=True)
        for result in refine_video.starmap(args,order_outputs=False):
            result['scope']='retrieval engineering smoke only; not global source assignment'
            (out/(result['video_id']+'.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
            print('REFINEMENT_SMOKE',result['video_id'],[round(c['six_frame_rgb_mae'],6) for c in result['candidates']],flush=True)
        return
    by_clip={f'{i:03}':[] for i in range(1,301)}
    for file in coarse:
        for c in json.loads(file.read_text())['candidates']:by_clip[c['clip_id']].append(c)
    by_video={f'{i:02}':[] for i in range(1,22)}
    for cid,items in by_clip.items():
        ranked=sorted(items,key=lambda x:x['coarse_rgb_mse'])
        selected=ranked[:top_k]
        # Include the best candidate from a different source for an identity margin.
        alternate=next(x for x in ranked if x['source_video_id']!=ranked[0]['source_video_id'])
        if alternate not in selected:selected.append(alternate)
        for c in selected:by_video[c['source_video_id']].append(c)
    args=[(manifest['revision'],v,c) for v,c in by_video.items() if c]
    out=ROOT/'materials/verification/task1_refinement_v1';out.mkdir(exist_ok=True)
    for result in refine_video.starmap(args,order_outputs=False):
        (out/(result['video_id']+'.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
        print('REFINED',result['video_id'],len(result['candidates']),flush=True)
