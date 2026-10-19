"""Diagnose bounded per-frame timestamp mismatch, without approving identity."""
from pathlib import Path
import json,modal
app=modal.App('isbi-gynsurg-identity-jitter-diagnostic-v4',tags={'project':'isbi-gynsurg'})
ROOT=Path(__file__).resolve().parents[1]
volume=modal.Volume.from_name('isbi-gynsurg-task1-recovery-v1',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.11').apt_install('ffmpeg')
 .pip_install('numpy==1.26.4','opencv-python-headless==4.11.0.86')
 .env({'OPENBLAS_NUM_THREADS':'2'})
 .add_local_file(str(ROOT/'data/manifests/task3_identity_verification_cv2_v2.npz'),'/opt/queries.npz')
 .add_local_file(str(ROOT/'src/source_identity.py'),'/opt/source_identity.py'))

@app.function(image=image,cpu=2,memory=3072,timeout=3600,max_containers=8,volumes={'/recovery':volume})
def diagnose(revision,vid,items,all_candidates=False):
    import cv2,numpy as np,sys,hashlib
    sys.path.insert(0,'/opt');from source_identity import frame_correlations
    folder=Path('/recovery')/revision/vid;receipt=json.loads((folder/'download_verified.json').read_text())
    cap=cv2.VideoCapture(str(folder/(vid+'.mp4')),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);q=np.load('/opt/queries.npz')['rgb'];cv2.setNumThreads(1)
    result=[];evidence=folder/('identity_temporal_v5' if all_candidates else 'identity_jitter_diagnostic_v4');evidence.mkdir(exist_ok=True)
    for c in items:
        start=c['refined_start_frame'];lo=max(0,start-25);hi=start+150;cap.set(cv2.CAP_PROP_POS_FRAMES,lo);frames=[]
        for _ in range(lo,hi):
            ok,bgr=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(cv2.resize(bgr,(128,72),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
        a=np.asarray(frames);target=q[int(c['clip_id'])-1];selected=[];errors=[]
        for j,off in enumerate([0,24,49,74,99,124]):
            # Wider first-frame diagnostic is recorded separately; eligibility uses <=5 below.
            radius=25 if j==0 else 5
            ts=np.arange(max(0,start+off-radius-lo),min(len(a),start+off+radius-lo+1))
            distances=np.abs(a[ts].astype(np.float32)/255-target[j].astype(np.float32)/255).mean((1,2,3));i=int(distances.argmin())
            selected.append(int(ts[i]+lo));errors.append(float(distances[i]))
        chosen=a[np.array(selected)-lo];corr=frame_correlations(chosen,target).tolist()
        panel=np.concatenate([np.concatenate(list(target),1),np.concatenate(list(chosen),1)],0);path=evidence/(c['clip_id']+('_'+str(start) if all_candidates else '')+'.png');cv2.imwrite(str(path),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
        gaps=np.diff(selected);temporal_pass=bool((gaps>=10).all() and (gaps<=50).all() and 100<=selected[-1]-selected[0]<=160)
        mean=float(np.mean(errors));max_error=max(errors)
        record=dict(c);record.update(dict(nominal_start=start,matched_frames=selected,
          jitter=[t-start-o for t,o in zip(selected,[0,24,49,74,99,124])],frame_mae=errors,frame_correlations=corr,
          source_sha256=receipt['sha256'],evidence_path=str(path.relative_to('/recovery')),
          bounded_all_six=all(abs(t-start-o)<=5 for t,o in zip(selected,[0,24,49,74,99,124])),
          temporal_pass=temporal_pass,identity_approved=False));result.append(record)
        if all_candidates:
            result[-1].update(six_frame_rgb_mae=mean if temporal_pass else 1.0,max_frame_rgb_mae=max_error if temporal_pass else 1.0,
               per_frame_rgb_mae=errors,per_frame_rgb_correlation=corr,source_video_sha256=receipt['sha256'],
               identity_model='bounded_monotonic_temporal_v5')
        if len(result)%40==0:print(vid,'temporal',len(result),'/',len(items),flush=True)
    cap.release();volume.commit();return result

@app.local_entrypoint()
def main(all_candidates:bool=False):
    rows=json.loads((ROOT/'data/manifests/source_identity_recovery_v3/all_decisions.json').read_text())
    # Include every clip's best source, not only failed IDs. Diagnostic, not acceptance.
    by_source={}
    if all_candidates:
        for p in (ROOT/'materials/verification/task1_refinement_v3').glob('??.json'):
            by_source[p.stem]=json.loads(p.read_text())['candidates']
    else:
        for r in rows:by_source.setdefault(r['best']['source_video_id'],[]).append(r['best'])
    manifest=json.loads((ROOT/'materials/verification/task1_mirror_manifest.json').read_text());output=ROOT/('materials/verification/task1_temporal_identity_v5' if all_candidates else 'materials/verification/task1_jitter_diagnostic_v4');output.mkdir(exist_ok=True)
    for result in diagnose.starmap([(manifest['revision'],v,items,all_candidates) for v,items in by_source.items()],order_outputs=False):
        vid=result[0]['source_video_id'];report=dict(video_id=vid,candidates=result,query_sha256=__import__('hashlib').file_digest((ROOT/'data/manifests/task3_identity_verification_cv2_v2.npz').open('rb'),'sha256').hexdigest()) if all_candidates else result
        (output/(vid+'.json')).write_text(json.dumps(report,indent=2));print('DIAGNOSED',vid,len(result),flush=True)
