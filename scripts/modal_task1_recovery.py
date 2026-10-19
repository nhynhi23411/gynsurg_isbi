"""Recover source-video identity from a pinned public mirror, never from clip order.

No segmentation model, policy selection, clinical labels or endpoint evaluation.
Raw video and retrieval evidence remain on the user's private Modal volume.
"""
from pathlib import Path
import json, datetime
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('isbi-gynsurg-task1-recovery-v1', tags={'project': 'isbi-gynsurg'})
volume = modal.Volume.from_name('isbi-gynsurg-task1-recovery-v1', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11').apt_install('ffmpeg')
         .pip_install('requests==2.32.3', 'numpy==1.26.4', 'opencv-python-headless==4.11.0.86')
         .env({'OPENBLAS_NUM_THREADS': '2'})
         .add_local_file(str(ROOT/'data/manifests/task3_identity_queries_v1.npz'), '/opt/queries.npz'))

@app.function(image=image, cpu=4, memory=4096, timeout=10800, max_containers=4,
              volumes={'/recovery': volume})
def recover_video(manifest: dict, entry: dict):
    import hashlib, time, subprocess, os
    import numpy as np
    import requests
    from urllib.parse import quote
    started = time.perf_counter()
    vid = Path(entry['path']).stem
    folder = Path('/recovery')/manifest['revision']/vid
    folder.mkdir(parents=True, exist_ok=True)
    video = folder/(vid+'.mp4')
    receipt = folder/'download_verified.json'
    expected = entry['lfs']['oid']
    url = f"https://huggingface.co/datasets/{manifest['repo']}/resolve/{manifest['revision']}/{quote(entry['path'])}"
    if not (video.exists() and receipt.exists() and video.stat().st_size == entry['size']
            and json.loads(receipt.read_text()).get('sha256') == expected):
        partial = video.with_suffix('.mp4.part')
        for attempt in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == entry['size']:
                break
            try:
                # Cache buster avoids an expired signed Xet redirect; never log redirect URLs.
                r = requests.get(url+f'?download=true&attempt={attempt}&at={int(time.time())}',
                                 headers={'Range': f'bytes={offset}-'} if offset else {},
                                 stream=True, timeout=(30, 120))
                r.raise_for_status()
                if offset and r.status_code != 206:
                    offset = 0
                with partial.open('ab' if offset else 'wb') as f:
                    for chunk in r.iter_content(4*1024*1024):
                        f.write(chunk)
                r.close()
                if partial.stat().st_size == entry['size']:
                    break
            except requests.RequestException as e:
                print(vid, 'download retry', attempt, type(e).__name__, flush=True)
        if not partial.exists() or partial.stat().st_size != entry['size']:
            volume.commit()
            raise RuntimeError('Incomplete video '+vid)
        with partial.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if digest != expected:
            raise RuntimeError('Mirror SHA256 mismatch '+vid)
        partial.replace(video)
        receipt.write_text(json.dumps(dict(repo=manifest['repo'], revision=manifest['revision'],
            source_path=entry['path'], bytes=entry['size'], sha256=digest,
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2))
        volume.commit()
    print(vid, 'download SHA256 verified', entry['size'], flush=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                                               '-show_format', '-of', 'json', str(video)]))
    (folder/'ffprobe.json').write_text(json.dumps(probe, indent=2))
    stream = next(s for s in probe['streams'] if s['codec_type']=='video')
    assert stream['width']==1920 and stream['height']==1080, 'Unexpected mirror coordinates'
    # Full decode for temporal retrieval; no black-border crop or learned embeddings.
    index = folder/'rgb_2fps_32x18.npy'
    if not index.exists():
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(video),
            '-vf', 'fps=2:start_time=0,scale=32:18:flags=area', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', 'pipe:1'])
        frames = np.frombuffer(raw, np.uint8).reshape(-1,18,32,3)
        np.save(index, frames)
    frames = np.load(index)
    # Average 2x2 cells for fast RGB squared-distance retrieval.
    def features(a):
        return a.astype(np.float32).reshape(len(a),9,2,16,2,3).mean((2,4)).reshape(len(a),-1)/255
    f = features(frames)
    q = np.load('/opt/queries.npz')['rgb']
    q0, q5 = features(q[:,0]), features(q[:,5])
    lag = 10
    a,b = f[:-lag],f[lag:]
    distance = ((q0*q0).sum(1)[:,None] + (a*a).sum(1)[None] - 2*q0@a.T
              + (q5*q5).sum(1)[:,None] + (b*b).sum(1)[None] - 2*q5@b.T)/(2*f.shape[1])
    candidates=[]
    for clip in range(300):
        d=distance[clip].copy()
        for rank in range(3):
            t=int(d.argmin());score=float(d[t])
            candidates.append(dict(clip_id=f'{clip+1:03}', source_video_id=vid,
                coarse_start_s=t/2, coarse_rgb_mse=score, local_rank=rank,
                mirror_video_sha256=expected))
            d[max(0,t-10):min(len(d),t+11)]=np.inf
    (folder/'coarse_candidates.json').write_text(json.dumps(candidates, indent=2))
    report=dict(video_id=vid, bytes=video.stat().st_size, sha256=expected,
        duration_s=float(probe['format']['duration']), fps=stream['r_frame_rate'],
        coarse_frames=len(frames), seconds=time.perf_counter()-started,
        status='download_hash_verified_and_indexed', procedure_mapping_verified=False)
    (folder/'recovery_report.json').write_text(json.dumps(report, indent=2))
    volume.commit()
    print(vid, 'indexed', len(frames), 'frames', round(report['seconds'],1),'seconds', flush=True)
    return dict(report=report, candidates=candidates)

@app.local_entrypoint()
def main(video_ids: str = 'all'):
    manifest=json.loads((ROOT/'materials/verification/task1_mirror_manifest.json').read_text())
    selected=None if video_ids=='all' else set(video_ids.split(','))
    entries=[f for f in manifest['files'] if selected is None or Path(f['path']).stem in selected]
    output=ROOT/'materials/verification/task1_recovery_v1'
    output.mkdir(exist_ok=True)
    for result in recover_video.map([manifest]*len(entries),entries,order_outputs=False):
        vid=result['report']['video_id']
        (output/(vid+'.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
        print('RECOVERED',vid,json.dumps(result['report']),flush=True)
