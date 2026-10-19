"""Check the 1-based Task3 frame naming against decoded Task2 video pixels.

This is image/video alignment evidence, not procedure identity or clinical review.
"""
import concurrent.futures,csv,json,sys,time,hashlib
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    cv2.setNumThreads(1);start=time.perf_counter();base=ROOT/'data/raw/AutoLaparo_Task3'
    assets=list(csv.DictReader((ROOT/'data/manifests/local_assets.csv').open(encoding='utf8')))
    videos={Path(r['destination']).stem:ROOT/r['destination'] for r in assets if r['destination'].lower().endswith('.mp4') and 'autolaparo_task2' in r['destination'].lower()}
    frames={}
    for p in sorted((base/'imgs').glob('*.jpg')):frames.setdefault(p.stem[:3],[]).append(p)
    if len(frames)!=300 or len(videos)!=300:raise ValueError('Expected 300 clips')
    def one(clip):
        refs=frames[clip];targets={int(p.stem[3:])-1:cv2.resize(cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_COLOR),(128,72),interpolation=cv2.INTER_AREA).astype(np.float32)/255 for p in refs}
        needed={j for t in targets for j in [max(0,t-1),t,t+1]};observed={};cap=cv2.VideoCapture(str(videos[clip]));i=0
        if not cap.isOpened():raise ValueError('Video unreadable '+clip)
        fps=cap.get(cv2.CAP_PROP_FPS)
        while i<=max(needed):
            ok,bgr=cap.read()
            if not ok:raise ValueError('Video truncated '+clip)
            if i in needed:observed[i]=cv2.resize(bgr,(128,72),interpolation=cv2.INTER_AREA).astype(np.float32)/255
            i+=1
        cap.release();rows=[]
        for t,ref in targets.items():
            errors={j:float(np.abs(ref-observed[j]).mean()) for j in [max(0,t-1),t,t+1]};best=min(errors,key=errors.get);expected=errors[t];gap=expected-errors[best]
            status='expected_low_error' if expected<=.02 and gap<=.001 else 'needs_review'
            rows.append(dict(clip_id=clip,image=f'{clip}{t+1:03d}.jpg',frame_index=t,best_neighbour_frame=best,expected_normalized_rgb_mae=expected,best_normalized_rgb_mae=errors[best],expected_minus_best=gap,alignment_status=status,fps=fps))
        return rows
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for i,part in enumerate(ex.map(one,sorted(frames)),1):
            rows.extend(part)
            if i%25==0:print('Aligned',i,'/300 clips',flush=True)
    out=ROOT/'data/manifests/task3_frame_alignment.csv'
    with out.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(frames=len(rows),clips=len(frames),status_counts={s:sum(r['alignment_status']==s for r in rows) for s in ['expected_low_error','needs_review']},maximum_expected_rgb_mae=max(r['expected_normalized_rgb_mae'] for r in rows),seconds=time.perf_counter()-start,thresholds={'normalized_rgb_mae_max':.02,'expected_minus_best_neighbour_max':.001},note='Downsampled pixel check with +/-1 frame alternatives; does not verify procedure ID or annotation clinical accuracy')
    (out.with_suffix('.summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
