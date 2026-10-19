"""Export preselected, model-blind tail frames for clinician annotation.

Original video pixels/coordinates are retained; no pseudo-labels are generated.
"""
from pathlib import Path
import csv,concurrent.futures,json,shutil,threading
import cv2
ROOT=Path(__file__).resolve().parents[1];INDICES=[149,174,199,224]
def main():
    cv2.setNumThreads(1);out=ROOT/'clinical_annotation/tail_reference_v1';images=out/'images';images.mkdir(parents=True,exist_ok=True);lock=threading.Lock()
    assets=list(csv.DictReader((ROOT/'data/manifests/local_assets.csv').open(encoding='utf8')))
    videos={Path(r['destination']).stem:ROOT/r['destination'] for r in assets if r['destination'].lower().endswith('.mp4') and 'autolaparo_task2' in r['destination'].lower()}
    if len(videos)!=300:raise ValueError('Expected 300 source clips')
    def one(clip):
        rows=[];needed=[t for t in INDICES if not (images/f'{clip}_{t:05d}.jpg').exists()]
        if needed:
            if shutil.disk_usage(ROOT).free<700_000_000:raise RuntimeError('Keep 700 MB free; resume after freeing space')
            cap=cv2.VideoCapture(str(videos[clip]));cap.set(cv2.CAP_PROP_POS_FRAMES,min(needed));i=min(needed)
            while i<=max(needed):
                ok,bgr=cap.read()
                if not ok:raise ValueError('Video truncated '+clip)
                if i in needed:
                    ok,encoded=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
                    (images/f'{clip}_{i:05d}.jpg').write_bytes(encoded.tobytes())
                i+=1
            cap.release()
        for t in INDICES:
            for k,name in enumerate(['uterus','primary_tools_union']):
                rows.append(dict(clip_id=clip,procedure_id='',frame_index=t,timestamp_s=t/25,width=1920,height=1080,image=str((images/f'{clip}_{t:05d}.jpg').resolve()),structure=k,structure_name=name,visibility='',annotation_complete='',mask='',annotator='',adjudicator='',notes=''))
        return rows
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for i,part in enumerate(ex.map(one,sorted(videos)),1):
            rows.extend(part)
            if i%25==0:print('Annotation frames',i,'/300 clips',flush=True)
    with (out/'annotation_tasks.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'preparation.json').write_text(json.dumps(dict(clips=len(videos),frames=len(videos)*4,structure_tasks=len(rows),frame_indices_zero_based=INDICES,pseudo_labels=False,clinical_annotation_complete=False,coordinates='native 1920x1080',selection='Fixed before any model prediction on these frames'),indent=2),encoding='utf8');print('Prepared',len(rows),'blank annotation tasks')
if __name__=='__main__':main()
