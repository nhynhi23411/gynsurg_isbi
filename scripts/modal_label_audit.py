"""Read existing AutoLaparo labels on Modal; never modify the old volume."""
from pathlib import Path
import datetime,json
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('isbi-gynsurg-label-audit-v1',tags={'project':'isbi-gynsurg'})
source=modal.Volume.from_name('gynsurg',create_if_missing=False)
output=modal.Volume.from_name('isbi-gynsurg-results-v1',create_if_missing=True)
image=modal.Image.debian_slim(python_version='3.11').pip_install('numpy==1.26.4','Pillow==11.2.1')
@app.function(image=image,cpu=2,memory=4096,timeout=900,volumes={'/datasets':source,'/results':output})
def audit(run_id:str):
    import collections,concurrent.futures,hashlib,time
    import numpy as np
    from PIL import Image
    start=time.perf_counter();base=Path('/datasets/AutoLaparo_Dataset');images=sorted((base/'imgs').glob('*.jpg'));masks=sorted((base/'masks').glob('*.png'))
    def one(p):
        m=base/'masks'/(p.stem+'.png')
        with Image.open(p) as im:size=im.size
        with Image.open(m) as im:raw=np.asarray(im);mode=im.mode;masksize=im.size
        assert size==masksize,(p.name,size,masksize,raw.shape)
        rgb_equal=None
        if raw.ndim==3:
            rgb_equal=bool(raw.shape[2]==3 and np.array_equal(raw[...,0],raw[...,1]) and np.array_equal(raw[...,1],raw[...,2]))
            assert rgb_equal,('Non-gray RGB mask: explicit color LUT required',m.name)
            raw=raw[...,0]
        assert raw.ndim==2
        hist=np.bincount(raw.ravel(),minlength=256);vals=np.flatnonzero(hist);counts=hist[vals]
        return dict(image=p.name,mask=m.name,clip_id=p.stem[:3],frame_number_one_based=int(p.stem[3:]),width=size[0],height=size[1],mask_mode=mode,rgb_channels_equal=rgb_equal,values={str(int(v)):int(c) for v,c in zip(vals,counts)},image_sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),mask_sha256=hashlib.file_digest(m.open('rb'),'sha256').hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:rows=list(ex.map(one,images))
    counter=collections.Counter();clips=collections.defaultdict(list)
    for r in rows:counter.update(r['values']);clips[r['clip_id']].append(r['frame_number_one_based'])
    expected={1,25,50,75,100,125};bad={k:v for k,v in clips.items() if set(v)!=expected}
    report=dict(images=len(images),masks=len(masks),pairs=len(rows),clips=len(clips),palette_pixel_counts=dict(counter),sizes=dict(collections.Counter(f"{r['width']}x{r['height']}" for r in rows)),mask_modes=dict(collections.Counter(r['mask_mode'] for r in rows)),unexpected_frame_patterns=bad,expected_one_based_frames=sorted(expected),group_mapping_verified=False,semantic_mapping_requires_official_metadata_and_overlay_review=True,seconds=time.perf_counter()-start)
    out=Path('/results')/run_id;out.mkdir(parents=True,exist_ok=False);(out/'pairs.json').write_text(json.dumps(rows));(out/'audit.json').write_text(json.dumps(report,indent=2));output.commit();return report
@app.local_entrypoint()
def main():
    run_id='label_audit_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');report=audit.remote(run_id)
    (ROOT/'data/manifests/modal_task3_label_audit.json').write_text(json.dumps(dict(run_id=run_id,**report),indent=2),encoding='utf8');print(json.dumps(report,indent=2))
