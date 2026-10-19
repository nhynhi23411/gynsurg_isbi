"""CPU-only post-hoc diagnostics on saved development inputs; no propagation/test."""
from pathlib import Path
import datetime, json, modal
ROOT=Path(__file__).resolve().parents[1]
SOURCE='official_sparse_feasibility_20261016T144620Z'
IDS=['001','038','059','086','105','116','138','146','160']
app=modal.App('isbi-gynsurg-policy-diagnostics')
volume=modal.Volume.from_name('isbi-gynsurg-results-v1',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.11')
 .pip_install('numpy==1.26.4','scipy==1.15.2','Pillow==11.2.1')
 .add_local_file(str(ROOT/'src/policy.py'),'/opt/policy.py'))

@app.function(image=image,cpu=2,memory=8192,timeout=600,max_containers=3,volumes={'/results':volume})
def extract(cid):
    import sys,hashlib,csv,io
    import numpy as np
    from PIL import Image,ImageDraw
    from scipy.ndimage import distance_transform_edt
    sys.path.insert(0,'/opt')
    from policy import features
    if cid not in IDS:raise ValueError('Development-only clip allowlist')
    base=Path('/results')/SOURCE/cid
    execution=json.loads((base/'execution.json').read_text())
    if execution['official_split']!='train':raise ValueError('Not official training split')
    path=base/'prepared/sequence.npz'
    with np.load(path,allow_pickle=False) as d:
        p=d['framewise'];g=d['gold'];q=d['query_ids'];s=d['scoring_ids'];jpeg=str(d['jpeg_dir'])
    ids=list(map(int,np.concatenate([q,s])))
    small=p[ids];gold=g[ids];ent,dis,prox=features(small,small,[0],[1],.03)
    rows=[]
    for i,t in enumerate(ids):
        for k in range(2):
            a=small[i,k]>.5;b=gold[i,k].astype(bool);den=a.sum()+b.sum()
            target=gold[i,1-k].astype(bool)
            distance=float(distance_transform_edt(~target)[b].min()/np.hypot(*b.shape)) if b.any() and target.any() else None
            rows.append(dict(clip_id=cid,procedure_id=execution['procedure_id'],frame=t,structure=k,pool='query' if t in q else 'score',entropy=float(ent[i,k]),proximity=float(prox[i,k]),dice=float(2*(a&b).sum()/den) if den else 1.,gold_present=bool(b.any()),prediction_present=bool(a.any()),gold_distance=distance))
    # Reconstruct only the first decision, before any propagation, using predictions only.
    initial=[]
    for policy in ['uncertainty','error','error_proximity','error_diversity','full']:
        import csv
        log=list(csv.DictReader((base/'suite'/f'{policy}_seed20261016'/'queries.csv').open()))
        first=log[0];key=(int(first['frame']),int(first['structure']))
        r=next(x for x in rows if (x['frame'],x['structure'])==key)
        initial.append(dict(policy=policy,frame=key[0],structure=key[1],entropy=r['entropy'],proximity=r['proximity'],baseline_dice=r['dice'],recorded_queries=[(int(x['frame']),int(x['structure'])) for x in log]))
    panels=[]
    for t in ids:
        i=ids.index(t);canvas=Image.new('RGB',(960,242),'white');draw=ImageDraw.Draw(canvas)
        raw=Image.open(Path(jpeg)/f'{t:05d}.jpg').convert('RGB')
        for j,(title,masks) in enumerate([('Raw',None),('Baseline prediction',small[i]>.5),('Official reference',gold[i].astype(bool))]):
            im=raw.resize((320,180))
            if masks is not None:
                arr=np.asarray(im).copy()
                for k,color in enumerate([(255,90,30),(40,210,255)]):
                    mask=np.asarray(Image.fromarray(masks[k].astype('uint8')*255).resize((320,180),Image.Resampling.NEAREST))>0
                    arr[mask]=(.55*arr[mask]+.45*np.asarray(color)).astype('uint8')
                im=Image.fromarray(arr)
            canvas.paste(im,(j*320,35));draw.text((j*320+5,8),title,fill='black')
        draw.text((5,225),f'clip {cid} frame {t}; orange=uterus, cyan=tools; baseline only; official labels',fill='black')
        stream=io.BytesIO();canvas.save(stream,format='PNG');panels.append((f'{cid}_{t:03d}.png',stream.getvalue()))
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(clip_id=cid,input_sha256=digest,rows=rows,initial=initial),panels

@app.local_entrypoint()
def main():
    out=ROOT/'results'/('policy_diagnostics_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir();(out/'panels').mkdir();records=[]
    for record,panels in extract.map(IDS,order_outputs=False):
        records.append(record)
        for name,data in panels:(out/'panels'/name).write_bytes(data)
        print('DIAGNOSTIC_COMPLETE',record['clip_id'],flush=True)
    (out/'diagnostics.json').write_text(json.dumps(dict(source_run=SOURCE,scope='Post-hoc development diagnostics; no GPU propagation, no test, no new labels',records=records),indent=2),encoding='utf8')
    print(out,flush=True)
