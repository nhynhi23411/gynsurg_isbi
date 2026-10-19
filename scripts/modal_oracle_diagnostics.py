"""Extract small CPU diagnostics from private saved raw SAM states."""
from pathlib import Path
import datetime,json,modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('isbi-gynsurg-oracle-diagnostics-cpu')
volume=modal.Volume.from_name('isbi-gynsurg-results-v1',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.11').pip_install('numpy==1.26.4')
 .add_local_file(str(ROOT/'scripts/oracle_diagnostic_metrics.py'),'/opt/oracle_diagnostic_metrics.py'))
SOURCE='reliability_scaled_a10_development_20261017T031408Z'
INPUT='official_sparse_feasibility_20261016T144620Z'
IDS=['001','038','059','086','105','116','138','146','160']

@app.function(image=image,cpu=2,memory=8192,timeout=600,max_containers=3,volumes={'/results':volume})
def extract(cid):
    import sys,csv,hashlib
    import numpy as np
    sys.path.insert(0,'/opt');from oracle_diagnostic_metrics import observable,outcome
    if cid not in IDS:raise ValueError('Development allowlist failed')
    base=Path('/results')/SOURCE/cid;execution=json.loads((base/'execution.json').read_text())
    if execution['official_split']!='train':raise ValueError('Not official training split')
    path=Path('/results')/INPUT/cid/'prepared/sequence.npz'
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=execution['input_sha256']:raise ValueError('Input hash mismatch')
    with np.load(path,allow_pickle=False) as d:
        f=d['framewise'];g=d['gold'];scoring=list(map(int,d['scoring_ids']));fps=float(d['fps'])
    folder=base/'suite/uncertainty_raw';queries=list(csv.DictReader((folder/'queries.csv').open()));rows=[];states=[];state_hashes={}
    for n in range(6):
        p=folder/'debug'/f'state_{n:02d}.npz'
        with p.open('rb') as stream:state_hashes[p.name]=hashlib.file_digest(stream,'sha256').hexdigest()
        with np.load(p,allow_pickle=False) as d:ids=list(map(int,d['frame_ids']));current=d['probabilities']
        if states:previous=states[-1]
        else:previous=current
        charged=[(int(q['frame']),int(q['structure'])) for q in queries[:n]]
        for t in scoring:
            i=ids.index(t);prior=ids[i-1]
            for k in range(2):
                same=[abs(t-a)/fps for a,j in charged if j==k]
                feature=observable(current[i,k],f[t,k],previous[i,k],current[i-1,k],min(same) if same else None)
                rows.append(dict(clip_id=cid,procedure_id=execution['procedure_id'],query_count=n,frame=t,structure=k,structure_prompted=bool(same),previous_snapshot_frame=prior,previous_snapshot_gap_s=(t-prior)/fps,previous_snapshot_charged=(prior,k) in charged,**feature,**outcome(current[i,k],f[t,k],g[t,k])))
        states=[current] # only previous correction state needed in memory
    # Verify outcome extraction against already accepted debug metrics.
    saved=list(csv.DictReader((folder/'debug/state_metrics.csv').open()));lookup={(int(r['query_count']),int(r['frame']),int(r['structure'])):float(r['dice']) for r in saved}
    if any(abs(r['raw_dice']-lookup[(r['query_count'],r['frame'],r['structure'])])>1e-12 for r in rows):raise ValueError('Extracted metrics disagree with saved trace')
    return dict(clip_id=cid,input_sha256=digest,state_sha256=state_hashes,rows=rows)

@app.local_entrypoint()
def main():
    import csv,hashlib
    out=ROOT/'results'/('oracle_diagnostics_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir();records=[]
    for result in extract.map(IDS,order_outputs=False):records.append(result);print('CPU_DIAGNOSTIC',result['clip_id'],flush=True)
    rows=[r for case in records for r in case['rows']]
    with (out/'observations.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    receipts=[{k:v for k,v in case.items() if k!='rows'} for case in records]
    (out/'provenance.json').write_text(json.dumps(dict(source_run=SOURCE,input_run=INPUT,cpu_only=True,model_rerun=False,final_test_open=False,cases=receipts,diagnostic_code_sha256=hashlib.file_digest((ROOT/'scripts/oracle_diagnostic_metrics.py').open('rb'),'sha256').hexdigest()),indent=2),encoding='utf8')
    print(out,flush=True)
