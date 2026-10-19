"""Two approved development cases, two frozen policies; observational replay."""
from pathlib import Path
import datetime,json,modal
ROOT=Path(__file__).resolve().parents[1]
volume=modal.Volume.from_name('isbi-gynsurg-results-v1',create_if_missing=False)
sam=ROOT/'materials/code/extracted/sam2/sam2-main'
image=(modal.Image.debian_slim(python_version='3.11')
 .pip_install('torch==2.7.1','torchvision==0.22.1','numpy==1.26.4','scipy==1.15.2','pandas==2.2.3','matplotlib==3.10.1','Pillow==11.2.1','transformers==4.53.3','hydra-core==1.3.2','omegaconf==2.3.0','iopath==0.1.10','opencv-python-headless==4.11.0.86','tqdm==4.67.1')
 .env({'SAM2_BUILD_CUDA':'0','PYTHONPATH':'/opt/isbi','PYTHONHASHSEED':'20261016','HF_HUB_OFFLINE':'1'})
 .add_local_dir(str(sam),'/opt/sam2',copy=True,ignore=['.git','__pycache__'])
 .run_commands('pip install --no-deps --no-build-isolation /opt/sam2')
 .add_local_dir(str(ROOT/'src'),'/opt/isbi/src',ignore=['__pycache__'])
 .add_local_dir(str(ROOT/'configs'),'/opt/isbi/configs')
 .add_local_file(str(ROOT/'materials/checkpoints/sam2.1_hiera_tiny.pt'),'/opt/isbi/materials/checkpoints/sam2.1_hiera_tiny.pt')
 .add_local_file(str(ROOT/'materials/verification/official_sparse_scope_v1/scope_evidence.json'),'/opt/isbi/materials/verification/official_sparse_scope_v1/scope_evidence.json'))
app=modal.App('isbi-gynsurg-debug-replay-v1',tags={'project':'isbi-gynsurg'})
SOURCE='official_sparse_feasibility_20261016T144620Z'

@app.function(image=image,gpu='A10G',cpu=4,memory=24576,timeout=1800,max_containers=2,volumes={'/results':volume})
def replay(run_id,cid):
    import os,sys,subprocess,time,hashlib
    if cid not in ['059','086']:raise ValueError('Debug development allowlist failed')
    source=Path('/results')/SOURCE/cid;meta=json.loads((source/'execution.json').read_text())
    if meta['official_split']!='train':raise ValueError('Debug is development only')
    os.chdir('/opt/isbi');out=Path('/results')/run_id/cid;out.mkdir(parents=True,exist_ok=False)
    registry=[dict(run_id=p+'_seed20261016',config='configs/official_sparse_v1/'+p+'_seed20261016.json') for p in ['full','uncertainty']]
    registry_path=out/'debug_registry.json';registry_path.write_text(json.dumps(registry,indent=2))
    seq=source/'prepared/sequence.npz'
    with seq.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    old=json.loads((source/'suite/suite_provenance.json').read_text())
    if digest!=old['input_sha256']:raise ValueError('Saved input changed')
    start=time.perf_counter()
    cmd=[sys.executable,'-m','src.run_suite','--input',str(seq),'--registry',str(registry_path),'--output',str(out/'suite'),'--split','val','--cache-gib','4','--debug-trace']
    with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:result=subprocess.run(cmd,stdout=stdout,stderr=stderr)
    volume.commit()
    if result.returncode:raise RuntimeError((out/'stderr.txt').read_text()[-4000:])
    record=dict(clip_id=cid,source_run=SOURCE,input_sha256=digest,official_split='train',final_test=False,total_seconds=time.perf_counter()-start,debug_overhead_in_runtime=True)
    (out/'execution.json').write_text(json.dumps(record,indent=2));volume.commit()
    files=[p.relative_to(Path('/results')).as_posix() for p in out.rglob('*') if p.is_file() and p.suffix in ['.json','.csv','.png','.txt']]
    return record,files

@app.local_entrypoint()
def main():
    run_id='debug_replay_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'results'/run_id;out.mkdir();reports=[]
    for record,files in replay.starmap([(run_id,cid) for cid in ['059','086']],order_outputs=False):
        reports.append(record);print('REPLAY_COMPLETE',record['clip_id'],round(record['total_seconds'],2),flush=True)
        for remote in files:
            relative=Path(remote).relative_to(run_id);local=out/relative;local.parent.mkdir(parents=True,exist_ok=True)
            local.write_bytes(b''.join(volume.read_file(remote)))
    (out/'cohort_execution.json').write_text(json.dumps(dict(run_id=run_id,source_run=SOURCE,reports=reports,final_test=False),indent=2),encoding='utf8')
    print(out,flush=True)
