"""Explicit development launcher for frozen v2; preparation never calls this."""
from pathlib import Path
import datetime,json,modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('isbi-gynsurg-reliability-scaled-a10-v1',tags={'project':'isbi-gynsurg'})
volume=modal.Volume.from_name('isbi-gynsurg-results-v1',create_if_missing=False)
sam=ROOT/'materials/code/extracted/sam2/sam2-main'
image=(modal.Image.debian_slim(python_version='3.11')
 .pip_install('torch==2.7.1','torchvision==0.22.1','numpy==1.26.4','scipy==1.15.2','pandas==2.2.3','matplotlib==3.10.1','Pillow==11.2.1','transformers==4.53.3','hydra-core==1.3.2','omegaconf==2.3.0','iopath==0.1.10','opencv-python-headless==4.11.0.86','tqdm==4.67.1')
 .env({'SAM2_BUILD_CUDA':'0','PYTHONPATH':'/opt/isbi','PYTHONHASHSEED':'20261017','HF_HUB_OFFLINE':'1'})
 .add_local_dir(str(sam),'/opt/sam2',copy=True,ignore=['.git','__pycache__'])
 .run_commands('pip install --no-deps --no-build-isolation /opt/sam2')
 .add_local_dir(str(ROOT/'src'),'/opt/isbi/src',ignore=['__pycache__'])
 .add_local_dir(str(ROOT/'configs'),'/opt/isbi/configs')
 .add_local_dir(str(ROOT/'protocols/20261017_reliability_scaled_a10_v1'),'/opt/isbi/protocols/20261017_reliability_scaled_a10_v1')
 .add_local_file(str(ROOT/'materials/checkpoints/sam2.1_hiera_tiny.pt'),'/opt/isbi/materials/checkpoints/sam2.1_hiera_tiny.pt')
 .add_local_file(str(ROOT/'materials/verification/official_sparse_scope_v1/scope_evidence.json'),'/opt/isbi/materials/verification/official_sparse_scope_v1/scope_evidence.json'))
SOURCE='official_sparse_feasibility_20261016T144620Z'
IDS=['001','038','059','086','105','116','138','146','160']

@app.function(image=image,gpu='A10G',cpu=4,memory=24576,timeout=2400,max_containers=2,volumes={'/results':volume})
def execute(run_id,cid,stage):
    import os,sys,subprocess,time,hashlib,torch
    if cid not in IDS or (stage=='engineering' and cid not in ['059','086']):raise ValueError('Development allowlist failed')
    root=Path('/opt/isbi');freeze=json.loads((root/'protocols/20261017_reliability_scaled_a10_v1/FREEZE.json').read_text())
    for relative,digest in freeze['hashes'].items():
        # CLI scripts are recorded in the local freeze but not executed remotely.
        if relative.startswith('scripts/') or relative.startswith('tests/'):continue
        with (root/relative).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=digest:raise ValueError('Frozen source/config changed: '+relative)
    name=torch.cuda.get_device_name(0)
    if name!=freeze['actual_gpu_required']:raise RuntimeError('Frozen GPU contract requires '+freeze['actual_gpu_required']+', got '+name+'; no automatic hardware substitution')
    source=Path('/results')/SOURCE/cid;old=json.loads((source/'execution.json').read_text())
    if old['official_split']!='train':raise ValueError('Not official training split')
    seq=source/'prepared/sequence.npz'
    with seq.open('rb') as f:input_sha=hashlib.file_digest(f,'sha256').hexdigest()
    prior=json.loads((source/'suite/suite_provenance.json').read_text())
    if input_sha!=prior['input_sha256']:raise ValueError('Saved sequence changed')
    os.chdir(root);out=Path('/results')/run_id/cid;out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    cmd=[sys.executable,'-m','src.run_suite','--input',str(seq),'--registry','configs/reliability_scaled_a10_v1/experiment_registry.json','--output',str(out/'suite'),'--split','val','--cache-gib','4','--debug-trace']
    with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:r=subprocess.run(cmd,stdout=stdout,stderr=stderr)
    volume.commit()
    if r.returncode:raise RuntimeError((out/'stderr.txt').read_text()[-4000:])
    record=dict(clip_id=cid,procedure_id=old['procedure_id'],official_split='train',stage=stage,source_run=SOURCE,input_sha256=input_sha,actual_gpu=name,final_test=False,total_seconds=time.perf_counter()-start)
    (out/'execution.json').write_text(json.dumps(record,indent=2));volume.commit()
    files=[p.relative_to(Path('/results')).as_posix() for p in out.rglob('*') if p.is_file() and p.suffix in ['.json','.csv','.png','.txt']]
    return record,files

@app.local_entrypoint()
def main(stage:str='engineering'):
    if stage not in ['engineering','development']:raise ValueError('No validation/test dispatcher in this prototype')
    # Verify the complete local freeze before any GPU dispatch.
    import hashlib
    freeze=json.loads((ROOT/'protocols/20261017_reliability_scaled_a10_v1/FREEZE.json').read_text())
    for relative,digest in freeze['hashes'].items():
        with (ROOT/relative).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=digest:raise ValueError('Freeze invalidated: '+relative)
    run_id='reliability_scaled_a10_'+stage+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'results'/run_id;out.mkdir();reports=[]
    for record,files in execute.starmap([(run_id,cid,stage) for cid in (['059','086'] if stage=='engineering' else IDS)],order_outputs=False):
        reports.append(record);print('V2_COMPLETE',record['clip_id'],flush=True)
        for remote in files:
            dest=out/Path(remote).relative_to(run_id);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b''.join(volume.read_file(remote)))
    (out/'cohort_execution.json').write_text(json.dumps(dict(run_id=run_id,stage=stage,source_run=SOURCE,reports=reports,final_test=False),indent=2),encoding='utf8')
    print(out,flush=True)
