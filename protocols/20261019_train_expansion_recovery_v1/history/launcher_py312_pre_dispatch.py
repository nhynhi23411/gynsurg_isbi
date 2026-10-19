"""Standalone recovery launcher. Scientific algorithm/config unchanged."""
from pathlib import Path
import argparse,concurrent.futures,datetime,hashlib,json,os,subprocess,sys
import modal
ROOT=Path(__file__).resolve().parents[1] if modal.is_local() else Path('/opt/isbi')
P=ROOT/'protocols/20261017_train_expansion_v1'
RECOVERY=ROOT/'protocols/20261019_train_expansion_recovery_v1'
LOCAL=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1')
RUN='train_expansion_20261017_v1'
REUSE_RUN='reliability_scaled_a10_development_20261017T031408Z'
INPUT_RUN='official_sparse_feasibility_20261016T144620Z'
app=modal.App('isbi-gynsurg-train-expansion-recovery-v1',tags={'project':'isbi-gynsurg'})
volume=modal.Volume.from_name('isbi-gynsurg-results-v1',create_if_missing=False)
if modal.is_local():
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
    image=(image.add_local_dir(str(P),'/opt/isbi/protocols/20261017_train_expansion_v1')
        .add_local_dir(str(RECOVERY),'/opt/isbi/protocols/20261019_train_expansion_recovery_v1')
        .add_local_file(str(Path(__file__)),'/opt/isbi/recovery_launcher_snapshot.py')
        .add_local_file(str(ROOT/'data/manifests/procedure_mapping_verified_v5.csv'),'/opt/isbi/data/manifests/procedure_mapping_verified_v5.csv')
        .add_local_file(str(ROOT/'protocols/20261016_official_sparse_v1/PROTOCOL.md'),'/opt/isbi/protocols/20261016_official_sparse_v1/PROTOCOL.md')
        .add_local_dir(str(LOCAL/'transport_parts'),'/opt/train_transport'))
else:
    image=None

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

@app.function(image=image,cpu=2,memory=4096,timeout=600,serialized=True,volumes={'/results':volume})
def runtime_preflight():
    from pathlib import Path
    import hashlib,json,importlib,platform
    root=Path('/opt/isbi')
    def sha(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    recovery=json.loads((root/'protocols/20261019_train_expansion_recovery_v1/FREEZE.json').read_text())
    assert sha(root/'recovery_launcher_snapshot.py')==recovery['launcher_sha256']
    assert sha(root/'protocols/20261017_train_expansion_v1/FREEZE.json')==recovery['scientific_freeze_sha256']
    modules=['src.prepare_sequence','src.run_suite','src.surgery','src.sam_engine','sam2']
    for name in modules:importlib.import_module(name)
    import torch
    assert torch.__version__.split('+')[0]=='2.7.1'
    return dict(passed=True,modules_imported=modules,standalone_serialized=True,torch=torch.__version__,python=platform.python_version(),final_test=False)

@app.function(image=image,gpu='A10G',cpu=4,memory=24576,timeout=2400,max_containers=4,max_inputs=1,serialized=True,volumes={'/results':volume})
def execute(cid):
    import subprocess,sys,time,torch,csv,shutil,zipfile
    def digest(p):
        with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    root=Path('/opt/isbi');os.chdir(root)
    recovery=json.loads((root/'protocols/20261019_train_expansion_recovery_v1/FREEZE.json').read_text())
    if digest(root/'recovery_launcher_snapshot.py')!=recovery['launcher_sha256']:raise ValueError('Recovery source mismatch')
    if digest(root/'protocols/20261017_train_expansion_v1/FREEZE.json')!=recovery['scientific_freeze_sha256']:raise ValueError('Scientific freeze mismatch')
    freeze=json.loads((root/'protocols/20261017_train_expansion_v1/FREEZE.json').read_text())
    rows=json.loads((root/'protocols/20261017_train_expansion_v1/inputs.json').read_text())
    row=next((r for r in rows if r['clip_id']==cid),None)
    if row is None or row['official_split']!='train':raise ValueError('Training allowlist rejected')
    actual=torch.cuda.get_device_name(0)
    if actual!='NVIDIA A10':raise RuntimeError('Expansion GPU contract requires NVIDIA A10; got '+actual)
    if not row['reuse']:
        names=[name for name in freeze['sha256'] if name.startswith(('data/raw/','data/derived/','data/manifests/sequence_')) and Path(name).name.startswith(cid)]
        if len(names)!=15:raise ValueError('Wrong per-clip transport membership')
        transport_index=json.loads(Path('/opt/train_transport/index.json').read_text())
        with zipfile.ZipFile(Path('/opt/train_transport')/transport_index[cid]) as archive:
            for name in names:archive.extract(name,root)
    # Verify code, config, checkpoint and this clip before any model is loaded.
    for name,expected in freeze['sha256'].items():
        if name.startswith(('scripts/','tests/')):continue
        is_input=name.startswith(('data/raw/','data/derived/','data/manifests/sequence_'))
        if is_input and (row['reuse'] or not Path(name).name.startswith(cid)):continue
        if digest(root/name)!=expected:raise ValueError('Frozen artifact mismatch: '+name)
    cfg=root/'configs/reliability_scaled_a10_v1/uncertainty_raw.json'
    out=Path('/results')/RUN/cid
    if (out/'execution.json').exists():
        record=json.loads((out/'execution.json').read_text())
        if record['freeze_sha256']!=digest(root/'protocols/20261017_train_expansion_v1/FREEZE.json'):raise ValueError('Resume freeze mismatch')
        return record
    start=time.perf_counter()
    if out.exists():
        history=out.parent/'_incomplete_attempts';history.mkdir(exist_ok=True)
        old=history/(cid+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        out.rename(old)
    out.mkdir(parents=True,exist_ok=False)
    if row['reuse']:
        source=Path('/results')/REUSE_RUN/cid;old=json.loads((source/'execution.json').read_text())
        suite=json.loads((source/'suite/suite_provenance.json').read_text())
        if old['official_split']!='train' or old['actual_gpu']!='NVIDIA A10' or old['procedure_id']!=row['procedure_id']:raise ValueError('Reuse scope mismatch')
        if suite['input_sha256']!=old['input_sha256'] or suite['sam_checkpoint_sha256']!=digest(root/'materials/checkpoints/sam2.1_hiera_tiny.pt'):raise ValueError('Reuse input/checkpoint mismatch')
        for name,h in suite['source_hashes'].items():
            if digest(root/name)!=h:raise ValueError('Reuse source changed: '+name)
        recipe=source/'suite/uncertainty_raw';prior=json.loads((recipe/'provenance.json').read_text())
        if prior['config']!=json.loads(cfg.read_text()):raise ValueError('Reuse config differs')
        seq=Path('/results')/INPUT_RUN/cid/'prepared/sequence.npz'
        if digest(seq)!=old['input_sha256']:raise ValueError('Reused input changed')
        prepared=json.loads((seq.parent/'provenance.json').read_text())
        if prepared['video_sha256']!=row['video_sha256']:raise ValueError('Reused video differs')
        target=out/'suite/uncertainty_raw';target.mkdir(parents=True)
        for name in ['per_clip.csv','per_frame.csv','queries.csv','provenance.json']:shutil.copy2(recipe/name,target/name)
        shutil.copy2(source/'suite/suite_provenance.json',out/'suite/suite_provenance.json')
        shutil.copy2(seq.parent/'provenance.json',out/'input_provenance.json')
        record=dict(row,reused=True,source_run=REUSE_RUN,input_sha256=old['input_sha256'],actual_gpu=old['actual_gpu'],source_full_suite_seconds=old['total_seconds'])
    else:
        meta_path=root/f'data/manifests/sequence_official_sparse_v1/{cid}.metadata.json'
        meta=json.loads(meta_path.read_text())
        if meta['official_split']!='train' or meta['procedure_id']!=row['procedure_id']:raise ValueError('Metadata split mismatch')
        for field,proof in [('grouping_evidence_sha256','grouping_evidence'),('protocol_sha256','protocol'),('reference_scope_evidence_sha256','reference_scope_evidence')]:
            if digest(root/meta[proof])!=meta[field]:raise ValueError('Evidence mismatch')
        refs=list(csv.DictReader((root/f'data/manifests/sequence_official_sparse_v1/{cid}.references.csv').open()))
        for r in refs:
            r['mask']='data/derived/task3_official_v1/'+r['mask'].replace('\\','/').split('/')[-1]
            if digest(root/r['mask'])!=r['mask_sha256']:raise ValueError('Reference hash mismatch')
        refout=out/'references.remote.csv'
        with refout.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(refs[0]));w.writeheader();w.writerows(refs)
        export=Path('/results/technical_smoke_20261016T091048Z/exports/uterus_primary_tools.ts')
        proof=json.loads(export.with_suffix('.ts.provenance.json').read_text())
        if digest(export)!=proof['export_sha256'] or proof['export_sha256']!='cc90d72faaa5687c5ef925fbd00e6ae4aeed4e4b657a19d85eaaefc88ffd2e3b' or proof['checkpoint_sha256']!='c8c763312cba9cff5bb02d492e548cce0cd0975d13c54c1a40f93e8f578bf192':raise ValueError('Export changed')
        registry=out/'registry.json';registry.write_text(json.dumps([dict(run_id='uncertainty_raw',config='configs/reliability_scaled_a10_v1/uncertainty_raw.json')]))
        cmds=[('prepare',[sys.executable,'-m','src.prepare_sequence','--video',str(root/f'data/raw/local/AutoLaparo/autolaparo_task2/clips/clips/{cid}.mp4'),'--segmenter',str(export),'--metadata',str(meta_path),'--references',str(refout),'--out',str(out/'prepared')]),('suite',[sys.executable,'-m','src.run_suite','--input',str(out/'prepared/sequence.npz'),'--registry',str(registry),'--output',str(out/'suite'),'--split','val','--cache-gib','4'])]
        for stage,cmd in cmds:
            with (out/f'{stage}_stdout.txt').open('w') as stdout,(out/f'{stage}_stderr.txt').open('w') as stderr:result=subprocess.run(cmd,stdout=stdout,stderr=stderr)
            if result.returncode:
                volume.commit();raise RuntimeError(stage+' failed: '+(out/f'{stage}_stderr.txt').read_text()[-4000:])
        record=dict(row,reused=False,actual_gpu=actual,input_sha256=digest(out/'prepared/sequence.npz'),export_sha256=proof['export_sha256'])
    record.update(recovery_freeze_sha256=digest(root/'protocols/20261019_train_expansion_recovery_v1/FREEZE.json'),run_id=RUN,freeze_sha256=digest(root/'protocols/20261017_train_expansion_v1/FREEZE.json'),final_test=False,total_seconds=time.perf_counter()-start)
    (out/'execution.json').write_text(json.dumps(record,indent=2));volume.commit();return record

def put(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(value,indent=2),encoding='utf8');temp.replace(path)

def fetch_case(cid):
    entries=[e for e in volume.iterdir('/'+RUN+'/'+cid,recursive=True) if Path(e.path).suffix in ['.csv','.json','.txt']]
    def fetch(e):
        dest=LOCAL/Path(e.path.lstrip('/')).relative_to(RUN);dest.parent.mkdir(parents=True,exist_ok=True)
        data=b''.join(volume.read_file(e.path))
        if len(data)!=e.size:raise ValueError('Download incomplete: '+e.path)
        temp=dest.with_suffix(dest.suffix+'.part');temp.write_bytes(data);temp.replace(dest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(fetch,entries))

def validate_subset(ids, label):
    rows=json.loads((P/'inputs.json').read_text());selected=[r for r in rows if r['clip_id'] in ids]
    folder=LOCAL/'recovery_20261019'/label;folder.mkdir(parents=True,exist_ok=True)
    import csv
    cohort=folder/'cohort.csv'
    with cohort.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['clip_id','procedure_id']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in selected)
    registry=folder/'registry.json'
    put(registry,[dict(run_id='uncertainty_raw',config='configs/reliability_scaled_a10_v1/uncertainty_raw.json')])
    receipt=folder/'cohort_acceptance.json'
    if receipt.exists():receipt.rename(folder/('previous_acceptance_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json'))
    subprocess.run([sys.executable,str(ROOT/'scripts/validate_cohort_results.py'),'--results-root',str(LOCAL),'--cohort',str(cohort),'--registry',str(registry),'--output',str(receipt)],check=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot-only',action='store_true');args=parser.parse_args()
    recovery=json.loads((RECOVERY/'FREEZE.json').read_text())
    if digest(Path(__file__))!=recovery['launcher_sha256'] or digest(P/'FREEZE.json')!=recovery['scientific_freeze_sha256']:raise ValueError('Recovery freeze mismatch')
    science=json.loads((P/'FREEZE.json').read_text())
    for name,h in science['sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Scientific input/source changed: '+name)
    rows=json.loads((P/'inputs.json').read_text());expected={r['clip_id']:r for r in rows}
    if len(expected)!=170 or sum(not r['reuse'] for r in rows)!=161:raise ValueError('Wrong expansion scope')
    logdir=LOCAL/'recovery_20261019';logdir.mkdir(parents=True,exist_ok=True)
    recovery_hash=digest(RECOVERY/'FREEZE.json');records={};attempts=[];calls=[]
    if (logdir/'attempts.json').exists():attempts=json.loads((logdir/'attempts.json').read_text())
    try:entries=list(volume.iterdir('/'+RUN,recursive=True))
    except (FileNotFoundError,modal.exception.NotFoundError):entries=[]
    for e in entries:
        if not e.path.endswith('/execution.json') or '/_incomplete_attempts/' in e.path:continue
        r=json.loads(b''.join(volume.read_file(e.path)));cid=r['clip_id']
        if cid not in expected or r['freeze_sha256']!=recovery['scientific_freeze_sha256'] or r.get('recovery_freeze_sha256')!=recovery_hash or r['actual_gpu']!='NVIDIA A10':raise ValueError('Unexpected resume receipt')
        records[cid]=r
    def progress(stage):
        put(LOCAL/'cohort_progress.json',dict(run_id=RUN,expected=170,completed=len(records),complete=len(records)==170,stage=stage,reports=list(records.values()),failures=[x for x in attempts if not x['accepted']],final_test=False,recovery_freeze_sha256=recovery_hash,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        put(logdir/'attempts.json',attempts);put(logdir/'calls.json',calls)
    def accept(cid, result, attempt):
        if result['clip_id']!=cid or result['actual_gpu']!='NVIDIA A10' or result['freeze_sha256']!=recovery['scientific_freeze_sha256'] or result.get('recovery_freeze_sha256')!=recovery_hash:raise ValueError('Wrong returned evidence')
        records[cid]=result;attempts.append(dict(clip_id=cid,attempt=attempt,accepted=True,actual_gpu=result['actual_gpu']));progress('compute')
        print('ACCEPTED',cid,'completed',len(records),'/170','reused',result['reused'],flush=True)
    progress('recovery_preflight')
    # CPU packaging and one real new clip must pass before any batch dispatch.
    with app.run():
        print('PREFLIGHT_APP',app.app_id,flush=True)
        preflight=runtime_preflight.remote();put(logdir/'runtime_preflight.json',preflight)
        print('CPU_IMPORT_PREFLIGHT_PASS',flush=True)
        if '002' not in records:
            for attempt in range(1,13):
                call=execute.spawn('002');calls.append(dict(stage='pilot',clip_id='002',call_id=call.object_id,attempt=attempt));progress('pilot')
                try:result=call.get()
                except Exception as error:
                    attempts.append(dict(clip_id='002',attempt=attempt,accepted=False,error_type=type(error).__name__,error=str(error)));progress('pilot_failed')
                    if 'Expansion GPU contract' not in str(error):raise
                    continue
                accept('002',result,attempt);break
            else:raise RuntimeError('Pilot allocation contract not obtained after12 attempts')
        fetch_case('002');validate_subset({'002'},'pilot_002')
        put(logdir/'pilot_passed.json',dict(passed=True,clip_id='002',quality_threshold_used=False,actual_gpu='NVIDIA A10',final_test=False))
        print('PILOT_002_END_TO_END_ACCEPTED',flush=True)
    if args.pilot_only:return
    for allocation_round in range(1,13):
        pending=[r['clip_id'] for r in rows if r['clip_id'] not in records]
        if not pending:break
        print('BATCH_ROUND',allocation_round,'pending',len(pending),flush=True)
        app_id=None
        try:
            with app.run(detach=True):
                app_id=app.app_id;put(logdir/'active_app.json',dict(app_id=app_id,allocation_round=allocation_round))
                inflight=[]
                for cid in pending:
                    call=execute.spawn(cid);inflight.append((cid,call));calls.append(dict(stage='batch',clip_id=cid,call_id=call.object_id,attempt=allocation_round));progress('dispatch')
                for cid,call in inflight:
                    try:result=call.get()
                    except Exception as error:
                        attempts.append(dict(clip_id=cid,attempt=allocation_round,accepted=False,error_type=type(error).__name__,error=str(error)));progress('allocation_or_execution_failure')
                        if 'Expansion GPU contract' not in str(error):raise
                        print('ALLOCATION_MISMATCH',cid,flush=True);continue
                    accept(cid,result,allocation_round)
        finally:
            if app_id:
                subprocess.run([sys.executable,'-m','modal','app','stop',app_id,'--yes'],check=True)
    if len(records)!=170:raise RuntimeError('Bounded allocation retries exhausted; receipts preserved')
    progress('download')
    for cid in expected:fetch_case(cid)
    put(LOCAL/'cohort_execution.json',dict(run_id=RUN,stage='all_official_training',reports=[records[r['clip_id']] for r in rows],final_test=False,recovery_freeze_sha256=recovery_hash))
    progress('acceptance_and_analysis')
    subprocess.run([sys.executable,str(ROOT/'scripts/analyze_train_expansion.py')],cwd=ROOT,check=True)
    put(logdir/'pipeline_complete.json',dict(complete=True,clips=170,source_groups=9,final_test=False,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    progress('complete')
    print('ALL_170_ANALYZED_AND_AUDITED',flush=True)

if __name__=='__main__':
    with modal.enable_output():main()

