"""Frozen train-only raw baseline expansion; private remote arrays, C: reports."""
from pathlib import Path
import concurrent.futures,datetime,hashlib,json,os
import modal
from modal_reliability_scaled_a10 import image as base_image,volume,ROOT
P=ROOT/'protocols/20261017_train_expansion_v1'
LOCAL=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1')
RUN='train_expansion_20261017_v1'
REUSE_RUN='reliability_scaled_a10_development_20261017T031408Z'
INPUT_RUN='official_sparse_feasibility_20261016T144620Z'
app=modal.App('isbi-gynsurg-train-expansion-v1',tags={'project':'isbi-gynsurg'})
inputs=json.loads((P/'inputs.json').read_text()) if (P/'inputs.json').exists() else []
TRANSPORT=LOCAL/'transport_parts'
image=(base_image
       .add_local_dir(str(P),'/opt/isbi/protocols/20261017_train_expansion_v1')
       .add_local_file(str(ROOT/'data/manifests/procedure_mapping_verified_v5.csv'),'/opt/isbi/data/manifests/procedure_mapping_verified_v5.csv')
       .add_local_file(str(ROOT/'protocols/20261016_official_sparse_v1/PROTOCOL.md'),'/opt/isbi/protocols/20261016_official_sparse_v1/PROTOCOL.md')
       .add_local_dir(str(TRANSPORT),'/opt/train_transport'))

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@app.function(image=image,gpu='A10G',cpu=4,memory=24576,timeout=2400,max_containers=4,volumes={'/results':volume})
def execute(cid):
    import subprocess,sys,time,torch,csv,shutil,zipfile
    root=Path('/opt/isbi');os.chdir(root)
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
    start=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
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
        record=dict(row,reused=True,source_run=REUSE_RUN,input_sha256=old['input_sha256'],actual_gpu=old['actual_gpu'],inference_seconds=old['total_seconds'])
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
    record.update(run_id=RUN,freeze_sha256=digest(root/'protocols/20261017_train_expansion_v1/FREEZE.json'),final_test=False,total_seconds=time.perf_counter()-start)
    (out/'execution.json').write_text(json.dumps(record,indent=2));volume.commit();return record

def main():
    freeze=json.loads((P/'FREEZE.json').read_text())
    for name,expected in freeze['sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Local freeze changed: '+name)
    LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/'FREEZE.json').write_bytes((P/'FREEZE.json').read_bytes())
    (LOCAL/'controller_snapshot.py').write_bytes(Path(__file__).read_bytes())
    reports={};failures=[]
    # One directory listing, rather than170 serial missing-file RPCs.
    try:entries=list(volume.iterdir('/'+RUN,recursive=True))
    except (FileNotFoundError,modal.exception.NotFoundError):entries=[]
    for e in entries:
        if Path(e.path).name!='execution.json':continue
        record=json.loads(b''.join(volume.read_file(e.path)))
        if record['freeze_sha256']!=digest(P/'FREEZE.json'):raise ValueError('Resume freeze changed')
        reports[record['clip_id']]=record
    def progress():
        (LOCAL/'cohort_progress.json').write_text(json.dumps(dict(run_id=RUN,expected=170,completed=len(reports),complete=len(reports)==170,reports=list(reports.values()),failures=failures,final_test=False),indent=2),encoding='utf8')
    progress()
    for attempt in range(1,5):
        pending=[r['clip_id'] for r in inputs if r['clip_id'] not in reports]
        if not pending:break
        print('ROUND',attempt,'pending',len(pending),flush=True)
        with app.run():
            for i,result in enumerate(execute.map(pending,return_exceptions=True,order_outputs=True)):
                cid=pending[i]
                if isinstance(result,BaseException):
                    failures.append(dict(round=attempt,clip_id=cid,error=str(result)));progress()
                    if 'Expansion GPU contract' not in str(result):raise result
                    print('GPU_MISMATCH',cid,flush=True)
                else:
                    reports[cid]=result;progress();print('COMPLETE',cid,'reused',result['reused'],'seconds',round(result['total_seconds'],1),flush=True)
    if len(reports)!=170:raise RuntimeError('Bounded allocation retries exhausted; partial receipts preserved')
    entries=[e for e in volume.iterdir('/'+RUN,recursive=True) if Path(e.path).suffix in ['.csv','.json','.txt']]
    def fetch(e):
        dest=LOCAL/Path(e.path.lstrip('/')).relative_to(RUN);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.stat().st_size==e.size:return
        data=b''.join(volume.read_file(e.path))
        if len(data)!=e.size:raise ValueError('Incomplete artifact download')
        temp=dest.with_suffix(dest.suffix+'.part');temp.write_bytes(data);temp.replace(dest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(fetch,entries))
    (LOCAL/'cohort_execution.json').write_text(json.dumps(dict(run_id=RUN,stage='all_official_training',reports=[reports[r['clip_id']] for r in inputs],final_test=False),indent=2),encoding='utf8')
    print('ALL_170_COMPLETE',LOCAL,flush=True)
if __name__=='__main__':main()
