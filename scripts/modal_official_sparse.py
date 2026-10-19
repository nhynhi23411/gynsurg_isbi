"""Official-label development only; no final-test dispatch in this launcher."""
from pathlib import Path
import modal,json,datetime
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('isbi-gynsurg-official-sparse-development-v1',tags={'project':'isbi-gynsurg'})
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
 .add_local_file(str(ROOT/'materials/verification/official_sparse_scope_v1/scope_evidence.json'),'/opt/isbi/materials/verification/official_sparse_scope_v1/scope_evidence.json')
 .add_local_file(str(ROOT/'data/manifests/procedure_mapping_verified_v5.csv'),'/opt/isbi/data/manifests/procedure_mapping_verified_v5.csv')
 .add_local_file(str(ROOT/'protocols/20261016_official_sparse_v1/PROTOCOL.md'),'/opt/isbi/protocols/20261016_official_sparse_v1/PROTOCOL.md'))
cohort_path=ROOT/'data/manifests/sequence_official_sparse_v1/development_feasibility_cohort.csv'
import csv
feasibility=['001','038','059','086','105','116','138','146','160']
allowed=sorted(set(feasibility+['058']))
VALIDATION=['171','190','211']
allowed=sorted(set(allowed+VALIDATION))
for cid in allowed:
    image=image.add_local_file(str(ROOT/f'data/raw/local/AutoLaparo/autolaparo_task2/clips/clips/{cid}.mp4'),f'/opt/isbi/inputs/{cid}.mp4')
    for ending in ['metadata.json','references.csv']:
        image=image.add_local_file(str(ROOT/f'data/manifests/sequence_official_sparse_v1/{cid}.{ending}'),f'/opt/isbi/inputs/{cid}.{ending}')
    for t in [1,25,50,75,100,125]:
        for target in ['uterus','primary_tools_union']:
            name=f'{cid}{t:03}_{target}.png';image=image.add_local_file(str(ROOT/'data/derived/task3_official_v1'/name),'/opt/isbi/data/derived/task3_official_v1/'+name)

@app.function(image=image,gpu='A10G',cpu=4,memory=24576,timeout=7200,max_containers=4,volumes={'/results':volume})
def run_clip(run_id,cid,primary_only=False,validation=False):
    import os,subprocess,sys,time,hashlib,csv,platform
    os.chdir('/opt/isbi');start=time.perf_counter();out=Path('/results')/run_id/cid;out.mkdir(parents=True,exist_ok=False)
    meta=json.loads(Path(f'inputs/{cid}.metadata.json').read_text())
    assert meta['split']=='val'
    if validation:assert cid in VALIDATION and meta['official_split']=='val'
    else:assert meta['official_split']=='train' and int(cid)<=170
    for field,evidence in [('grouping_evidence_sha256',meta['grouping_evidence']),('protocol_sha256',meta['protocol']),('reference_scope_evidence_sha256',meta['reference_scope_evidence'])]:
        assert hashlib.file_digest(open(evidence,'rb'),'sha256').hexdigest()==meta[field],field
    refs=list(csv.DictReader(open(f'inputs/{cid}.references.csv')))
    for row in refs:
        row['mask']='data/derived/task3_official_v1/'+row['mask'].replace('\\','/').split('/')[-1]
        assert hashlib.file_digest(open(row['mask'],'rb'),'sha256').hexdigest()==row['mask_sha256']
    refout=out/'references.remote.csv'
    with refout.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(refs[0]));w.writeheader();w.writerows(refs)
    export=Path('/results/technical_smoke_20261016T091048Z/exports/uterus_primary_tools.ts')
    export_proof=json.loads(export.with_suffix('.ts.provenance.json').read_text())
    assert hashlib.file_digest(export.open('rb'),'sha256').hexdigest()==export_proof['export_sha256']
    registry='configs/official_sparse_v1/experiment_registry.json'
    if validation:registry='configs/official_sparse_selected_v1/validation_registry.json'
    if primary_only:
        selected=[r for r in json.loads(Path(registry).read_text()) if r['run_id'] in ['full_seed20261016','uncertainty_seed20261016']]
        assert len(selected)==2
        registry=str(out/'primary_registry.json');Path(registry).write_text(json.dumps(selected,indent=2))
    commands=[('prepare',[sys.executable,'-m','src.prepare_sequence','--video',f'inputs/{cid}.mp4','--segmenter',str(export),'--metadata',f'inputs/{cid}.metadata.json','--references',str(refout),'--out',str(out/'prepared')]),
              ('suite',[sys.executable,'-m','src.run_suite','--input',str(out/'prepared/sequence.npz'),'--registry',registry,'--output',str(out/'suite'),'--split','val','--cache-gib','4'])]
    stages=[]
    for name,cmd in commands:
        before=time.perf_counter()
        with (out/(name+'_stdout.txt')).open('w') as stdout,(out/(name+'_stderr.txt')).open('w') as stderr:
            result=subprocess.run(cmd,stdout=stdout,stderr=stderr)
        stages.append(dict(stage=name,seconds=time.perf_counter()-before,returncode=result.returncode))
        if result.returncode:
            volume.commit();raise RuntimeError(name+' failed '+(out/(name+'_stderr.txt')).read_text()[-5000:])
    report=dict(run_id=run_id,clip_id=cid,procedure_id=meta['procedure_id'],official_split=meta['official_split'],
      scope='Development computational correction benchmark using official sparse labels; not final test',
      clinical_review_complete=False,new_annotations_created=0,frame_count=125,stages=stages,total_seconds=time.perf_counter()-start,
      export_sha256=export_proof['export_sha256'],input_video_sha256=hashlib.file_digest(open(f'inputs/{cid}.mp4','rb'),'sha256').hexdigest(),
      metadata_sha256=hashlib.file_digest(open(f'inputs/{cid}.metadata.json','rb'),'sha256').hexdigest(),source_hashes={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in Path('src').glob('*.py')},
      python=platform.python_version(),reference_mode=meta['reference_review_mode'],primary_only=primary_only,validation=validation,final_test=False)
    (out/'execution.json').write_text(json.dumps(report,indent=2));volume.commit();return report

@app.local_entrypoint()
def main(stage:str='pilot'):
    if stage not in ['pilot','feasibility','repeatability','validation']:raise ValueError('Development/validation stages only')
    actual=[r['clip_id'] for r in csv.DictReader(cohort_path.open(encoding='utf8'))]
    if actual!=feasibility:raise ValueError('Declared development cohort differs from packaged files')
    ids=VALIDATION if stage=='validation' else (['058'] if stage in ['pilot','repeatability'] else feasibility)
    run_id='official_sparse_'+stage+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'results'/run_id;out.mkdir()
    reports=[]
    for report in run_clip.starmap([(run_id,cid,stage=='repeatability',stage=='validation') for cid in ids],order_outputs=False):
        reports.append(report);(out/(report['clip_id']+'_execution.json')).write_text(json.dumps(report,indent=2))
        print('OFFICIAL_SPARSE_COMPLETE',report['clip_id'],round(report['total_seconds'],2),flush=True)
        # Only small CSV/JSON/logs are downloaded; frames/NPZ stay on private volume.
        entries=list(volume.iterdir('/'+run_id+'/'+report['clip_id'],recursive=True))
        for entry in entries:
            p=Path(entry.path.lstrip('/'))
            if p.suffix not in ['.csv','.json','.txt']:continue
            rel=p.relative_to(run_id);dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(b''.join(volume.read_file(entry.path)))
    (out/'cohort_execution.json').write_text(json.dumps(dict(run_id=run_id,stage=stage,reports=reports,final_test=False),indent=2))
    print('Saved development reports to',out,flush=True)
