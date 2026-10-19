"""Development grid on the saved nine reviewed sequences; no test dispatcher."""
from pathlib import Path
import modal,json,datetime
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('isbi-gynsurg-official-sparse-tuning-v1',tags={'project':'isbi-gynsurg'})
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
IDS=['001','038','059','086','105','116','138','146','160']
@app.function(image=image,gpu='A10G',cpu=4,memory=24576,timeout=3600,max_containers=4,volumes={'/results':volume})
def tune_clip(run_id,cid,source_run):
    import os,time,subprocess,sys,hashlib,shutil
    os.chdir('/opt/isbi');assert cid in IDS and int(cid)<=170
    original=Path('/results')/source_run/cid;source=original/'prepared/sequence.npz'
    source_report=json.loads((original/'execution.json').read_text());assert source_report['official_split']=='train'
    start=time.perf_counter();out=Path('/results')/run_id/cid;out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(original/'prepared/provenance.json',out/'input_provenance.json')
    command=[sys.executable,'-m','src.run_suite','--input',str(source),'--registry','configs/official_sparse_tuning_v1/experiment_registry.json','--output',str(out/'suite'),'--split','val','--cache-gib','4']
    with (out/'suite_stdout.txt').open('w') as stdout,(out/'suite_stderr.txt').open('w') as stderr:r=subprocess.run(command,stdout=stdout,stderr=stderr)
    if r.returncode:volume.commit();raise RuntimeError((out/'suite_stderr.txt').read_text()[-4000:])
    report=dict(run_id=run_id,clip_id=cid,procedure_id=source_report['procedure_id'],official_split='train',source_run=source_run,
      input_sha256=hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),total_seconds=time.perf_counter()-start,
      scope='12-recipe prespecified development alpha/beta grid; no final test',final_test=False)
    (out/'execution.json').write_text(json.dumps(report,indent=2));volume.commit();return report
@app.local_entrypoint()
def main(source_run:str='official_sparse_feasibility_20261016T144620Z'):
    run_id='official_sparse_tuning_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=ROOT/'results'/run_id;out.mkdir()
    reports=[]
    for report in tune_clip.starmap([(run_id,cid,source_run) for cid in IDS],order_outputs=False):
        reports.append(report);print('TUNING_COMPLETE',report['clip_id'],round(report['total_seconds'],2),flush=True)
    (out/'cohort_execution.json').write_text(json.dumps(dict(run_id=run_id,stage='tuning',reports=reports,final_test=False),indent=2))
    print('Tuning compute complete; fetch small artifacts with fetch_sparse_results.py --stage tuning --run',run_id,flush=True)
