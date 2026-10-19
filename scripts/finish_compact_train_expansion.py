"""Finish CPU acceptance after verified compact transport and stopped old downloader."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1')
P=ROOT/'protocols/20261019_train_expansion_recovery_v1'

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def put(path,value):
    temp=path.with_suffix(path.suffix+'.part');temp.write_text(json.dumps(value,indent=2),encoding='utf8');temp.replace(path)

def main():
    contract=json.loads((P/'COMPACT_FINISHER_FREEZE.json').read_text(encoding='utf-8-sig'))
    for name,h in contract['sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Transport finisher source changed')
    out=BASE/'transport_compact_20261019'
    verified=json.loads((out/'verified.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    progress=json.loads((out/'pre_handoff_progress.json').read_text());ids={r['clip_id'] for r in progress['reports']}
    if not verified['passed'] or len(ids)!=170 or verified['archive_sha256']!=manifest['archive_sha256']:raise ValueError('Wrong transport evidence')
    if digest(out/'metrics.zip')!=manifest['archive_sha256']:raise ValueError('Archive changed')
    stage=out/'verified_staging'
    from compact_train_expansion_transport import valid_member
    for name,h in manifest['sha256'].items():
        if not valid_member(name,ids) or digest(stage/name)!=h:raise ValueError('Staged artifact changed')
    handoff=json.loads((out/'local_handoff.json').read_text())
    # PID identity and stop are recorded by the calling PowerShell step, before copy.
    if not handoff['old_downloader_stopped'] or handoff['gpu_batch_complete']!=170:raise ValueError('Stop/handoff evidence missing')
    for name,h in manifest['sha256'].items():
        source=stage/name;target=BASE/name;target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix(target.suffix+'.compact-part');temp.write_bytes(source.read_bytes());temp.replace(target)
        if digest(target)!=h:raise ValueError('Copied artifact changed')
    expected=json.loads((ROOT/'protocols/20261017_train_expansion_v1/inputs.json').read_text())
    records={r['clip_id']:r for r in progress['reports']}
    for cid,r in records.items():
        if json.loads((BASE/cid/'execution.json').read_text())!=r:raise ValueError('Copied receipt mismatch')
    put(BASE/'cohort_execution.json',dict(run_id=progress['run_id'],stage='all_official_training',reports=[records[r['clip_id']] for r in expected],final_test=False,recovery_freeze_sha256=progress['recovery_freeze_sha256']))
    progress.update(stage='acceptance_and_analysis',transport='compact_metrics_20261019',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    put(BASE/'cohort_progress.json',progress)
    env=dict(os.environ,PYTHONUSERBASE='C:/Users/pc/AppData/Roaming/Python',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(['C:/Users/pc/.venvs/isbi_gynsurg/Scripts/python.exe',str(ROOT/'scripts/analyze_train_expansion.py')],cwd=ROOT,check=True,env=env)
    put(BASE/'recovery_20261019/pipeline_complete.json',dict(complete=True,clips=170,source_groups=9,final_test=False,transport='compact_metrics_20261019',transport_source_freeze_sha256=digest(P/'COMPACT_TRANSPORT_FREEZE.json'),finisher_freeze_sha256=digest(P/'COMPACT_FINISHER_FREEZE.json'),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    progress.update(stage='complete',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());put(BASE/'cohort_progress.json',progress)
    print('ALL170_FROZEN_ANALYSIS_COMPLETE_COMPACT_TRANSPORT',flush=True)

if __name__=='__main__':main()
