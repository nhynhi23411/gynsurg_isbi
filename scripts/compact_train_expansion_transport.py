"""Read-only CPU bundling and verified local transport; no model invocation."""
from pathlib import Path,PurePosixPath
import concurrent.futures,datetime,hashlib,json,os,shutil,subprocess,sys,zipfile
import modal
ROOT=Path(__file__).resolve().parents[1] if modal.is_local() else Path('/tmp')
BASE=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1')
RUN='train_expansion_20261017_v1'
P=ROOT/'protocols/20261019_train_expansion_recovery_v1'
app=modal.App('isbi-gynsurg-metrics-transport-v1',tags={'project':'isbi-gynsurg'})
volume=modal.Volume.from_name('isbi-gynsurg-results-v1',create_if_missing=False)
image=modal.Image.debian_slim(python_version='3.11') if modal.is_local() else None

def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def valid_member(name,ids):
    path=PurePosixPath(name)
    return ('\\' not in name and not path.is_absolute() and '..' not in path.parts
            and len(path.parts)>=2 and path.parts[0] in ids
            and path.suffix in ['.csv','.json','.txt'])

@app.function(image=image,cpu=2,memory=4096,timeout=600,serialized=True,volumes={'/results':volume})
def bundle(expected):
    from pathlib import Path
    import csv,hashlib,json,zipfile
    root=Path('/results/train_expansion_20261017_v1')
    if len(expected)!=170 or len({r['clip_id'] for r in expected})!=170:raise ValueError('Expected170 unique receipts')
    out=root/'transport_compact_20261019';out.mkdir(exist_ok=True)
    archive=out/'metrics.zip';hashes={}
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for receipt in expected:
            cid=receipt['clip_id'];folder=root/cid
            if json.loads((folder/'execution.json').read_text())!=receipt:raise ValueError('Remote receipt mismatch '+cid)
            for filename,n in [('per_frame.csv',24),('per_clip.csv',8),('queries.csv',5)]:
                with (folder/'suite/uncertainty_raw'/filename).open() as f:
                    if len(list(csv.DictReader(f)))!=n:raise ValueError('Remote metric grid incomplete '+cid)
            for path in sorted(folder.rglob('*')):
                if not path.is_file() or path.suffix not in ['.csv','.json','.txt']:continue
                name=path.relative_to(root).as_posix();data=path.read_bytes()
                hashes[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
    with archive.open('rb') as handle:archive_hash=hashlib.file_digest(handle,'sha256').hexdigest()
    manifest=dict(path='/train_expansion_20261017_v1/transport_compact_20261019/metrics.zip',archive_sha256=archive_hash,archive_bytes=archive.stat().st_size,clips=170,files=len(hashes),sha256=hashes,model_calls=0,final_test=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));volume.commit();return manifest

def extract_verified(archive,manifest,stage,ids):
    if sha(archive)!=manifest['archive_sha256']:raise ValueError('Archive hash mismatch')
    stage=Path(stage);stage.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(manifest['sha256']):raise ValueError('Archive membership mismatch')
        for name in names:
            if not valid_member(name,ids):raise ValueError('Unsafe/nonmetric archive member: '+name)
            data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=manifest['sha256'][name]:raise ValueError('Member hash mismatch')
            target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)

def main():
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Use py -3.11')
    contract=json.loads((P/'COMPACT_TRANSPORT_FREEZE.json').read_text(encoding='utf-8-sig'))
    for name,digest in contract['sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Transport source changed')
    progress=json.loads((BASE/'cohort_progress.json').read_text())
    if progress['completed']!=170 or not progress['complete'] or progress['stage']!='download':raise ValueError('Completed GPU batch in download required')
    records=progress['reports'];ids={r['clip_id'] for r in records}
    expected=json.loads((ROOT/'protocols/20261017_train_expansion_v1/inputs.json').read_text())
    if ids!={r['clip_id'] for r in expected}:raise ValueError('Wrong cohort')
    if any(r['actual_gpu']!='NVIDIA A10' or r['official_split']!='train' or r['final_test'] for r in records):raise ValueError('Wrong scope/hardware')
    out=BASE/'transport_compact_20261019';out.mkdir(exist_ok=True)
    (out/'pre_handoff_progress.json').write_text(json.dumps(progress,indent=2))
    with modal.enable_output(),app.run():manifest=bundle.remote(records)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=out/'metrics.zip'
    with archive.open('wb') as handle:
        for chunk in volume.read_file(manifest['path']):handle.write(chunk)
    if archive.stat().st_size!=manifest['archive_bytes']:raise ValueError('Incomplete download')
    stage=out/'verified_staging';extract_verified(archive,manifest,stage,ids)
    for record in records:
        if json.loads((stage/record['clip_id']/'execution.json').read_text())!=record:raise ValueError('Staged receipt mismatch')
    print('ALL170_COMPACT_TRANSPORT_VERIFIED',manifest['files'],manifest['archive_bytes'],flush=True)
    (out/'verified.json').write_text(json.dumps(dict(passed=True,files=manifest['files'],clips=170,archive_sha256=manifest['archive_sha256'],model_calls=0,source_freeze_sha256=sha(P/'COMPACT_TRANSPORT_FREEZE.json')),indent=2))

if __name__=='__main__':main()
