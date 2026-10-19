"""Freeze a train-only expansion without modifying existing experiments."""
from pathlib import Path
import csv,hashlib,json,datetime
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'protocols/20261017_train_expansion_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if (P/'FREEZE.json').exists():raise FileExistsError('Expansion already frozen')
    cohort=list(csv.DictReader((ROOT/'data/manifests/sequence_official_sparse_v1/cohort.csv').open(encoding='utf8')))
    rows=[r for r in cohort if r['official_split']=='train']
    assert len(rows)==170 and len({r['procedure_id'] for r in rows})==9
    reuse={'001','038','059','086','105','116','138','146','160'}
    hashes={};inputs=[]
    for row in rows:
        cid=row['clip_id'];meta=ROOT/f'data/manifests/sequence_official_sparse_v1/{cid}.metadata.json'
        refs=ROOT/f'data/manifests/sequence_official_sparse_v1/{cid}.references.csv'
        video=ROOT/f'data/raw/local/AutoLaparo/autolaparo_task2/clips/clips/{cid}.mp4'
        m=json.loads(meta.read_text());assert m['official_split']=='train' and m['procedure_id']==row['procedure_id']
        paths=[meta,refs,video]
        for t in [1,25,50,75,100,125]:
            for target in ['uterus','primary_tools_union']:
                paths.append(ROOT/f'data/derived/task3_official_v1/{cid}{t:03}_{target}.png')
        for path in paths:hashes[path.relative_to(ROOT).as_posix()]=sha(path)
        inputs.append(dict(clip_id=cid,procedure_id=row['procedure_id'],official_split='train',reuse=cid in reuse,video_sha256=sha(video)))
    for folder in ['src']:
        for path in (ROOT/folder).glob('*.py'):hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    extra=['configs/review_status.json','configs/reliability_scaled_a10_v1/uncertainty_raw.json','materials/checkpoints/sam2.1_hiera_tiny.pt','materials/verification/official_sparse_scope_v1/scope_evidence.json','data/manifests/procedure_mapping_verified_v5.csv','protocols/20261016_official_sparse_v1/PROTOCOL.md','protocols/20261017_train_expansion_v1/PROTOCOL.md','scripts/modal_train_expansion.py','scripts/prepare_train_expansion.py','scripts/analyze_train_expansion.py']
    extra.append('tests/test_train_expansion.py')
    for name in extra:hashes[name]=sha(ROOT/name)
    (P/'inputs.json').write_text(json.dumps(inputs,indent=2),encoding='utf8')
    hashes['protocols/20261017_train_expansion_v1/inputs.json']=sha(P/'inputs.json')
    freeze=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_gpu_dispatch=True,clip_count=170,new_clips=161,reused_clips=9,source_count=9,actual_gpu_required='NVIDIA A10',final_test=False,sha256=hashes)
    (P/'FREEZE.json').write_text(json.dumps(freeze,indent=2),encoding='utf8')
    print('Frozen',len(hashes),'hashes;170 training clips only')
if __name__=='__main__':main()
