"""Create reviewed-release metadata only from verified source mappings."""
from pathlib import Path
import csv,json,hashlib,datetime,argparse
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def write(p,rows):
    with p.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    a=argparse.ArgumentParser();a.add_argument('--mapping',required=True);args=a.parse_args()
    mapping=list(csv.DictReader((ROOT/args.mapping).open(encoding='utf8')))
    if not mapping or not all(r['mapping_verified'].lower()=='true' for r in mapping):raise ValueError('Use verified source mapping, never retrieval candidates')
    proof=ROOT/'materials/verification/official_sparse_scope_v1/scope_evidence.json'
    if not json.loads(proof.read_text())['engineering_scope_review_complete']:raise ValueError('Scope engineering QA pending')
    out=ROOT/'data/manifests/sequence_official_sparse_v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir();cohort=[]
    for r in mapping:
        cid=r['clip_id'];meta=json.loads((ROOT/'configs/sequence_metadata_template.json').read_text())
        official='train' if int(cid)<=170 else ('val' if int(cid)<=227 else 'test')
        meta.update(procedure_id=r['procedure_id'],source_video_id=r['source_video_id'],clip_id=cid,frame_count=125,
           query_ids=[0,24,49],scoring_ids=[74,99,124],group_verified=True,exposure_clean=True,permission_status='verified',
           official_split=official,split='test' if official=='test' else 'val',clinical_scope_reviewed=False,
           reference_review_mode='official_release_computational',official_reference_scope_verified=True,
           official_annotation_completeness='dataset_defined',human_annotation_complete=True,
           human_annotation_origin='AutoLaparo official released annotation; no new labels',
           reference_scope_evidence=proof.relative_to(ROOT).as_posix(),reference_scope_evidence_sha256=sha(proof),
           grouping_evidence=args.mapping,grouping_evidence_sha256=sha(ROOT/args.mapping),
           protocol='protocols/20261016_official_sparse_v1/PROTOCOL.md',protocol_sha256=sha(ROOT/'protocols/20261016_official_sparse_v1/PROTOCOL.md'),
           limitation='No independent clinical review; dataset-defined reference targets only')
        refs=[]
        for t in [0,24,49,74,99,124]:
            for k,target in enumerate(['uterus','primary_tools_union']):
                p=ROOT/f'data/derived/task3_official_v1/{cid}{t+1:03}_{target}.png'
                if not p.is_file():raise ValueError('Missing released reference '+str(p))
                refs.append(dict(frame_index=t,structure=k,mask=str(p.resolve()),annotation_complete='true',mask_sha256=sha(p),annotation_origin='official_Task3'))
        (out/(cid+'.metadata.json')).write_text(json.dumps(meta,indent=2),encoding='utf8');write(out/(cid+'.references.csv'),refs)
        cohort.append(dict(clip_id=cid,procedure_id=r['procedure_id'],source_video_id=r['source_video_id'],official_split=official,split=meta['split']))
    write(out/'cohort.csv',cohort)
    # Fixed source-balanced feasibility sample selected before segmentation metrics.
    selected=[min([r for r in cohort if r['source_video_id']==vid],key=lambda r:r['clip_id']) for vid in sorted({r['source_video_id'] for r in cohort if r['official_split']=='train'})]
    write(out/'development_feasibility_cohort.csv',selected)
    registry=[];cfgout=ROOT/'configs/official_sparse_v1';cfgout.mkdir(exist_ok=True)
    for recipe in json.loads((ROOT/'configs/experiment_registry.json').read_text()):
        cfg=json.loads((ROOT/recipe['config']).read_text());cfg.update(sam_precision='bfloat16',failure_threshold=None)
        path=cfgout/(recipe['run_id']+'.json');path.write_text(json.dumps(cfg,indent=2))
        registry.append(dict(run_id=recipe['run_id'],config=path.relative_to(ROOT).as_posix(),output=recipe['output']))
    (cfgout/'experiment_registry.json').write_text(json.dumps(registry,indent=2))
    (out/'preparation.json').write_text(json.dumps(dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),clips=len(cohort),reference_rows=len(cohort)*12,missing_required_masks=0,frame_count=125,query_ids=[0,24,49],scoring_ids=[74,99,124],correction_counts=[0,1,2,5],actual_densities=[0,.8,1.6,4],clinical_review_complete=False,engineering_official_scope_verified=True,final_test_unlocked=False,feasibility_clips=[r['clip_id'] for r in selected]),indent=2))
    print('Prepared',len(cohort),'official sparse sequences;',len(selected),'source-balanced development clips; final test remains locked')
if __name__=='__main__':main()
