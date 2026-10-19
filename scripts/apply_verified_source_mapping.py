"""Apply verified engineering source identity in a new metadata version only."""
from pathlib import Path
import csv,json,datetime,shutil,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1]

def main():
    a=argparse.ArgumentParser();a.add_argument('--base',default='data/manifests/source_identity_recovery_v2');a.add_argument('--version',default='v2');args=a.parse_args()
    if args.version not in ['v2','v5']:raise ValueError('Unsupported mapping output version')
    base=ROOT/args.base;summary=json.loads((base/'summary.json').read_text())
    for key in ['all_supplementary_counts_match','all_official_pools_match','all_durations_match_official_labels']:
        if summary[key] is not True:raise ValueError('Source identity gate failed: '+key)
    if summary['pixel_pass_clips']!=300:raise ValueError('Unresolved source identity candidates')
    mapping=list(csv.DictReader((base/'candidate_mapping.csv').open(encoding='utf8')))
    qa=list(csv.DictReader((base/'source_visual_review.csv').open(encoding='utf8')))
    expected={r['source_video_id'] for r in mapping}
    if {r['source_video_id'] for r in qa}!=expected:raise ValueError('Visual QA source coverage incomplete')
    if not all(r['source_identity_consistent'].lower()=='true' and r['reviewer'] and r['date'] and r['panels_reviewed'] for r in qa):raise ValueError('Visual QA not completed')
    out=ROOT/f'data/manifests/sequence_review_{args.version}_grouped'
    if out.exists():raise FileExistsError('Do not overwrite grouped metadata version')
    out.mkdir();qa_by_video={r['source_video_id']:r for r in qa};verified=[]
    for r in mapping:
        cid=r['clip_id'];old=ROOT/f'data/manifests/sequence_review_v1/{cid}.metadata.draft.json'
        meta=json.loads(old.read_text());meta.update(procedure_id=r['procedure_id'],source_video_id=r['source_video_id'],
            group_verified=True,grouping_evidence=str((base/'all_decisions.json').relative_to(ROOT)),
            grouping_review_evidence=str((base/'source_visual_review.csv').relative_to(ROOT)),
            grouping_method='six-frame RGB source correspondence, source margin, official counts/pools/durations and visual source QA',
            source_timestamp_note='Source identity verified; refined start is approximate, see matched per-frame source times in evidence; Task3-to-Task2 alignment audited separately',
            grouping_reviewer=qa_by_video[r['source_video_id']]['reviewer'],patient_id='')
        # Scientific reference/clinical scope/test authorization are independent gates.
        assert meta['clinical_scope_reviewed'] is False and meta['human_annotation_complete'] is False
        (out/(cid+'.metadata.draft.json')).write_text(json.dumps(meta,indent=2),encoding='utf8')
        shutil.copy2(old.with_name(cid+'.references.draft.csv'),out/(cid+'.references.draft.csv'))
        verified.append(dict(**{k:v for k,v in r.items() if k!='mapping_verified'},mapping_verified=True,
            verified_by=qa_by_video[r['source_video_id']]['reviewer'],verified_date=qa_by_video[r['source_video_id']]['date']))
    with (ROOT/f'data/manifests/procedure_mapping_verified_{args.version}.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(verified[0]));w.writeheader();w.writerows(verified)
    active=dict(directory=str(out.relative_to(ROOT)),grouping_verified=True,clinical_scope_reviewed=False,human_annotation_complete=False,
        grouping_evidence_sha256=hashlib.file_digest((base/'all_decisions.json').open('rb'),'sha256').hexdigest(),
        updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (ROOT/'configs/active_sequence_review.json').write_text(json.dumps(active,indent=2),encoding='utf8')
    summary['mapping_verified']=True;summary['applied_metadata_directory']=str(out.relative_to(ROOT))
    (base/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('Applied 300 verified source mappings to a new metadata version; clinical/test gates remain closed')

if __name__=='__main__':main()
