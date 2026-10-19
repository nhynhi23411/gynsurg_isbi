"""Report readiness without converting unknown fields into approvals."""
import argparse,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    a=argparse.ArgumentParser();a.add_argument('--require-ready',action='store_true');a.add_argument('--metadata-dir');args=a.parse_args()
    active=ROOT/'configs/active_sequence_review.json'
    directory=args.metadata_dir or (json.loads(active.read_text())['directory'] if active.exists() else 'data/manifests/sequence_review_v1')
    base=(ROOT/directory).resolve()
    if not base.is_relative_to(ROOT.resolve()):raise ValueError('Metadata directory must be inside this project')
    records=[]
    for p in sorted(base.glob('*.metadata.draft.json')):
        m=json.loads(p.read_text(encoding='utf8'));refs=list(csv.DictReader(p.with_name(p.name.replace('.metadata.draft.json','.references.draft.csv')).open(encoding='utf8')))
        missing=[r for r in refs if not r['mask'] or not Path(r['mask']).is_file()]
        pending=[r for r in refs if r['annotation_complete'].lower()!='true']
        records.append(dict(clip_id=m['clip_id'],group_ready=m['group_verified'] is True and bool(m['procedure_id']),exposure_ready=m['exposure_clean'] is True,permission_ready=m['permission_status']=='verified',clinical_scope_ready=m.get('clinical_scope_reviewed') is True,reference_rows=len(refs),missing_mask_rows=len(missing),pending_reference_rows=len(pending)))
    report=dict(clips=len(records),group_not_ready=sum(not r['group_ready'] for r in records),exposure_not_ready=sum(not r['exposure_ready'] for r in records),permission_not_ready=sum(not r['permission_ready'] for r in records),clinical_scope_not_ready=sum(not r['clinical_scope_ready'] for r in records),reference_rows=sum(r['reference_rows'] for r in records),missing_mask_rows=sum(r['missing_mask_rows'] for r in records),pending_reference_rows=sum(r['pending_reference_rows'] for r in records))
    report['ready_for_primary_evaluation']=bool(records) and all(r['group_ready'] and r['exposure_ready'] and r['permission_ready'] and r['clinical_scope_ready'] and not r['missing_mask_rows'] and not r['pending_reference_rows'] for r in records)
    (ROOT/'data/manifests/execution_gate_status.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
    if args.require_ready and not report['ready_for_primary_evaluation']:raise SystemExit(2)
if __name__=='__main__':main()
