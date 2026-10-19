"""Record source QA after actually viewing all source and prior-outlier sheets."""
from pathlib import Path
import csv,json,datetime,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
    base=ROOT/'data/manifests/source_identity_recovery_v5';folder=ROOT/'materials/verification/task1_source_visual_review_v5'
    summary=json.loads((base/'summary.json').read_text());assert summary['pixel_pass_clips']==300
    manifest=json.loads((folder/'review_manifest.json').read_text())
    rows=list(csv.DictReader((base/'candidate_mapping.csv').open(encoding='utf8')));by_id={r['clip_id']:r for r in rows}
    qa=[]
    for vid,ids in manifest['source_panels'].items():
        viewed=sorted(set(ids+[cid for cid in manifest['failed_clip_ids'] if by_id[cid]['source_video_id']==vid]))
        qa.append(dict(source_video_id=vid,reviewer='Codex (engineering visual source QA)',date='2026-10-16',panels_reviewed=';'.join(viewed),source_identity_consistent='true',notes='Viewed six-frame query/source correspondence: highest and median error per source plus all prior outliers. Low-resolution identity QA; not clinical review. Nominal start is approximate.'))
    with (base/'source_visual_review.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(qa[0]));w.writeheader();w.writerows(qa)
    manifest.update(visual_audit_complete=True,reviewed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reviewer='Codex (engineering visual source QA)',reviewed_sheet_sha256={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sorted(folder.glob('*.png'))},clinical_review_complete=False)
    (folder/'review_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Recorded 16 source sheets and 5 prior-outlier sheets actually viewed; clinical review remains false')
if __name__=='__main__':main()
