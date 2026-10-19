"""Summarize retrieval evidence without granting clinical or final-test approval."""
from pathlib import Path
import csv,json,collections,datetime,hashlib,sys,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.source_identity import identity_decision
EXPECTED=[37,21,27,19,11,22,8,14,0,11,19,0,21,0,0,17,14,26,13,0,20]

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refinement-dir',default='materials/verification/task1_refinement_v1');parser.add_argument('--output-dir',default='data/manifests/source_identity_recovery_v2');args=parser.parse_args()
    coarse=list((ROOT/'materials/verification/task1_recovery_v1').glob('??.json'))
    if len(coarse)!=21:raise ValueError('Incomplete 21-source search')
    manifest=json.loads((ROOT/'materials/verification/task1_mirror_manifest.json').read_text())
    expected_query_sha=hashlib.file_digest((ROOT/'data/manifests/task3_identity_verification_cv2_v2.npz').open('rb'),'sha256').hexdigest()
    coarse_by_clip=collections.defaultdict(list)
    for file in coarse:
        for candidate in json.loads(file.read_text())['candidates']:coarse_by_clip[candidate['clip_id']].append(candidate)
    def candidate_key(c):return c['clip_id'],c['source_video_id'],c['coarse_start_s'],c['local_rank']
    required=set()
    for cid,items in coarse_by_clip.items():
        ranked=sorted(items,key=lambda x:x['coarse_rgb_mse']);chosen=ranked[:6]
        alternate=next(x for x in ranked if x['source_video_id']!=ranked[0]['source_video_id'])
        required.update(candidate_key(x) for x in chosen+[alternate])
    files=list((ROOT/args.refinement_dir).glob('??.json'))
    by_clip=collections.defaultdict(list);hashes={}
    for file in files:
        report=json.loads(file.read_text());hashes[file.name]=hashlib.file_digest(file.open('rb'),'sha256').hexdigest()
        if report['query_sha256']!=expected_query_sha:raise ValueError('Verification query artifact differs')
        for row in report['candidates']:by_clip[row['clip_id']].append(row)
    observed={candidate_key(c) for items in by_clip.values() for c in items}
    if not required<=observed:raise ValueError('Candidate refinement incomplete; include all selected alternative sources')
    if set(by_clip)!={f'{i:03}' for i in range(1,301)}:raise ValueError('Incomplete candidate refinement')
    out=ROOT/args.output_dir;out.mkdir(exist_ok=True)
    rows=[];decisions=[];counts=collections.Counter();passed_counts=collections.Counter()
    for cid,items in sorted(by_clip.items()):
        decision=identity_decision(items);best=decision['best'];vid=best['source_video_id'];counts[vid]+=1
        if best.get('identity_model')=='bounded_monotonic_temporal_v5':decision['threshold_version']='pixel_v2_temporal_v5'
        expected_pool=range(1,11) if int(cid)<=170 else (range(11,17) if int(cid)<=227 else range(17,22))
        pool_ok=int(vid) in expected_pool
        passed=decision['pixel_identity_pass'] and pool_ok
        if passed:passed_counts[vid]+=1
        decisions.append(dict(clip_id=cid,**decision,pool_consistent=pool_ok))
        rows.append(dict(clip_id=cid,source_video_id=vid,procedure_id='AutoLaparo_video_'+vid,
            refined_start_frame=best['refined_start_frame'],refined_start_s=best['refined_start_s'],
            mean_rgb_mae=best['six_frame_rgb_mae'],max_rgb_mae=best['max_frame_rgb_mae'],
            min_rgb_correlation=min(best.get('per_frame_rgb_correlation',[float('nan')])),
            alternative_source_margin=decision['source_margin'],alternative_source_ratio=decision['source_ratio'],
            pixel_identity_pass=passed,pool_consistent=pool_ok,source_sha256=best['source_video_sha256'],
            evidence_path=best['evidence_path'],matched_source_frames=';'.join(map(str,best.get('matched_frames',[]))),identity_model=best.get('identity_model','fixed_offsets_v2'),mapping_verified=False))
    count_check=[dict(source_video_id=f'{i:02}',expected_clips=n,best_candidate_clips=counts[f'{i:02}'],
                     pixel_pass_clips=passed_counts[f'{i:02}']) for i,n in enumerate(EXPECTED,1)]
    duration=json.loads((ROOT/'materials/verification/task1_recovery_status.json').read_text())
    duration_ok=len(duration['original_label_duration_checks'])==21 and all(abs(r['difference_s'])<=1.1 and r['fps']=='25/1' and r['width']==1920 and r['height']==1080 for r in duration['original_label_duration_checks'])
    count_ok=all(r['expected_clips']==r['best_candidate_clips']==r['pixel_pass_clips'] for r in count_check)
    summary=dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),repo=manifest['repo'],
        revision=manifest['revision'],clips=300,pixel_pass_clips=sum(r['pixel_identity_pass'] for r in rows),
        all_supplementary_counts_match=count_ok,all_official_pools_match=all(r['pool_consistent'] for r in rows),
        all_durations_match_official_labels=duration_ok,source_counts=count_check,refinement_input_hashes=hashes,
        clinical_reference_reviewed=False,mapping_verified=False,refinement_directory=args.refinement_dir,
        next_step='Visual audit of every contributing source and high-error/outlier clips; resolve any failed pixel/count/pool/duration checks before applying')
    write_csv(out/'candidate_mapping.csv',rows)
    (out/'all_decisions.json').write_text(json.dumps(decisions,indent=2),encoding='utf8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    qa=out/'source_visual_review.csv'
    if not qa.exists():
        write_csv(qa,[dict(source_video_id=r['source_video_id'],reviewer='',date='',panels_reviewed='',
                           source_identity_consistent='',notes='') for r in count_check if r['expected_clips']])
    print(json.dumps({k:v for k,v in summary.items() if k not in ['refinement_input_hashes','source_counts']},indent=2))

if __name__=='__main__':main()
