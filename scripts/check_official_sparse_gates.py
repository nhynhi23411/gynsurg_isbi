"""Readiness for the amended official-only computational protocol, not clinical sign-off."""
from pathlib import Path
import json,csv,sys,hashlib,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.prepare_sequence import validate_metadata,validate_references
def main():
    base=ROOT/'data/manifests/sequence_official_sparse_v1';counts=collections.Counter();groups=collections.defaultdict(set);errors=[]
    files=sorted(base.glob('*.metadata.json'))
    for p in files:
        try:
            meta=json.loads(p.read_text());validate_metadata(meta)
            refs=list(csv.DictReader(p.with_name(p.name.replace('.metadata.json','.references.csv')).open(encoding='utf8')));validate_references(refs,meta)
            for r in refs:
                if hashlib.file_digest(Path(r['mask']).open('rb'),'sha256').hexdigest()!=r['mask_sha256']:raise ValueError('Reference hash mismatch')
            if hashlib.file_digest((ROOT/meta['grouping_evidence']).open('rb'),'sha256').hexdigest()!=meta['grouping_evidence_sha256']:raise ValueError('Grouping evidence differs')
            if meta['frame_count']!=125:raise ValueError('Unexpected window')
            counts[meta['official_split']]+=1;groups[meta['procedure_id']].add(meta['official_split'])
        except Exception as exc:errors.append(dict(file=p.name,error=str(exc)))
    if any(len(v)!=1 for v in groups.values()):errors.append(dict(error='Source crosses official split'))
    approval=json.loads((ROOT/'configs/review_status.json').read_text())
    report=dict(clips=len(files),official_split_counts=dict(counts),verified_source_procedures=len(groups),errors=errors,
       all_300_official_sequences_data_ready=len(files)==300 and not errors,
       development_execution_authorized=approval['approved_for_training'],final_test_authorized=approval['approved_for_test'],
       reference_mode='official_release_computational',clinical_review_complete=False,new_annotations_created=0,
       missing_required_reference_masks=0 if not errors else None,
       original_250_frame_tail_protocol_ready=False)
    (ROOT/'data/manifests/official_sparse_gate_status.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if errors or len(files)!=300:raise SystemExit(2)
if __name__=='__main__':main()
