"""Cross-check the audit's core counts and links against saved evidence."""
from pathlib import Path
import csv,hashlib,json,re,datetime
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def main():
    if (ROOT/'configs/active_protocol.json').exists():
        from audit_official_sparse_evidence import main as audit_current
        return audit_current()
    hashes=list(csv.DictReader((ROOT/'SOURCE_CHECKSUMS.csv').open(encoding='utf8')))
    mismatches=[r['path'] for r in hashes if sha(ROOT/r['path'])!=r['sha256']]
    assert not mismatches,mismatches
    auth=read('protocols/20261016_execution_v1/USER_AUTHORIZATION.json')
    assert sha(ROOT/'PLAN_RULES.md')==auth['protocol_sha256']==sha(ROOT/'protocols/20261016_execution_v1/PLAN_RULES_APPROVED_SCOPE.md')
    recovery=read('materials/verification/task1_recovery_status.json')
    assert recovery['downloaded_sha256_verified_videos']==recovery['indexed_videos']==21
    assert recovery['downloaded_sha256_verified_bytes']==104049666773
    identity=read('data/manifests/source_identity_recovery_v2/summary.json')
    candidates=list(csv.DictReader((ROOT/'data/manifests/source_identity_recovery_v2/candidate_mapping.csv').open(encoding='utf8')))
    assert len(candidates)==300 and sum(r['pixel_identity_pass']=='True' for r in candidates)==identity['pixel_pass_clips']==282
    assert sum(r['pool_consistent']=='False' for r in candidates)==2
    assert not identity['mapping_verified'] and not (ROOT/'configs/active_sequence_review.json').exists()
    gates=read('data/manifests/execution_gate_status.json')
    assert gates['missing_mask_rows']==2400 and gates['pending_reference_rows']==6000
    assert gates['group_not_ready']==gates['clinical_scope_not_ready']==300
    assert not gates['ready_for_primary_evaluation'] and not read('configs/review_status.json')['approved_for_test']
    tests=(ROOT/'materials/verification/execution_unit_tests.txt').read_text(encoding='utf8')
    assert 'Ran 22 tests' in tests and '\nOK' in tests
    search=read('materials/verification/noisoi_reference_search_20261016.json')
    assert search['total_files']==18813 and not search['read_errors']
    blank=[r for r in search['reference_related_tables'] if 'reference_rater' in r['path']]
    assert len(blank)==4 and all(not any(r['nonempty'].values()) for r in blank)
    links=re.findall(r'\]\(<(D:/[^>]+)>\)',(ROOT/'AUDIT_REPORT_20261016.md').read_text(encoding='utf8'))
    missing_links=[p for p in links if not Path(p).exists()]
    assert not missing_links,missing_links
    result=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes_verified=len(hashes),original_plan_hash_preserved=True,local_evidence_links_verified=len(links),
        clinical_primary_ready=False,test_unlocked=False,unit_tests=22,
        task1_verified_videos=21,identity_auto_pass=282,identity_unresolved=18,tail_masks_missing=2400,
        noisoi_inventory_files=18813,blank_reader_tables=4,
        report_scope='Consistency of saved artifacts/counts; not independent clinical approval or new full raw-data checksum audit')
    (ROOT/'materials/verification/audit_validation_20261016.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
