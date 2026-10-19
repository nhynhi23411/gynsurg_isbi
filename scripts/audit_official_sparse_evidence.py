"""Validate the current saved evidence without opening test or rerunning inference."""
from pathlib import Path
import csv, datetime, hashlib, json, re
ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf8'))

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    hashes = list(csv.DictReader((ROOT/'SOURCE_CHECKSUMS.csv').open(encoding='utf8')))
    require(bool(hashes), 'Empty source ledger')
    for row in hashes:
        require(sha(ROOT/row['path']) == row['sha256'], 'Source changed: '+row['path'])
    auth = read('protocols/20261016_execution_v1/USER_AUTHORIZATION.json')
    require(sha(ROOT/'PLAN_RULES.md') == auth['protocol_sha256'] == sha(ROOT/'protocols/20261016_execution_v1/PLAN_RULES_APPROVED_SCOPE.md'), 'Original plan changed')
    require(not read('configs/review_status.json')['approved_for_test'], 'Final test unexpectedly enabled')
    gates = read('data/manifests/official_sparse_gate_status.json')
    require(gates['all_300_official_sequences_data_ready'] and not gates['errors'], 'Sparse data gate failed')
    require(gates['clips'] == 300 and gates['verified_source_procedures'] == 16, 'Wrong source cohort')
    require(gates['official_split_counts'] == {'train':170,'val':57,'test':73}, 'Official splits changed')
    require(not gates['clinical_review_complete'] and not gates['final_test_authorized'] and gates['new_annotations_created'] == 0, 'Unsupported review/annotation/test claim')
    require(gates['missing_required_reference_masks'] == 0, 'Missing sparse reference')
    rows = list(csv.DictReader((ROOT/'data/manifests/procedure_mapping_verified_v5.csv').open(encoding='utf8')))
    require(len(rows)==300 and len({r['clip_id'] for r in rows})==300 and all(r['mapping_verified']=='True' for r in rows), 'Mapping incomplete')
    require(len({r['procedure_id'] for r in rows})==16, 'Mapping procedure mismatch')
    qa_dir = ROOT/'materials/verification/task1_source_visual_review_v5'
    qa = read('materials/verification/task1_source_visual_review_v5/review_manifest.json')
    require(qa['visual_audit_complete'] and not qa['clinical_review_complete'] and len(qa['reviewed_sheet_sha256'])==21, 'Source QA incomplete')
    for name, digest in qa['reviewed_sheet_sha256'].items():
        require(sha(qa_dir/name)==digest, 'Reviewed sheet changed: '+name)
    recovery = read('materials/verification/task1_recovery_status.json')
    require(recovery['downloaded_sha256_verified_videos']==recovery['indexed_videos']==21 and recovery['downloaded_sha256_verified_bytes']==104049666773, 'Task1 receipt mismatch')
    tests = (ROOT/'materials/verification/execution_unit_tests.txt').read_text(encoding='utf8')
    count = int(re.search(r'Ran (\d+) tests', tests).group(1))
    require(count>=33 and '\nOK' in tests, 'Tests did not pass')
    counts = {}
    for run, expected in [('official_sparse_feasibility_20261016T144620Z',144),('official_sparse_tuning_20261016T151226Z',108),('official_sparse_validation_20261016T153640Z',9)]:
        accepted = read('results/'+run+'/cohort_acceptance.json')
        require(accepted['complete'] and len(accepted['checks'])==expected, 'Incomplete accepted cohort: '+run)
        counts[run] = expected
    sensitivity=read('results/official_sparse_feasibility_20261016T144620Z/sensitivity_1s/cohort_acceptance.json')
    require(sensitivity['complete'] and len(sensitivity['checks'])==144, 'Incomplete sensitivity cohort')
    validation=read('results/official_sparse_validation_20261016T153640Z/analysis/validation_report.json')
    require(not validation['final_test_open'] and not validation['validation_used_to_retune'], 'Validation scope changed')
    require(all(c['delta']<=0 and c['n_groups']==3 for c in validation['comparisons']), 'Report conclusion needs review')
    selection=read('results/official_sparse_tuning_20261016T151226Z/analysis/selection.json')
    require(not selection['proceed_gate_met_by_development'] and not selection['final_test_unlocked'], 'Selection/test status mismatch')
    report=(ROOT/'REPORT_OFFICIAL_SPARSE_EXECUTION_20261016.md').read_text(encoding='utf8')
    links=re.findall(r'\]\(<(D:/[^>]+)>\)', report)
    require(all(Path(p).exists() for p in links), 'Report contains missing local links')
    debug_reports=list((ROOT/'results').glob('debug_replay_*/analysis/debug_acceptance.json'))
    if (ROOT/'protocols/20261017_reliability_scaled_v1/FREEZE.json').exists():
        from check_reliability_scaled import main as check_prototype
        check_prototype()
    execution_path=ROOT/'materials/verification/reliability_execution_acceptance.json'
    if execution_path.exists():
        execution=json.loads(execution_path.read_text())
        require(execution['passed'] and not execution['final_test_open'] and not execution['parameters_retuned'], 'Prototype execution scope changed')
        audit=read('materials/verification/reliability_execution_audit.json')
        require(audit['passed'] and not audit['final_test_open'], 'Prototype execution audit failed')
    oracle_path=ROOT/'materials/verification/oracle_diagnostics_audit.json'
    if oracle_path.exists():
        oracle=json.loads(oracle_path.read_text())
        require(oracle['passed'] and not oracle['model_rerun'] and not oracle['gate_trained'] and not oracle['final_test_open'], 'Oracle diagnostic scope changed')
    for path in debug_reports:
        debug=json.loads(path.read_text(encoding='utf8'))
        require(debug['passed'] and not debug['final_test_open'] and len(debug['checks'])==4, 'Debug replay acceptance failed')
        require(all(c['query_trajectory_exact'] and c['trace_reproduces_run_metrics'] for c in debug['checks']), 'Debug replay trajectory/metrics mismatch')
        require(count>=35, 'Debug tests missing')
    expansion_summary=None
    expansion_root=ROOT/'results/train_expansion_20261017_v1'
    expansion_seal=expansion_root/'final_recovery_audit.json'
    if expansion_seal.exists():
        expansion=json.loads(expansion_seal.read_text())
        require(expansion['passed'] and expansion['clips']==170 and expansion['new_inference']==161 and expansion['reused']==9, 'Full-training acceptance incomplete')
        require(expansion['source_groups']==9 and not expansion['final_test_open'] and not expansion['independent_validation'], 'Wrong expansion evaluation scope')
        require(expansion['scientific_freeze_unchanged'] and expansion['source_bootstrap_reproduced'] and expansion['present_sensitivity_reproduced'] and expansion['temporal_sensitivity_reproduced'] and expansion['full_restricted_oracle_reproduced'] and expansion['compact_metric_transport_verified'], 'Expansion audit missing checks')
        resolved=expansion_root.resolve()
        for name,h in expansion['artifact_sha256'].items():
            artifact=(resolved/name).resolve()
            require(artifact.is_relative_to(resolved) and sha(artifact)==h, 'Sealed expansion artifact changed: '+name)
        apps=read('materials/verification/modal_owned_apps_completed_20261019.json')
        require(len(apps)==4 and all(a['state']=='stopped' and int(a['tasks'])==0 for a in apps), 'Owned recovery apps not stopped')
        require(count>=62, 'Current recovery software tests missing')
        expansion_summary=dict(clips=170,new_inference=161,reused=9,source_groups=9,sealed_artifacts_verified=len(expansion['artifact_sha256']),owned_apps_stopped=4,final_test_open=False)
    result=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,source_hashes_verified=len(hashes),original_plan_preserved=True,verified_mapping_clips=300,source_procedures=16,reviewed_source_sheets=21,unit_tests=count,accepted_case_runs=counts,reliability_prototype_case_runs={r['run_id']:r['case_runs'] for r in execution['runs']} if execution_path.exists() else {},local_report_links_checked=len(links),final_test_open=False,clinical_review_complete=False,new_annotations_created=0,scope='Saved evidence consistency and source ledger verification; not a new full raw-data checksum audit or clinical review')
    result['full_training_expansion']=expansion_summary
    (ROOT/'materials/verification/official_sparse_evidence_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
