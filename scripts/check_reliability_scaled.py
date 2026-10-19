"""Read-only readiness check for the prepared prototype, not an efficacy gate."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]

def main():
    freeze=json.loads((ROOT/'protocols/20261017_reliability_scaled_v1/FREEZE.json').read_text())
    for relative,digest in freeze['hashes'].items():
        with (ROOT/relative).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=digest:raise ValueError('Freeze invalidated: '+relative)
    log=(ROOT/'materials/verification/execution_unit_tests.txt').read_text(encoding='utf8')
    if '\nOK' not in log or int(re.search(r'Ran (\d+) tests',log).group(1))<41:raise ValueError('Required test suite missing')
    smoke=json.loads((ROOT/'materials/verification/reliability_scaled_cpu_smoke.json').read_text())
    if not smoke['passed'] or not smoke['synthetic_only'] or len(smoke['recipes'])!=6:raise ValueError('CPU smoke incomplete')
    if json.loads((ROOT/'configs/review_status.json').read_text())['approved_for_test']:raise ValueError('Final test unexpectedly enabled')
    recipes=json.loads((ROOT/'configs/reliability_scaled_v1/experiment_registry.json').read_text())
    if len(recipes)!=6:raise ValueError('Wrong recipe count')
    for r in recipes:
        c=json.loads((ROOT/r['config']).read_text())
        if c['required_gpu_name']!=freeze['actual_gpu_required'] or c['densities']!=[0,1,2,4]:raise ValueError('Hardware/budget contract changed')
    result=dict(passed=True,status='prepared_and_synthetically_tested_no_new_method_gpu_results',freeze_hashes_verified=len(freeze['hashes']),unit_tests=int(re.search(r'Ran (\d+) tests',log).group(1)),recipes=6,cpu_smoke=True,efficacy_established=False,final_test_open=False)
    execution=ROOT/'materials/verification/reliability_execution_acceptance.json'
    if execution.exists():
        saved=json.loads(execution.read_text())
        result['executed_hardware_variant_runs']=saved['runs']
        result['status']='hardware_variant_exploratory_results_available_final_test_locked'
    (ROOT/'materials/verification/reliability_scaled_readiness.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
