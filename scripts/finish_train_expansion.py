"""Continuation of the current execution pipeline, not a scheduled automation.

Wait for the running controller to finish, then run frozen acceptance/analysis.
Never dispatch inference, modify a freeze or accept partial metrics.
"""
from pathlib import Path
import datetime,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1')
def main():
    limit=time.monotonic()+4*3600
    while time.monotonic()<limit:
        if (BASE/'cohort_execution.json').exists():
            result=subprocess.run([sys.executable,str(ROOT/'scripts/analyze_train_expansion.py')],cwd=ROOT)
            (BASE/'analysis_pipeline_status.json').write_text(json.dumps(dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=result.returncode,final_test_open=False),indent=2),encoding='utf8')
            if result.returncode:raise SystemExit(result.returncode)
            for script in ['finalize_official_sparse.py','write_sparse_execution_report.py','finalize_official_sparse.py','audit_official_sparse_evidence.py']:
                subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
            return
        time.sleep(15)
    raise TimeoutError('No complete cohort receipt after four hours; no partial-result analysis performed')
if __name__=='__main__':main()
