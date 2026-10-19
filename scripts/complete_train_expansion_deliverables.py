"""Sequential CPU deliverables after the frozen controller has fully exited."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1')

def main():
    done=json.loads((BASE/'recovery_20261019/pipeline_complete.json').read_text())
    progress=json.loads((BASE/'cohort_progress.json').read_text())
    if not done['complete'] or progress['stage']!='complete' or progress['completed']!=170:
        raise ValueError('Complete controller and170 receipts required')
    jobs=[('rescore_train_expansion_present_only.py','analysis/present_only/report.json'),
          ('rescore_train_expansion_temporal.py','analysis/temporal_1s/report.json'),
          ('diagnose_full_training_oracle.py','analysis/restricted_oracle/report.json'),
          ('plot_train_expansion.py','analysis/train_expansion_summary.png')]
    for script,artifact in jobs:
        if (BASE/artifact).exists():
            print('PRESERVED existing',artifact,flush=True);continue
        print('RUN',script,flush=True)
        subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
    for script in ['audit_train_expansion_recovery.py','report_full_gynsurg_execution.py']:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
    print('CPU deliverables complete; inspect plot and refresh final documentation/source ledger',flush=True)

if __name__=='__main__':main()
