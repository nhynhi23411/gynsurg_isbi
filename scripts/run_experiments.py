"""Dry-run by default. Real runs remain gated by reviewed protocol in each CLI."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--manifest');a.add_argument('--input');a.add_argument('--registry',default=str(ROOT/'configs/experiment_registry.json'));a.add_argument('--split',choices=['val','test'],default='val');a.add_argument('--execute',action='store_true');args=a.parse_args()
runs=json.loads(Path(args.registry).read_text())
for run in runs:
    if ROOT.name=='ISBI_OvarianUS':
        if not args.manifest:a.error('--manifest required for ultrasound')
        cmd=[sys.executable,'-m','src.ultrasound','train','--manifest',args.manifest,'--config',run['config'],'--out',run['output']]
    else:
        if not args.input:a.error('--input required for surgery')
        cmd=[sys.executable,'-m','src.surgery','--input',args.input,'--config',run['config'],'--output',run.get('output','results/'+run['run_id']),'--split',args.split]
    print(subprocess.list2cmdline(cmd),flush=True)
    if args.execute:subprocess.run(cmd,cwd=ROOT,check=True)
