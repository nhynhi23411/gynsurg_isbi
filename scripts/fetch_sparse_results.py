"""Recover completed development outputs in parallel; leave large tensors remote."""
from pathlib import Path
import argparse,json,concurrent.futures
import modal
ROOT=Path(__file__).resolve().parents[1]
def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--stage',choices=['pilot','feasibility','repeatability','tuning','debug','v2_engineering','v2_development'],required=True);args=a.parse_args()
    volume=modal.Volume.from_name('isbi-gynsurg-results-v1',create_if_missing=False)
    entries=list(volume.iterdir('/'+args.run,recursive=True));by_path={e.path.lstrip('/'):e for e in entries}
    expected=['058'] if args.stage in ['pilot','repeatability'] else ['001','038','059','086','105','116','138','146','160']
    if args.stage in ['debug','v2_engineering']:expected=['059','086']
    markers=[args.run+'/'+cid+'/execution.json' for cid in expected]
    absent=[m for m in markers if m not in by_path]
    if absent:
        print(json.dumps(dict(complete=False,finished=len(markers)-len(absent),expected=len(expected),pending=[Path(m).parent.name for m in absent])));return
    out=ROOT/'results'/args.run;out.mkdir(exist_ok=True)
    def fetch(entry):
        path=Path(entry.path.lstrip('/'));dest=out/path.relative_to(args.run)
        # Skip complete local files, including those already delivered by the launcher.
        if dest.is_file() and dest.stat().st_size==entry.size:return
        data=b''.join(volume.read_file(entry.path));dest.parent.mkdir(parents=True,exist_ok=True)
        if len(data)!=entry.size:raise ValueError('Incomplete output transfer')
        temp=dest.with_suffix(dest.suffix+'.fetch.part');temp.write_bytes(data);temp.replace(dest)
    selected=[e for e in entries if Path(e.path).suffix in (['.csv','.json','.txt','.png'] if args.stage in ['debug','v2_engineering','v2_development'] else ['.csv','.json','.txt'])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:list(ex.map(fetch,selected))
    reports=[json.loads((out/cid/'execution.json').read_text()) for cid in expected]
    cohort=dict(run_id=args.run,stage=args.stage,reports=reports,final_test=False,download_recovered_parallel=True)
    if args.stage=='debug':cohort['source_run']='official_sparse_feasibility_20261016T144620Z'
    if args.stage.startswith('v2_'):
        cohort['source_run']='official_sparse_feasibility_20261016T144620Z'
        cohort['stage']=args.stage.removeprefix('v2_')
    (out/'cohort_execution.json').write_text(json.dumps(cohort,indent=2))
    print(json.dumps(dict(complete=True,clips=len(reports),small_artifacts=len(selected),large_tensors_retained_remote=True)))
if __name__=='__main__':main()
