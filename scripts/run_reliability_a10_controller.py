"""Resume frozen A10 cases; retry allocation mismatch, never change hardware contract."""
from pathlib import Path
import argparse,concurrent.futures,datetime,hashlib,json
from modal_reliability_scaled_a10 import app,execute,volume,ROOT,IDS

def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['engineering','development'],default='engineering');a.add_argument('--run');a.add_argument('--max-allocation-rounds',type=int,default=12);args=a.parse_args()
    if not 1<=args.max_allocation_rounds<=12:raise ValueError('Allocation rounds must be 1–12')
    freeze=json.loads((ROOT/'protocols/20261017_reliability_scaled_a10_v1/FREEZE.json').read_text())
    for relative,digest in freeze['hashes'].items():
        with (ROOT/relative).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=digest:raise ValueError('Freeze invalidated: '+relative)
    run_id=args.run or ('reliability_scaled_a10_'+args.stage+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out=ROOT/'results'/run_id;out.mkdir(exist_ok=True);expected=['059','086'] if args.stage=='engineering' else IDS
    previous=out/'controller_snapshot.py'
    if previous.exists():
        snapshots=out/'controller_history';snapshots.mkdir(exist_ok=True)
        (snapshots/('controller_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.py')).write_bytes(previous.read_bytes())
    (out/'controller_snapshot.py').write_text(Path(__file__).read_text(encoding='utf8'),encoding='utf8')
    (out/'invocation.json').write_text(json.dumps(dict(stage=args.stage,freeze_sha256=hashlib.sha256((ROOT/'protocols/20261017_reliability_scaled_a10_v1/FREEZE.json').read_bytes()).hexdigest(),controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),final_test=False,development_png_policy='Only query1/query2 score124 panels downloaded; all other panels retained private remote'),indent=2),encoding='utf8')
    records={};attempts=[]
    for cid in expected:
        try:records[cid]=json.loads(b''.join(volume.read_file(run_id+'/'+cid+'/execution.json')))
        except FileNotFoundError:pass
    for attempt in range(1,args.max_allocation_rounds+1):
        pending=[cid for cid in expected if cid not in records]
        if not pending:break
        print('ALLOCATION_ROUND',attempt,'pending',pending,flush=True)
        with app.run():
            for index,result in enumerate(execute.starmap([(run_id,cid,args.stage) for cid in pending],return_exceptions=True,order_outputs=True)):
                cid=pending[index]
                if isinstance(result,BaseException):
                    attempts.append(dict(round=attempt,clip_id=cid,error=str(result),accepted=False))
                    (out/'allocation_attempts.json').write_text(json.dumps(attempts,indent=2),encoding='utf8')
                    if 'Frozen GPU contract' not in str(result):raise result
                    print('ALLOCATION_MISMATCH',cid,'no inference on this worker',flush=True)
                else:
                    record,_=result;records[cid]=record;attempts.append(dict(round=attempt,clip_id=cid,accepted=True,actual_gpu=record['actual_gpu']))
                    print('V2_ACCEPTED',cid,round(record['total_seconds'],2),flush=True)
        (out/'allocation_attempts.json').write_text(json.dumps(attempts,indent=2),encoding='utf8')
    if set(records)!=set(expected):raise RuntimeError('Matching A10 allocation unavailable after bounded rounds; completed cases preserved for resume')
    entries=list(volume.iterdir('/'+run_id,recursive=True));selected=[e for e in entries if Path(e.path).suffix in ['.json','.csv','.png','.txt']]
    if args.stage=='development':selected=[e for e in selected if Path(e.path).suffix!='.png' or Path(e.path).name in ['query_01_score_124.png','query_02_score_124.png']]
    (out/'remote_artifact_inventory.json').write_text(json.dumps([dict(path=e.path,bytes=e.size) for e in entries],indent=2),encoding='utf8')
    def fetch(e):
        dest=out/Path(e.path.lstrip('/')).relative_to(run_id)
        if dest.is_file() and dest.stat().st_size==e.size:return
        data=b''.join(volume.read_file(e.path));dest.parent.mkdir(parents=True,exist_ok=True)
        if len(data)!=e.size:raise ValueError('Incomplete download')
        temp=dest.with_suffix(dest.suffix+'.part');temp.write_bytes(data);temp.replace(dest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:list(pool.map(fetch,selected))
    (out/'cohort_execution.json').write_text(json.dumps(dict(run_id=run_id,stage=args.stage,source_run='official_sparse_feasibility_20261016T144620Z',reports=[records[cid] for cid in expected],final_test=False),indent=2),encoding='utf8')
    print(out,flush=True)

if __name__=='__main__':main()
