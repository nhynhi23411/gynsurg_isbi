"""Inventory persisted debug tensors without downloading or rerunning GPU."""
from pathlib import Path
import argparse,datetime,json
import modal
ROOT=Path(__file__).resolve().parents[1]

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);args=a.parse_args()
    volume=modal.Volume.from_name('isbi-gynsurg-results-v1',create_if_missing=False)
    entries=list(volume.iterdir('/'+args.run,recursive=True))
    states=[dict(path=e.path,bytes=e.size) for e in entries if Path(e.path).name.startswith('state_') and Path(e.path).suffix=='.npz']
    expected={f'{args.run}/{cid}/suite/{policy}_seed20261016/debug/state_{n:02d}.npz' for cid in ['059','086'] for policy in ['full','uncertainty'] for n in range(6)}
    if {s['path'].lstrip('/') for s in states}!=expected or any(s['bytes']<=0 for s in states):raise ValueError('Missing remote probability state')
    result=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_id=args.run,passed=True,states=states,state_count=len(states),total_bytes=sum(s['bytes'] for s in states),scope='Remote directory inventory/size only; no new tensor checksum or quality audit')
    (ROOT/'materials/verification/debug_remote_state_inventory.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('Remote debug states:',len(states),'bytes:',result['total_bytes'])

if __name__=='__main__':main()
