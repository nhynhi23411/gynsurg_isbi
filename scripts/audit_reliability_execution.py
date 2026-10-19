"""Audit real prototype evidence, frozen A10 contract and cross-stage repetition."""
from pathlib import Path
import argparse,datetime,hashlib,json,re
import numpy as np,pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--engineering',required=True);a.add_argument('--development');args=a.parse_args()
    freeze=json.loads((ROOT/'protocols/20261017_reliability_scaled_a10_v1/FREEZE.json').read_text())
    for name,digest in freeze['hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source/config changed: '+name)
    for recipe in json.loads((ROOT/'configs/reliability_scaled_a10_v1/experiment_registry.json').read_text()):
        original=json.loads((ROOT/'configs/reliability_scaled_v1'/(recipe['run_id']+'.json')).read_text())
        revised=json.loads((ROOT/recipe['config']).read_text())
        changed={key for key in set(original)|set(revised) if original.get(key)!=revised.get(key)}
        if changed!={'required_gpu_name','prototype_protocol'}:raise ValueError('Hardware-only revision changed algorithm parameters')
    if json.loads((ROOT/'configs/review_status.json').read_text())['approved_for_test']:raise ValueError('Test unlocked unexpectedly')
    runs=[args.engineering]+([args.development] if args.development else []);inventories=[]
    for run in runs:
        base=ROOT/'results'/run;execution=json.loads((base/'cohort_execution.json').read_text());accepted=json.loads((base/'analysis/cohort_acceptance.json').read_text())
        if not accepted['complete']:raise ValueError('Incomplete cohort')
        expected=12 if execution['stage']=='engineering' else 54
        if len(accepted['checks'])!=expected:raise ValueError('Wrong accepted case-runs')
        for r in execution['reports']:
            if r['actual_gpu']!='NVIDIA A10' or r['official_split']!='train':raise ValueError('Hardware/split contract failed')
            for recipe in json.loads((ROOT/'configs/reliability_scaled_a10_v1/experiment_registry.json').read_text()):
                folder=base/r['clip_id']/'suite'/recipe['run_id'];config=json.loads((folder/'provenance.json').read_text())['config']
                if config!=json.loads((ROOT/recipe['config']).read_text()):raise ValueError('Run config differs from freeze')
                if len(pd.read_csv(folder/'debug/state_metrics.csv'))!=72:raise ValueError('State count mismatch')
        png=list(base.glob('*/suite/*/debug/*.png'))
        for p in png:
            with Image.open(p) as image:image.verify()
        remote=base/'remote_artifact_inventory.json'
        nstates=None
        if remote.exists():
            entries=json.loads(remote.read_text());states=[e for e in entries if Path(e['path']).name.startswith('state_') and Path(e['path']).suffix=='.npz']
            if len(states)!=expected*6 or any(e['bytes']<=0 for e in states):raise ValueError('Missing remote probability snapshots')
            nstates=len(states)
        inventories.append(dict(run=run,case_runs=expected,local_png_integrity_checked=len(png),remote_states_inventoried=nstates))
    repeated=[]
    if args.development:
        for cid in ['059','086']:
            for recipe in json.loads((ROOT/'configs/reliability_scaled_a10_v1/experiment_registry.json').read_text()):
                aroot=ROOT/'results'/args.engineering/cid/'suite'/recipe['run_id'];broot=ROOT/'results'/args.development/cid/'suite'/recipe['run_id']
                qa=pd.read_csv(aroot/'queries.csv');qb=pd.read_csv(broot/'queries.csv')
                fa=pd.read_csv(aroot/'per_frame.csv');fb=pd.read_csv(broot/'per_frame.csv')
                keys=['density','structure','frame_index'];pair=fa.merge(fb,on=keys,suffixes=('_a','_b'),validate='one_to_one')
                repeated.append(dict(clip_id=cid,recipe=recipe['run_id'],query_trajectory_exact=qa.equals(qb),max_dice_difference=float(np.abs(pair.dice_a-pair.dice_b).max())))
    text=(ROOT/'REPORT_RELIABILITY_EXECUTION_20261017.md').read_text(encoding='utf8')
    links=re.findall(r'\]\(<(D:/[^>]+)>\)',text)
    if not all(Path(p).exists() for p in links):raise ValueError('Broken report link')
    result=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,freeze_hashes_verified=len(freeze['hashes']),inventories=inventories,cross_stage_repetition=repeated,final_test_open=False,new_annotations=0,parameters_retuned=False,scope='Computational saved-artifact audit, not clinical review or confirmatory efficacy')
    (ROOT/'materials/verification/reliability_execution_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
