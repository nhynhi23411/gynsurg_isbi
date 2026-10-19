"""Write fixed prototype configs/freeze manifest; never launches GPU."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    folder=ROOT/'configs/reliability_scaled_v1'
    if folder.exists():raise FileExistsError('Prototype configs already exist; do not silently change frozen version')
    folder.mkdir();base=json.loads((ROOT/'configs/official_sparse_v1/full_seed20261016.json').read_text())
    recipes=[('uncertainty_raw','uncertainty','raw'),('uncertainty_fallback','uncertainty','confidence_fallback_v1'),('error_raw','error','raw'),('scaled_raw','scaled_diversity_v1','raw'),('scaled_fallback','scaled_diversity_v1','confidence_fallback_v1'),('legacy_full_raw','full','raw')]
    registry=[]
    for name,policy,guard in recipes:
        cfg=dict(base,seed=20261017,policy=policy,propagation_reliability=guard,reliability_delta=.25,relative_diversity=.25,required_gpu_name='NVIDIA A10G',prototype_protocol='20261017_reliability_scaled_v1')
        path=folder/(name+'.json');path.write_text(json.dumps(cfg,indent=2),encoding='utf8')
        registry.append(dict(run_id=name,config=path.relative_to(ROOT).as_posix()))
    (folder/'experiment_registry.json').write_text(json.dumps(registry,indent=2),encoding='utf8')
    paths=list((ROOT/'src').glob('*.py'))+list(folder.glob('*.json'))+list((ROOT/'tests').glob('*.py'))+[ROOT/'protocols/20261017_reliability_scaled_v1/PROTOCOL.md']+[ROOT/'scripts'/name for name in ['modal_reliability_scaled.py','analyze_reliability_scaled.py','prepare_reliability_scaled.py','smoke_reliability_scaled.py']]
    receipt=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_id='20261017_reliability_scaled_v1',status='prepared_not_run',final_test_open=False,recipes=6,actual_gpu_required='NVIDIA A10G',parameters_selected_using_scoring_gold=False,design_informed_by_previous_negative_results=True,hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})
    (ROOT/'protocols/20261017_reliability_scaled_v1/FREEZE.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print('Prepared six recipes; source/config/protocol frozen; no GPU launched')

if __name__=='__main__':main()
