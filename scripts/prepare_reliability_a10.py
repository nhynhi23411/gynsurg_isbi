"""Create a hardware-only execution revision after pre-inference A10G rejection."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OLD='20261017_reliability_scaled_v1';NEW='20261017_reliability_scaled_a10_v1'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    original=json.loads((ROOT/'protocols'/OLD/'FREEZE.json').read_text())
    for relative,digest in original['hashes'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Original freeze changed: '+relative)
    protocol=ROOT/'protocols'/NEW;configs=ROOT/'configs/reliability_scaled_a10_v1'
    protocol.mkdir(exist_ok=False);configs.mkdir(exist_ok=False)
    text=(ROOT/'protocols'/OLD/'PROTOCOL.md').read_text(encoding='utf8')
    text='# Hardware-only execution revision: NVIDIA A10\n\nThe 2026-10-17 first attempt received NVIDIA A10 and was rejected by the frozen A10G contract before inference. This new version requires actual NVIDIA A10. Algorithm, six recipes, lambda/delta/tau, reference/scoring pools and budgets are unchanged. Original A10G freeze remains immutable. No new-method outcome was inspected to make this change. Compare all recipes within this fixed actual hardware; never pool A10/A10G numerical runs silently.\n\n'+text
    (protocol/'PROTOCOL.md').write_text(text,encoding='utf8')
    registry=[]
    for r in json.loads((ROOT/'configs/reliability_scaled_v1/experiment_registry.json').read_text()):
        c=json.loads((ROOT/r['config']).read_text());c['required_gpu_name']='NVIDIA A10';c['prototype_protocol']=NEW
        path=configs/(r['run_id']+'.json');path.write_text(json.dumps(c,indent=2),encoding='utf8');registry.append(dict(run_id=r['run_id'],config=path.relative_to(ROOT).as_posix()))
    (configs/'experiment_registry.json').write_text(json.dumps(registry,indent=2),encoding='utf8')
    for old_name,new_name in [('modal_reliability_scaled.py','modal_reliability_scaled_a10.py'),('analyze_reliability_scaled.py','analyze_reliability_scaled_a10.py')]:
        source=(ROOT/'scripts'/old_name).read_text(encoding='utf8')
        source=source.replace('20261017_reliability_scaled_v1',NEW).replace('configs/reliability_scaled_v1','configs/reliability_scaled_a10_v1')
        if old_name.startswith('modal_'):
            source=source.replace("'isbi-gynsurg-reliability-scaled-v1'","'isbi-gynsurg-reliability-scaled-a10-v1'")
            source=source.replace("run_id='reliability_scaled_'","run_id='reliability_scaled_a10_'")
        (ROOT/'scripts'/new_name).write_text(source,encoding='utf8')
    paths=list((ROOT/'src').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+list(configs.glob('*.json'))+[protocol/'PROTOCOL.md']+[ROOT/'scripts'/n for n in ['modal_reliability_scaled_a10.py','analyze_reliability_scaled_a10.py','prepare_reliability_a10.py']]
    freeze=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_id=NEW,parent_protocol=OLD,parent_freeze_sha256=sha(ROOT/'protocols'/OLD/'FREEZE.json'),status='frozen_before_inference',actual_gpu_required='NVIDIA A10',hardware_only_revision=True,algorithm_and_parameters_unchanged=True,final_test_open=False,hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})
    (protocol/'FREEZE.json').write_text(json.dumps(freeze,indent=2),encoding='utf8')
    (protocol/'EXECUTION_AUTHORIZATION.json').write_text(json.dumps(dict(date_local='2026-10-17',source='User instruction: thực hiện; prior Modal/development authorization persists',scope='Frozen two-case engineering and fixed nine-source development, hardware-only A10 execution version',final_test_authorized=False),indent=2),encoding='utf8')
    print('Hardware-only A10 revision frozen; original A10G version preserved')

if __name__=='__main__':main()
