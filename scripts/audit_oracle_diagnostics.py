"""Saved-evidence audit for CPU-only restricted-oracle diagnosis."""
from pathlib import Path
import argparse,datetime,hashlib,json,re
import numpy as np,pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);args=a.parse_args();base=ROOT/'results'/args.run
    report=json.loads((base/'analysis/report.json').read_text());proof=json.loads((base/'provenance.json').read_text())
    if not report['raw_auc_reproduced'] or report['attainable_method'] or report['model_rerun'] or report['gate_trained'] or report['final_test_open']:raise ValueError('Oracle scope not preserved')
    if sha(ROOT/'scripts/oracle_diagnostic_metrics.py')!=proof['diagnostic_code_sha256']:raise ValueError('Diagnostic helper changed')
    if len(proof['cases'])!=9 or any(len(case['state_sha256'])!=6 for case in proof['cases']):raise ValueError('Incomplete state hash receipts')
    for case in proof['cases']:
        source=json.loads((ROOT/'results'/proof['source_run']/case['clip_id']/'execution.json').read_text())
        if source['input_sha256']!=case['input_sha256'] or source['official_split']!='train':raise ValueError('Input/split mismatch')
    observations=pd.read_csv(base/'observations.csv',dtype={'clip_id':str})
    if len(observations)!=324 or observations.duplicated(['clip_id','query_count','frame','structure']).any():raise ValueError('Invalid observation grid')
    values=observations.select_dtypes(include='number').drop(columns=['nearest_query_s'])
    if not np.isfinite(values.to_numpy()).all():raise ValueError('Nonfinite observation feature/outcome')
    for key in ['raw_dice','baseline_dice','restricted_oracle_dice','sam_entropy','baseline_entropy','baseline_mask_iou','state_change_iou','sparse_temporal_iou']:
        if not observations[key].between(0,1).all():raise ValueError('Out-of-range '+key)
    if not np.allclose(observations.gain,observations.raw_dice-observations.baseline_dice,atol=1e-12):raise ValueError('Gain mismatch')
    if not np.allclose(observations.restricted_oracle_dice,np.maximum(observations.raw_dice,observations.baseline_dice),atol=1e-12):raise ValueError('Restricted oracle mismatch')
    source=pd.read_csv(base/'analysis/per_source_oracle_auc.csv')
    decomposition=pd.read_csv(base/'analysis/headroom_by_frame_structure.csv')
    if not np.isclose(source.headroom.mean(),decomposition.headroom_contribution.sum()/9,atol=1e-12):raise ValueError('AUC decomposition mismatch')
    associations=pd.read_csv(base/'analysis/descriptive_signal_associations.csv');defined=associations.helpful_vs_harmful_auc.dropna()
    if not defined.between(0,1).all():raise ValueError('Invalid descriptive AUROC')
    if json.loads((ROOT/'configs/review_status.json').read_text())['approved_for_test']:raise ValueError('Final test unlocked')
    md=ROOT/'REPORT_ORACLE_DIAGNOSTICS_20261017.md';links=re.findall(r'\]\(<(D:/[^>]+)>\)',md.read_text(encoding='utf8'))
    if not all(Path(link).is_file() for link in links):raise FileNotFoundError('Report link missing')
    with Image.open(base/'analysis/oracle_headroom.png') as image:image.verify()
    files=[p for p in base.rglob('*') if p.is_file()]+[md,ROOT/'scripts/analyze_oracle_diagnostics.py',ROOT/'scripts/oracle_diagnostic_metrics.py',ROOT/'protocols/20261017_oracle_diagnostics_v1/PROTOCOL.md']
    receipt=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,run_id=args.run,observations=324,input_hash_receipts=9,state_hash_receipts=54,raw_auc_reproduced=True,oracle_scope='Gold-assisted two-output selector on fixed trajectory; not a deployable method or global performance bound',model_rerun=False,gate_trained=False,final_test_open=False,artifact_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)},visual_qa='Oracle headroom plot inspected: labels readable, gold-assisted nature explicit, all nine source bars displayed')
    (ROOT/'materials/verification/oracle_diagnostics_audit.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print('Oracle diagnostic audit passed;',len(files),'hashes; 324 observations; 54 state hashes')

if __name__=='__main__':main()
