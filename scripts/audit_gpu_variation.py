"""Measure equivalent-selector numerical variation without asserting its cause."""
from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    first=ROOT/'results/official_sparse_feasibility_20261016T144620Z';second=ROOT/'results/official_sparse_tuning_20261016T151226Z';rows=[]
    for cid in ['001','038','059','086','105','116','138','146','160']:
        p1=json.loads((first/cid/'suite/suite_provenance.json').read_text());p2=json.loads((second/cid/'suite/suite_provenance.json').read_text())
        assert p1['input_sha256']==p2['input_sha256'] and p1['sam_checkpoint_sha256']==p2['sam_checkpoint_sha256']
        a=first/cid/'suite/error_seed20261016';b=second/cid/'suite/full_alpha0_beta0';one=pd.read_csv(a/'per_frame.csv');two=pd.read_csv(b/'per_frame.csv')
        q1=pd.read_csv(a/'queries.csv');q2=pd.read_csv(b/'queries.csv');same=q1.equals(q2)
        rows.append(dict(clip_id=cid,first_gpu=p1['gpu'],second_gpu=p2['gpu'],same_gpu_name=p1['gpu']==p2['gpu'],same_input_and_checkpoint=True,query_trajectory_identical=same,max_abs_frame_dice_difference=float(np.max(np.abs(one.dice-two.dice)))))
    report=dict(scope='Equivalent error selector versus full with alpha=beta=0 across independent jobs',cases=rows,
      all_query_trajectories_identical=all(r['query_trajectory_identical'] for r in rows),
      max_abs_frame_dice_difference=max(r['max_abs_frame_dice_difference'] for r in rows),
      same_named_gpu_cases_exact_in_this_check=all(r['max_abs_frame_dice_difference']==0 for r in rows if r['same_gpu_name']),
      observation='Nonzero differences in this sample coincide with NVIDIA A10 versus NVIDIA A10G labels. This check does not isolate all causal factors.',
      cross_hardware_bitwise_reproducibility_claimed=False,
      action='Optional required_gpu_name contract added; keep all compared policies on the same device within each clip suite; freeze hardware/runtime before any confirmatory test',
      official_reference='https://docs.pytorch.org/docs/2.7/notes/randomness.html')
    out=ROOT/'materials/verification/gpu_variant_numerical_audit.json';out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
if __name__=='__main__':main()
