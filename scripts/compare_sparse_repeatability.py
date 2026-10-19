"""Compare primary query trajectories and scored metrics across independent jobs."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    a=argparse.ArgumentParser();a.add_argument('--first',required=True);a.add_argument('--second',required=True);args=a.parse_args()
    first=ROOT/'results'/args.first;second=ROOT/'results'/args.second
    reports=[]
    for run in ['full_seed20261016','uncertainty_seed20261016']:
        f=first/'058/suite'/run;s=second/'058/suite'/run
        q1=pd.read_csv(f/'queries.csv');q2=pd.read_csv(s/'queries.csv')
        queries_equal=q1.equals(q2)
        one=pd.read_csv(f/'per_frame.csv');two=pd.read_csv(s/'per_frame.csv')
        keys=['density','structure','frame_index'];one=one.sort_values(keys).reset_index(drop=True);two=two.sort_values(keys).reset_index(drop=True)
        if len(one)!=len(two) or not one[keys].equals(two[keys]):raise ValueError('Frame alignment mismatch across repeats')
        difference=float(np.max(np.abs(one[['dice','iou']].to_numpy()-two[['dice','iou']].to_numpy())))
        reports.append(dict(run_id=run,query_trajectory_exact=queries_equal,metric_max_abs_difference=difference,metric_equal_within_1e_7=difference<=1e-7,scored_rows=len(one)))
    report=dict(first=args.first,second=args.second,scope='Independent-job repeat of selected queries and scored Dice/IoU; full probability tensors were not retained for this check',
       passed=all(r['query_trajectory_exact'] and r['metric_equal_within_1e_7'] for r in reports),checks=reports,
       selector_optimization_between_runs=True,selector_equivalence_unit_test='test_policy_subset_equivalence.py',final_test=False)
    (second/'repeatability_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(2)
if __name__=='__main__':main()
