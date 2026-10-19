"""Reuse unchanged predictions for a shared temporal exclusion, without GPU reruns."""
from pathlib import Path
import argparse,json,csv,sys,hashlib,shutil
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.scoring import common_scoring
def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--seconds',type=float,default=1);args=a.parse_args()
    source=ROOT/'results'/args.run;record=json.loads((source/'cohort_execution.json').read_text())
    if record['stage'] not in ['pilot','feasibility'] or record['final_test'] is not False:raise ValueError('Use a completed development suite')
    out=source/'sensitivity_1s'
    if out.exists():raise FileExistsError(out)
    out.mkdir();registry=json.loads((ROOT/'configs/official_sparse_v1/experiment_registry.json').read_text());evidence=[]
    for case in record['reports']:
        cid=case['clip_id'];queries=[];logs={}
        for recipe in registry:
            p=source/cid/'suite'/recipe['run_id']/'queries.csv'
            if not p.exists():raise ValueError('Missing compared query log')
            queries.extend(pd.read_csv(p).frame.tolist());logs[recipe['run_id']]=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
        common=common_scoring([74,99,124],queries,25,args.seconds);digest=hashlib.sha256(np.asarray(sorted(common),dtype='<i8').tobytes()).hexdigest()
        for recipe in registry:
            old=source/cid/'suite'/recipe['run_id'];dest=out/cid/'suite'/recipe['run_id'];dest.mkdir(parents=True)
            frames=pd.read_csv(old/'per_frame.csv',dtype={'clip_id':str,'procedure_id':str});frames=frames[frames.frame_index.isin(common)].copy();frames['scoring_frame_sha256']=digest
            metrics=pd.read_csv(old/'per_clip.csv',dtype={'clip_id':str,'procedure_id':str});updated=[]
            for _,row in metrics.iterrows():
                block=frames[(frames.density==row.density)&(frames.structure==row.structure)];present=block[block.gold_present.astype(str).str.lower()=='true']
                r=row.to_dict();r.update(dice=float(block.dice.mean()),iou=float(block.iou.mean()),n_scored=len(block),scoring_frame_sha256=digest,gold_present_frames=len(present),dice_present_only=float(present.dice.mean()) if len(present) else None,runtime_interpretation='Original propagation/query runtime reused; score-only temporal sensitivity')
                if len(block)!=len(common):raise ValueError('Incomplete shared frame subset')
                updated.append(r)
            frames.to_csv(dest/'per_frame.csv',index=False);pd.DataFrame(updated).to_csv(dest/'per_clip.csv',index=False);shutil.copy2(old/'queries.csv',dest/'queries.csv')
            prov=json.loads((old/'provenance.json').read_text());prov.update(derived_from=str(old.relative_to(ROOT)),temporal_exclusion_seconds=args.seconds,common_scoring_frames=common.tolist(),query_trajectory_unchanged=True,engine_not_rerun=True)
            (dest/'provenance.json').write_text(json.dumps(prov,indent=2))
        proof=json.loads((source/cid/'prepared/provenance.json').read_text());proof['metadata']['scoring_ids']=common.tolist();(out/cid/'input_provenance.json').write_text(json.dumps(proof,indent=2))
        evidence.append(dict(clip_id=cid,original_scoring_ids=[74,99,124],common_scoring_ids=common.tolist(),coverage=len(common)/3,union_query_frames=sorted(set(queries)),all_compared_query_log_hashes=logs))
    (out/'cohort_execution.json').write_text(json.dumps(dict(record,run_id=args.run+'/sensitivity_1s',derived_scoring_only=True,engine_rerun=False),indent=2))
    (out/'sensitivity_provenance.json').write_text(json.dumps(dict(seconds=args.seconds,source_run=args.run,common_pool_from_all_policies_and_seeds=True,query_selection_independent_of_scoring=True,cases=evidence),indent=2))
    print('Prepared shared temporal sensitivity for',len(evidence),'cases; unchanged queries, no new inference')
if __name__=='__main__':main()
