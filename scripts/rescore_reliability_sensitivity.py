"""Shared 1-second score exclusion for frozen v2 outputs, no model rerun."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.scoring import common_scoring

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);args=a.parse_args();source=ROOT/'results'/args.run
    record=json.loads((source/'cohort_execution.json').read_text())
    if record['stage']!='development' or record['final_test']:raise ValueError('Development-only sensitivity')
    out=source/'sensitivity_1s';out.mkdir(exist_ok=False)
    registry=json.loads((ROOT/'configs/reliability_scaled_a10_v1/experiment_registry.json').read_text());evidence=[]
    for case in record['reports']:
        cid=case['clip_id'];queries=[];hashes={}
        for recipe in registry:
            path=source/cid/'suite'/recipe['run_id']/'queries.csv';queries.extend(pd.read_csv(path).frame.tolist())
            with path.open('rb') as f:hashes[recipe['run_id']]=hashlib.file_digest(f,'sha256').hexdigest()
        common=common_scoring([74,99,124],queries,25,1);digest=hashlib.sha256(np.asarray(sorted(common),dtype='<i8').tobytes()).hexdigest()
        for recipe in registry:
            old=source/cid/'suite'/recipe['run_id'];dest=out/cid/'suite'/recipe['run_id'];dest.mkdir(parents=True)
            frames=pd.read_csv(old/'per_frame.csv',dtype={'clip_id':str,'procedure_id':str});frames=frames[frames.frame_index.isin(common)].copy();frames['scoring_frame_sha256']=digest
            metrics=pd.read_csv(old/'per_clip.csv',dtype={'clip_id':str,'procedure_id':str});updated=[]
            for _,row in metrics.iterrows():
                block=frames[(frames.density==row.density)&(frames.structure==row.structure)]
                if len(block)!=len(common):raise ValueError('Incomplete shared scoring subset')
                present=block[block.gold_present.astype(str).str.lower()=='true'];r=row.to_dict()
                r.update(dice=float(block.dice.mean()),iou=float(block.iou.mean()),n_scored=len(block),scoring_frame_sha256=digest,gold_present_frames=len(present),dice_present_only=float(present.dice.mean()) if len(present) else None,runtime_interpretation='Original inference/query runtime; score-only common temporal sensitivity')
                updated.append(r)
            frames.to_csv(dest/'per_frame.csv',index=False);pd.DataFrame(updated).to_csv(dest/'per_clip.csv',index=False);shutil.copy2(old/'queries.csv',dest/'queries.csv')
            prov=json.loads((old/'provenance.json').read_text());prov.update(derived_from=old.relative_to(ROOT).as_posix(),common_scoring_frames=common.tolist(),temporal_exclusion_seconds=1,query_trajectory_unchanged=True,engine_rerun=False)
            (dest/'provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8')
        proof=json.loads((ROOT/'results'/record['source_run']/cid/'prepared/provenance.json').read_text());proof['metadata']['scoring_ids']=common.tolist()
        (out/cid/'input_provenance.json').write_text(json.dumps(proof,indent=2),encoding='utf8')
        evidence.append(dict(clip_id=cid,common_scoring_frames=common.tolist(),coverage=len(common)/3,query_hashes=hashes,union_query_frames=sorted(set(queries))))
    (out/'cohort_execution.json').write_text(json.dumps(dict(record,run_id=args.run+'/sensitivity_1s',derived_scoring_only=True,engine_rerun=False),indent=2),encoding='utf8')
    (out/'sensitivity_provenance.json').write_text(json.dumps(dict(common_pool_from_all_recipes=True,query_selection_independent_of_scoring=True,cases=evidence),indent=2),encoding='utf8')
    print('Shared sensitivity prepared:',len(evidence),'sources; no inference rerun')

if __name__=='__main__':main()
