"""Audit all declared cases/recipes plus per-frame output before collection."""
from pathlib import Path
import argparse,json,csv,sys,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.result_validation import validate_metric_table
def main():
    a=argparse.ArgumentParser();a.add_argument('--results-root',required=True);a.add_argument('--cohort',required=True);a.add_argument('--registry',default=str(ROOT/'configs/experiment_registry.json'));a.add_argument('--output',required=True);args=a.parse_args()
    cohort=list(csv.DictReader(open(args.cohort,encoding='utf-8-sig')));recipes=json.loads(Path(args.registry).read_text())
    if not cohort or len({r['clip_id'] for r in cohort})!=len(cohort):raise ValueError('Empty/duplicate cohort')
    checks=[]
    for case in cohort:
        for recipe in recipes:
            case_folder=Path(args.results_root)/case['clip_id'];folder=(case_folder/'suite' if (case_folder/'suite').exists() else case_folder)/recipe['run_id'];cfg=json.loads((ROOT/recipe['config']).read_text())
            df=pd.read_csv(folder/'per_clip.csv',dtype={'clip_id':str,'procedure_id':str})
            validate_metric_table(df,densities=tuple(cfg['densities']),expected_seeds=[cfg['seed']],cohort={(case['clip_id'],case['procedure_id'])})
            if set(df.policy)!={cfg['policy']}:raise ValueError('Recipe policy mismatch')
            prepared=case_folder/'prepared/provenance.json'
            if not prepared.exists():prepared=case_folder/'input_provenance.json'
            if prepared.exists():
                meta=json.loads(prepared.read_text())['metadata'];n_frames=meta['frame_count'];cap=len(meta['query_ids'])*meta['n_structures']
                expected_counts={b:min(int(b*n_frames//100),cap) for b in cfg['densities']}
                if not (df.n_queries==df.density.map(expected_counts)).all():raise ValueError('Charged count differs from declared budget')
                if not ((df.actual_density-100*df.n_queries/n_frames).abs()<1e-8).all():raise ValueError('Incorrect actual density')
            frames=pd.read_csv(folder/'per_frame.csv',dtype={'clip_id':str,'procedure_id':str})
            if set(frames.clip_id)!={case['clip_id']} or set(frames.procedure_id)!={case['procedure_id']}:raise ValueError('Frame cohort mismatch')
            if frames.scoring_frame_sha256.nunique()!=1 or frames.scoring_frame_sha256.iloc[0]!=df.scoring_frame_sha256.iloc[0]:raise ValueError('Frame scoring identity mismatch')
            if not frames.dice.between(0,1).all() or not frames.iou.between(0,1).all():raise ValueError('Invalid frame metrics')
            key=['policy','seed','density','structure']
            if frames.duplicated(key+['frame_index']).any():raise ValueError('Duplicate frame metrics')
            import numpy as np
            for _,block in frames.groupby(key):
                digest=hashlib.sha256(np.asarray(sorted(block.frame_index),dtype='<i8').tobytes()).hexdigest()
                if digest!=block.scoring_frame_sha256.iloc[0]:raise ValueError('Frame IDs do not match scoring hash')
            means=frames.groupby(key)[['dice','iou']].mean().reset_index().merge(df,on=key,suffixes=('_frames','_clip'),validate='one_to_one')
            if len(means)!=len(df) or any((means[k+'_frames']-means[k+'_clip']).abs().max()>1e-8 for k in ['dice','iou']):raise ValueError('Frame aggregate mismatch')
            counts=frames.groupby(key).size().to_numpy()
            if not (counts==df.sort_values(key).n_scored.to_numpy()).all():raise ValueError('Frame count mismatch')
            if not (folder/'provenance.json').is_file():raise ValueError('Missing provenance')
            if json.loads((folder/'provenance.json').read_text())['config']!=cfg:raise ValueError('Run config differs from declared recipe')
            checks.append(dict(clip_id=case['clip_id'],run_id=recipe['run_id'],metric_rows=len(df),frame_rows=len(frames)))
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(dict(complete=True,checks=checks,registry_sha256=hashlib.file_digest(Path(args.registry).open('rb'),'sha256').hexdigest(),cohort_sha256=hashlib.file_digest(Path(args.cohort).open('rb'),'sha256').hexdigest()),indent=2))
    print('Complete cohort/recipe/frame acceptance:',len(checks),'case-runs')
if __name__=='__main__':main()
