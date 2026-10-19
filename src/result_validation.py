"""Require complete finite metric grids before any scientific aggregation."""
import itertools
import numpy as np

def validate_metric_table(df,densities=(0,1,2,4),structures=(0,1),expected_seeds=None,cohort=None):
    required=['procedure_id','clip_id','policy','seed','density','actual_density','n_queries','structure','dice','iou','n_scored','scoring_frame_sha256']
    if any(k not in df for k in required) or df.empty:raise ValueError('Incomplete metric schema or empty table')
    keys=['procedure_id','clip_id','policy','seed','density','structure']
    if df.duplicated(keys).any():raise ValueError('Duplicate metric rows')
    for k in ['density','actual_density','n_queries','dice','iou','n_scored']:
        if not np.isfinite(df[k].to_numpy(dtype=float)).all():raise ValueError('Nonfinite '+k)
    if not df.dice.between(0,1).all() or not df.iou.between(0,1).all():raise ValueError('Metrics out of range')
    if (df.n_queries<0).any() or (df.n_scored<=0).any() or not np.equal(df.n_queries,np.floor(df.n_queries)).all():raise ValueError('Invalid counts')
    if df.groupby('clip_id').procedure_id.nunique().max()!=1:raise ValueError('Clip assigned to multiple procedures')
    observed=set(zip(df.clip_id.astype(str),df.procedure_id.astype(str)))
    if cohort is not None and observed!=set(cohort):raise ValueError('Cohort differs from declared cases')
    for policy,block in df.groupby('policy'):
        seeds=set(expected_seeds) if expected_seeds is not None else set(block.seed)
        expected=set(itertools.product(seeds,densities,structures))
        for _,clip in block.groupby(['procedure_id','clip_id']):
            if set(zip(clip.seed,clip.density,clip.structure))!=expected:raise ValueError('Incomplete seed/density/structure grid: '+str(policy))
            if clip.scoring_frame_sha256.nunique()!=1 or clip.n_scored.nunique()!=1:raise ValueError('Scoring set differs across budgets/seeds')
            for _,point in clip.groupby(['seed','density']):
                if point.actual_density.nunique()!=1 or point.n_queries.nunique()!=1:raise ValueError('Structure budget mismatch')
    return dict(rows=len(df),clips=df.clip_id.nunique(),procedures=df.procedure_id.nunique())
