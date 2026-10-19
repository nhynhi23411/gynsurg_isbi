import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from src.common import paired_bootstrap
from src.result_validation import validate_metric_table
def main():
    a=argparse.ArgumentParser();a.add_argument('--kind',choices=['ultrasound','surgery'],required=True);a.add_argument('--proposed',required=True);a.add_argument('--baseline',required=True);a.add_argument('--output',required=True);a.add_argument('--expected-seeds',nargs='+',type=int);args=a.parse_args()
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(args.proposed);b=pd.read_csv(args.baseline)
    if args.kind=='ultrasound':
        p=p[p.postprocess=='largest'];b=b[b.postprocess=='largest']
        pairs=p.merge(b,on='image_id',suffixes=('_p','_b'),validate='one_to_one')
        if len(pairs)!=len(p) or len(pairs)!=len(b):raise ValueError('Unmatched test cases')
        if not (pairs.group_id_p==pairs.group_id_b).all():raise ValueError('Different grouping')
        eligible=pairs.dropna(subset=['nae_dmax_p','nae_dmax_b'])
        g=eligible.groupby('group_id_p')[['nae_dmax_p','nae_dmax_b']].mean()
        # Cluster resampling, but retain image-weighted mean endpoint by resampling all views.
        blocks=[x[['nae_dmax_p','nae_dmax_b']].to_numpy() for _,x in eligible.groupby('group_id_p')]
        if not blocks:raise ValueError('No positive-reference geometry cases')
        rng=np.random.default_rng(20261016);delta=(eligible.nae_dmax_p-eligible.nae_dmax_b).mean();boot=[]
        for _ in range(2000):
            arr=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))]);boot.append(np.mean(arr[:,0]-arr[:,1]))
        result=dict(endpoint='image_mean_normalized_Dmax_error',delta_proposed_minus_baseline=float(delta),ci95=np.quantile(boot,[.025,.975]).tolist(),groups=len(blocks),images=len(eligible),grouping_verified=bool(eligible.group_verified_p.astype(str).str.lower().eq('true').all()),excluded_undefined_geometry=len(pairs)-len(eligible))
        pairs.to_csv(out/'paired_cases.csv',index=False)
        fig,ax=plt.subplots(1,2,figsize=(9,4));ax[0].scatter(eligible.dice_p,eligible.nae_dmax_p,s=10,alpha=.5);ax[0].set(xlabel='Dice proposed',ylabel='Normalized Dmax error');ax[1].boxplot([eligible.nae_dmax_b,eligible.nae_dmax_p],tick_labels=['Baseline','Width loss']);ax[1].set_ylabel('Normalized Dmax error')
    else:
        if p.policy.nunique()!=1 or b.policy.nunique()!=1:raise ValueError('Each comparison input must contain exactly one policy')
        validate_metric_table(p,expected_seeds=args.expected_seeds);validate_metric_table(b,expected_seeds=args.expected_seeds)
        keys=['procedure_id','clip_id','density','structure']
        for df in (p,b):
            if 'scoring_frame_sha256' not in df:raise ValueError('Missing scoring frame identity')
            if (df.groupby(keys).scoring_frame_sha256.nunique()!=1).any():raise ValueError('Different scoring frames across seeds')
        ps=p[keys+['scoring_frame_sha256']].drop_duplicates()
        bs=b[keys+['scoring_frame_sha256']].drop_duplicates()
        matched=ps.merge(bs,on=keys,suffixes=('_p','_b'),validate='one_to_one')
        if len(matched)!=len(ps) or len(matched)!=len(bs) or not (matched.scoring_frame_sha256_p==matched.scoring_frame_sha256_b).all():raise ValueError('Paired comparison requires identical scoring frames')
        def aggregate(df):
            # Equal classes within clips, equal clips within procedure, average random seeds.
            clip=df.groupby(['procedure_id','clip_id','seed','density']).agg(dice=('dice','mean'),actual_density=('actual_density','first')).reset_index()
            proc=clip.groupby(['procedure_id','seed','density']).agg(dice=('dice','mean'),actual_density=('actual_density','mean')).reset_index()
            curve=proc.groupby(['procedure_id','density']).agg(dice=('dice','mean'),actual_density=('actual_density','mean')).reset_index()
            values={}
            for pid,x in curve.groupby('procedure_id'):
                x=x.sort_values('actual_density');xx=x.actual_density.to_numpy();yy=x.dice.to_numpy()
                if len(xx)<2 or xx[-1]<=xx[0]:raise ValueError('AUC needs at least two distinct actual densities')
                values[pid]=float(trapezoid(yy,xx)/(xx[-1]-xx[0]))
            return pd.Series(values),curve
        pp,pc=aggregate(p);bb,bc=aggregate(b)
        if set(pp.index)!=set(bb.index):raise ValueError('Unmatched procedures')
        for pid in pp.index:
            if not np.array_equal(pc[pc.procedure_id==pid].actual_density,bc[bc.procedure_id==pid].actual_density):raise ValueError('Unmatched budgets')
        bb=bb.reindex(pp.index);result=paired_bootstrap(pp.to_numpy(),bb.to_numpy());result['endpoint']='procedure_macro_Dice_normalized_AUC';result['direction']='positive favors proposed'
        pd.DataFrame({'proposed_auc':pp,'baseline_auc':bb}).to_csv(out/'per_procedure_auc.csv')
        fig,ax=plt.subplots(figsize=(6,4))
        for label,df in [('Proposed',pc),('Baseline',bc)]:
            xx=[];yy=[];lo=[];hi=[];rng=np.random.default_rng(20261016)
            for density,block in df.groupby('density'):
                values=block.dice.to_numpy();boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
                xx.append(block.actual_density.mean());yy.append(values.mean());ci=np.quantile(boot,[.025,.975]);lo.append(ci[0]);hi.append(ci[1])
            line=ax.plot(xx,yy,marker='o',label=label)[0];ax.fill_between(xx,lo,hi,alpha=.15,color=line.get_color())
        ax.set(xlabel='Corrections per 100 frames',ylabel='Procedure macro Dice');ax.legend()
    (out/'statistics.json').write_text(json.dumps(result,indent=2));fig.tight_layout();fig.savefig(out/'results.png',dpi=200);fig.savefig(out/'results.pdf');plt.close(fig)
if __name__=='__main__':main()
