"""Publication-exportable descriptive figures from accepted full-training metrics."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1/analysis')

def main():
    report=json.loads((BASE/'report.json').read_text())
    if not report['complete'] or report['clips']!=170 or report['source_groups']!=9 or report['final_test_open']:raise ValueError('Complete training cohort required')
    curve=pd.read_csv(BASE/'per_source_curves.csv')
    auc=pd.read_csv(BASE/'per_source_auc.csv').sort_values('procedure_id')
    table=curve.pivot(index='procedure_id',columns='n_queries',values='dice').sort_index().reindex(columns=[0,1,2,5])
    if table.shape!=(9,4) or table.isna().any().any():raise ValueError('Incomplete source curves')
    if list(table.index)!=list(auc.procedure_id):raise ValueError('Wrong source order')
    np.testing.assert_allclose(table.to_numpy()@np.array([.1,.2,.4,.3]),auc.raw_dice_auc,rtol=0,atol=1e-12)
    rng=np.random.default_rng(20261017);indices=rng.integers(0,9,size=(2000,9))
    boot=table.to_numpy()[indices].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975],axis=0)
    x=np.array([0,.8,1.6,4]);mean=table.mean(axis=0).to_numpy()
    baseline=float(table[0].mean())
    plt.rcParams.update({'font.size':8.5,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(7.16,3.4),layout='constrained')
    axes[0].fill_between(x,lo,hi,color='#2369a4',alpha=.16,label='Descriptive source-bootstrap 95%')
    axes[0].plot(x,mean,'o-',color='#2369a4',label='Raw SAM + uncertainty queries')
    axes[0].axhline(baseline,color='#666666',linestyle='--',label='Frozen framewise, no correction')
    axes[0].set(xlabel='Charged full masks per 100 raw frames',ylabel='Source-macro Dice',ylim=(0,1.02));axes[0].set_xticks(x);axes[0].legend(fontsize=8)
    delta=auc.raw_dice_auc-auc.baseline_dice_auc
    axes[1].bar(np.arange(9),delta,color=np.where(delta>=0,'#2369a4','#bd4141'))
    axes[1].axhline(0,color='black',linewidth=.7)
    axes[1].set_xticks(np.arange(9),[p.rsplit('_',1)[-1] for p in auc.procedure_id]);axes[1].set(xlabel='Verified source video',ylabel='Raw − no-correction normalized Dice AUC')
    fig.suptitle('170 official-training clips / 9 previously observed sources; no final test',fontsize=9)
    for ext in ['png','pdf','svg']:fig.savefig(BASE/f'train_expansion_summary.{ext}',dpi=300)
    plt.close(fig)
    pd.DataFrame(dict(actual_density=x,mean_dice=mean,ci95_low=lo,ci95_high=hi,baseline_dice=baseline)).to_csv(BASE/'plot_curve_values.csv',index=False)
    (BASE/'plot_notes.json').write_text(json.dumps(dict(pointwise_descriptive_source_bootstrap=True,source_resamples=2000,seed=20261017,clip_averaged_within_source=True,independent_validation=False,final_test_open=False),indent=2),encoding='utf8')
    print('Saved PNG/PDF/SVG from accepted9-source metrics')

if __name__=='__main__':main()
