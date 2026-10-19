"""Scientific plots from accepted prototype outputs; no inference."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);args=a.parse_args();base=ROOT/'results'/args.run
    accepted=json.loads((base/'analysis/cohort_acceptance.json').read_text())
    if not accepted['complete']:raise ValueError('Unaccepted cohort')
    auc=pd.read_csv(base/'analysis/recipe_auc.csv',dtype={'clip_id':str});cases=auc.pivot(index='clip_id',columns='recipe',values='dice_auc')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained');x=np.arange(len(cases))
    for offset,(lhs,rhs,label) in enumerate([('scaled_fallback','uncertainty_raw','Prototype − uncertainty'),('uncertainty_fallback','uncertainty_raw','Fallback under entropy'),('scaled_raw','error_raw','Relative diversity vs error')]):
        axes[0].bar(x+(offset-1)*.25,cases[lhs]-cases[rhs],width=.24,label=label)
    axes[0].axhline(0,color='black',linewidth=.8);axes[0].set_xticks(x,cases.index);axes[0].set(xlabel='Development clip (one per source)',ylabel='Paired normalized Dice AUC delta',title='Source heterogeneity');axes[0].legend(fontsize=8)
    curves=[]
    for cid in cases.index:
        for recipe in cases.columns:
            frame=pd.read_csv(base/cid/'suite'/recipe/'per_clip.csv')
            b=frame.groupby('actual_density').dice.mean().reset_index();b['recipe']=recipe;b['clip_id']=cid;curves.append(b)
    curves=pd.concat(curves);curves.to_csv(base/'analysis/recipe_curves.csv',index=False)
    for recipe,group in curves.groupby('recipe'):
        b=group.groupby('actual_density').dice.mean();axes[1].plot(b.index,b.values,marker='o',label=recipe.replace('_',' '))
    axes[1].set(xlabel='Corrections / 100 raw frames',ylabel='Procedure-macro Dice',title='Same fixed score snapshots');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    stage=json.loads((base/'cohort_execution.json').read_text())['stage'];fig.suptitle(f'Exploratory {stage}: {len(cases)} previously observed sources; no final test')
    for ext in ['png','pdf','svg']:fig.savefig(base/'analysis'/f'prototype_comparison.{ext}',dpi=180)
    print('Rendered prototype_comparison PNG/PDF/SVG; CIs are in the paired endpoint JSON, not these overview curves')

if __name__=='__main__':main()
