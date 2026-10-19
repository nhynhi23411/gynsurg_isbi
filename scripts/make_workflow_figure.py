"""Reproducible protocol diagram; no simulated performance or patient image."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'paper/figures';out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,2.6));ax.set(xlim=(0,10),ylim=(0,2.6));ax.axis('off')
    labels=['Frozen framewise\nprobabilities','Choose frame + target\nfrom predictions only','Read selected mask\ncharge one correction','Fixed SAM2\npropagation','Evaluate fixed\ndisjoint score frames']
    for i,label in enumerate(labels):
        x=.08+i*2
        ax.add_patch(FancyBboxPatch((x,.75),1.75,.9,boxstyle='round,pad=0.035',facecolor='#edf2f6',edgecolor='#273746',linewidth=1))
        ax.text(x+.875,1.2,label,ha='center',va='center',fontsize=9)
        if i<4:ax.add_patch(FancyArrowPatch((x+1.8,1.2),(x+1.97,1.2),arrowstyle='-|>',mutation_scale=10,color='#273746'))
    ax.text(5,2.35,'Official released references: query and score pools fixed before policy runs',ha='center',fontsize=10)
    for x in [4.95,8.95]:ax.add_patch(FancyArrowPatch((x,2.1),(x,1.7),arrowstyle='-|>',mutation_scale=10,color='#273746',linestyle='--'))
    ax.plot([6.95,6.95,2.95],[.72,.42,.42],color='#273746',linewidth=1)
    ax.add_patch(FancyArrowPatch((2.95,.42),(2.95,.72),arrowstyle='-|>',mutation_scale=10,color='#273746'))
    ax.text(5,.46,'Updated predictions; repeat to budget',ha='center',va='bottom',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',pad=1))
    ax.text(5,.14,'Budget zero: framewise only. All policies share models, correction units and scoring frames.',ha='center',fontsize=9)
    fig.tight_layout(pad=.2)
    for suffix in ['pdf','svg','png']:fig.savefig(out/('workflow_official_sparse_v1.'+suffix),dpi=220,bbox_inches='tight')
    plt.close(fig);print('Created protocol workflow figure in',out)
if __name__=='__main__':main()
