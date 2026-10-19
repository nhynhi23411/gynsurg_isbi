"""Instantiate the alpha/beta grid already declared in the original plan."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def main():
    folder=ROOT/'configs/official_sparse_tuning_v1';folder.mkdir(exist_ok=True);base=json.loads((ROOT/'configs/official_sparse_v1/full_seed20261016.json').read_text());registry=[]
    for alpha in [0,.5,1,2]:
        for beta in [0,.25,.5]:
            run_id=('full_alpha'+str(alpha)+'_beta'+str(beta)).replace('.','p')
            cfg=dict(base,alpha=alpha,beta=beta);path=folder/(run_id+'.json');path.write_text(json.dumps(cfg,indent=2))
            registry.append(dict(run_id=run_id,config=path.relative_to(ROOT).as_posix()))
    (folder/'experiment_registry.json').write_text(json.dumps(registry,indent=2))
    (folder/'selection_rule.json').write_text(json.dumps(dict(alpha=[0,.5,1,2],beta=[0,.25,.5],sigma=.03,tau_s=2,cohort='all nine fixed official-train feasibility sources, one clip each',criterion='maximum mean procedure normalized Dice AUC; ties smaller alpha then smaller beta',report_global_best_and_best_both_nonzero=True,final_test_access=False,source='PLAN_RULES.md validation grid; instantiated after default feasibility, before this grid execution'),indent=2))
    print('Prepared 12 prespecified alpha/beta recipes; no final test')
if __name__=='__main__':main()
