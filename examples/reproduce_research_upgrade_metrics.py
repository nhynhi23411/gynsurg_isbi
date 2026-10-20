"""Verify published source effects/AUC from aggregate CSVs; no data download or GPU."""
from pathlib import Path
import numpy as np
import pandas as pd
from research_upgrade.acceptance_v2 import bootstrap

ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'benchmarks/research_upgrade_20261020'
    source=pd.read_csv(folder/'all_methods_source.csv')
    reported=pd.read_csv(folder/'all_methods_auc.csv').set_index('method')
    budgets=pd.read_csv(folder/'all_methods_budget.csv')
    raw=source[source.method=='uncertainty'].set_index('procedure_id').dice_auc
    for method,row in reported.iterrows():
        values=source[source.method==method].set_index('procedure_id').dice_auc.reindex(raw.index)
        if len(values)!=9 or values.isna().any():raise ValueError('All nine sources required')
        differences=values-raw;interval=bootstrap(differences.to_numpy())
        curve=budgets[budgets.method==method].set_index('n_queries').dice.reindex([0,1,2,5])
        expected=[values.mean(),differences.mean(),*interval['ci95']]
        published=[row.dice_auc,row.delta,row.ci_low,row.ci_high]
        np.testing.assert_allclose(expected,published,rtol=0,atol=1e-12)
        np.testing.assert_allclose(np.dot(curve,[.1,.2,.4,.3]),row.dice_auc,rtol=0,atol=1e-12)
        if int((differences>0).sum())!=int(row.positive_sources):raise ValueError('Positive-source count mismatch')
        print(f'{method}: AUC={values.mean():.9f}, delta={differences.mean():+.9f}, interval={interval["ci95"]}')
    curve=budgets[budgets.method=='uncertainty'].set_index('n_queries').dice
    first=(curve.loc[1]-curve.loc[0])/(curve.loc[5]-curve.loc[0])
    print(f'All seven methods reproduced from aggregate sources/budgets; first-mask gain fraction={first:.6f}')

if __name__=='__main__':main()
