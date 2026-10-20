"""Reproduce complete source-equal external descriptive results from numeric CSVs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from research_upgrade.summarize_external_controls import source_statistics

def main():
    folder=Path(__file__).resolve().parents[1]/'benchmarks/research_upgrade_external_20261020'
    rows=pd.read_csv(folder/'window_curves.csv').to_dict('records')
    actual=source_statistics(rows); reported=json.loads((folder/'reported_summary.json').read_text())
    assert actual['sources']==reported['sources']==17
    for p in ['uncertainty','uniform','random']:
        np.testing.assert_allclose(actual['mean_auc'][p],reported['mean_auc'][p],rtol=0,atol=1e-12)
    for a,b in zip(actual['contrasts'],reported['contrasts']):
        for k in ['delta_auc','lower95','upper95','positive_sources']:
            np.testing.assert_allclose(a[k],b[k],rtol=0,atol=1e-12)
    print('All101windows/17sources: AUC, paired bootstrap intervals and positive-source counts reproduced at1e-12. Descriptive replication on a previously explored release, not pristine validation.')

if __name__=='__main__':main()
