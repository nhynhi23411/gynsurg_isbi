"""CPU-only reproduction from released numeric summaries; no medical media."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import paired_bootstrap
base=ROOT/'benchmarks/train_expansion_20261017_v1'
clips=pd.read_csv(base/'per_clip_auc.csv',dtype={'clip_id':str})
report=json.loads((base/'report.json').read_text())
source=clips.groupby('procedure_id')[['raw_dice_auc','baseline_dice_auc']].mean()
contrast=paired_bootstrap(source.raw_dice_auc.to_numpy(),source.baseline_dice_auc.to_numpy(),seed=20261017)
assert len(clips)==170 and len(source)==9
np.testing.assert_allclose(contrast['delta'],report['raw_minus_budget_zero']['delta'],rtol=0,atol=1e-12)
np.testing.assert_allclose(contrast['ci95'],report['raw_minus_budget_zero']['ci95'],rtol=0,atol=1e-12)
print(json.dumps(dict(clips=len(clips),source_groups=len(source),mean_raw_auc=float(source.raw_dice_auc.mean()),mean_baseline_auc=float(source.baseline_dice_auc.mean()),paired_source_bootstrap=contrast,independent_validation=False,final_test_open=False),indent=2))
