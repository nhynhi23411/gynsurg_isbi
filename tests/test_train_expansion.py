import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_train_expansion import source_summary
class TrainExpansionTest(unittest.TestCase):
    def test_source_weight_not_clip_weight(self):
        rows=[dict(procedure_id=p,raw_dice_auc=x,baseline_dice_auc=0,raw_iou_auc=x,baseline_iou_auc=0) for p,x in [('a',1.),('a',1.),('a',1.),('b',0.)]]
        table=pd.DataFrame(rows);source=source_summary(table)
        self.assertEqual(source.raw_dice_auc.mean(),.5)
        self.assertEqual(table.raw_dice_auc.mean(),.75)
