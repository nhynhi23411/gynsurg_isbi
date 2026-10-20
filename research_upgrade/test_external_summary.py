import unittest
from research_upgrade.summarize_external_controls import source_statistics

class ExternalSummaryTests(unittest.TestCase):
    def test_sources_equal_not_frames_or_windows(self):
        rows=[]
        for source,count,value in [('one',1,.2),('two',9,.8)]:
            for i in range(count):
                for policy in ['uncertainty','uniform','random']:
                    for n in range(4):rows.append(dict(source_video=source,policy=policy,n_queries=n,dice=value,iou=value))
        r=source_statistics(rows,100)
        self.assertAlmostEqual(r['mean_auc']['uncertainty'],.5)
        self.assertTrue(all(c['delta_auc']==0 for c in r['contrasts']))
    def test_unavailable_not_silently_dropped(self):
        with self.assertRaises(ValueError):source_statistics([dict(source_video='one',policy='uncertainty',n_queries=0,dice=None,iou=None)],10)

if __name__=='__main__':unittest.main()
