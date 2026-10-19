import unittest,tempfile
from pathlib import Path
import numpy as np
import test_core
from src.surgery import run
from src.debug_trace import DebugTrace

class DebugTraceTest(unittest.TestCase):
    def test_observation_does_not_change_selection_or_probabilities(self):
        fixture=test_core.Core();d=fixture.data();c=fixture.cfg()
        rows,q,p=run(d,c,test_core.FakeEngine)
        events=[]
        def sink(e):
            self.assertFalse(e['probabilities'].flags.writeable)
            events.append((e['stage'],e['query_count']))
        traced,queries,pred=run(d,c,test_core.FakeEngine,trace_sink=sink)
        self.assertEqual(q,queries);np.testing.assert_array_equal(p,pred)
        self.assertEqual([r['dice'] for r in rows],[r['dice'] for r in traced])
        self.assertEqual(events[0],('initial',0));self.assertEqual(len(events),9)

    def test_debug_scores_reconstruct_actual_choices_without_gold(self):
        fixture=test_core.Core();d=fixture.data();c=fixture.cfg()
        with tempfile.TemporaryDirectory() as tmp:
            trace=DebugTrace(Path(tmp)/'debug',d,c)
            trace.panels=lambda *args:None
            run(d,c,test_core.FakeEngine,trace_sink=trace);trace.finish()
            self.assertEqual(sum(r['selected'] for r in trace.decisions),4)
            self.assertTrue((Path(tmp)/'debug/state_04.npz').is_file())
