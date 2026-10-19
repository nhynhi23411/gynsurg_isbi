import tempfile,unittest
from pathlib import Path
import numpy as np
import test_core
from src.policy import choose,scaled_candidate_scores
from src.reliability import trust_weight,ReliabilityFallback,wrap_engine
from src.surgery import run
from src.debug_trace import DebugTrace

class ReliabilityScaledTest(unittest.TestCase):
    def test_relative_penalty_bounded_and_nonnegative(self):
        rng=np.random.default_rng(7);p=rng.random((60,2,5,7));f=rng.random(p.shape)
        cfg=dict(relative_diversity=.25,tau_s=2,fps=25)
        rows=scaled_candidate_scores(p,f,[(t,k) for t in [0,24,49] for k in [0,1]],[(0,0)],cfg)
        for r in rows:
            self.assertGreaterEqual(r['score'],.75*r['error_proxy']-1e-15)
            self.assertLessEqual(r['score'],r['error_proxy'])
        self.assertEqual(len(rows),5)

    def test_zero_disagreement_and_sparse_raw_time_equivalence(self):
        rng=np.random.default_rng(2);p=rng.random((125,2,4,6)).astype('float32');p[0]=0
        cfg=dict(relative_diversity=.25,tau_s=2,fps=25);q=[(t,k) for t in [0,24,49] for k in [0,1]]
        full=scaled_candidate_scores(p,p,q,[(0,0)],cfg)
        ids=[0,24,49,74,99,124];sparse=scaled_candidate_scores(p[ids],p[ids],q,[(0,0)],cfg,frame_ids=ids)
        self.assertEqual(full,sparse);self.assertTrue(all(r['disagreement']==0 for r in full))

    def test_invalid_scale_and_unknown_guard_fail(self):
        p=np.full((60,2,3,3),.5)
        for value in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):scaled_candidate_scores(p,p,[(0,0)],[],dict(relative_diversity=value,tau_s=2,fps=25))
        with self.assertRaises(ValueError):wrap_engine(None,dict(propagation_reliability='typo'),p)
        with self.assertRaises(ValueError):trust_weight(p[0,0],p[0,0],0)

    def test_prediction_only_weight_bounds_and_confident_conflict(self):
        f=np.full((3,4),.01);s=np.full((3,4),.99)
        w,_=trust_weight(s,f,.25);self.assertLess(w,.05)
        uncertain,_=trust_weight(np.full((3,4),.5),f,.25);self.assertEqual(uncertain,0)
        empty,_=trust_weight(np.zeros((3,4)),np.zeros((3,4)),.25);self.assertTrue(0<=empty<=1)
        # No API accepting gold/reference labels for reliability computation.
        with self.assertRaises(TypeError):trust_weight(s,f,.25,gold=np.ones_like(s))

    def test_charged_masks_exact_unprompted_unchanged_raw_not_mutated(self):
        baseline=np.full((5,2,4,4),.01,dtype='float32')
        class Engine:
            def correct(self,t,k,mask):
                self.raw=np.full_like(baseline,.99);return self.raw
        engine=Engine();guard=ReliabilityFallback(engine,dict(reliability_delta=.25),baseline)
        first=np.zeros((4,4),bool);first[1:3,1:3]=True
        result=guard.correct(0,0,first)
        np.testing.assert_array_equal(result[0,0],first)
        np.testing.assert_array_equal(result[:,1],baseline[:,1])
        self.assertTrue((result[1:,0]<.5).all());self.assertTrue((engine.raw==.99).all())
        second=np.ones((4,4),bool);result=guard.correct(3,0,second)
        np.testing.assert_array_equal(result[0,0],first);np.testing.assert_array_equal(result[3,0],second)
        self.assertEqual(len(guard.prompts),2)

    def test_zero_budget_identity_and_debug_consistency(self):
        fixture=test_core.Core();d=fixture.data();cfg=fixture.cfg()
        cfg.update(policy='scaled_diversity_v1',relative_diversity=.25,propagation_reliability='confidence_fallback_v1',reliability_delta=.25)
        events=[];rows,q,p=run(d,cfg,test_core.FakeEngine,trace_sink=lambda e:events.append(e))
        np.testing.assert_array_equal(events[0]['probabilities'],d['framewise'][list(events[0]['frame_ids'])])
        with tempfile.TemporaryDirectory() as tmp:
            trace=DebugTrace(Path(tmp)/'debug',d,cfg);trace.panels=lambda *args:None
            r,q2,p2=run(d,cfg,test_core.FakeEngine,trace_sink=trace);trace.finish()
            self.assertEqual(q,q2);np.testing.assert_array_equal(p,p2)
            self.assertEqual(sum(x['selected'] for x in trace.decisions),4)
            self.assertTrue((Path(tmp)/'debug/reliability_weights.csv').is_file())

if __name__=='__main__':unittest.main()
