import unittest,numpy as np
import test_core
from src.surgery import validate_sequence
from src.policy import choose
class InputRobustness(unittest.TestCase):
    def test_fractional_and_multidimensional_frame_ids_rejected(self):
        for ids in [np.array([.5,2]),np.array([[0,2]]),np.array([True,False])]:
            d=test_core.Core().data();d['query_ids']=ids
            with self.assertRaises(ValueError):validate_sequence(d)
    def test_nonbinary_gold_and_invalid_fps_rejected(self):
        for value in [np.nan,2,255]:
            d=test_core.Core().data();d['gold'][0,0,0,0]=value
            with self.assertRaises(ValueError):validate_sequence(d)
        for fps in [0,-1,np.nan]:
            d=test_core.Core().data();d['fps']=fps
            with self.assertRaises(ValueError):validate_sequence(d)
    def test_unknown_policy_and_invalid_scale_rejected(self):
        h=test_core.Core();d=h.data();cfg=h.cfg()
        with self.assertRaises(ValueError):choose(d['probabilities'],d['framewise'],[(0,0)],[],'ful',cfg,np.random.default_rng(1))
        with self.assertRaises(ValueError):choose(d['probabilities'],d['framewise'],[(0,0)],[],'full',dict(cfg,sigma=0),np.random.default_rng(1))
if __name__=='__main__':unittest.main()
