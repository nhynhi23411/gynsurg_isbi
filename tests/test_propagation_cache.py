import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from src.propagation_cache import PropagationCache
from src.sam_engine import SAM2Engine

class FakePredictor:
    def __init__(self):self.propagations=0
    def init_state(self,**kwargs):return {}
    def add_new_mask(self,state,frame_idx,obj_id,mask):state[frame_idx]=np.array(mask)
    def propagate_in_video(self,state,**kwargs):
        self.propagations+=1;mask=np.mean(list(state.values()),axis=0)
        for t in range(6):yield t,[1],torch.tensor(mask*4-2)[None,None]

class CacheTest(unittest.TestCase):
    def test_precision_rejected_before_model_loading(self):
        baseline=np.zeros((6,2,4,5),np.float16)
        for precision in ['float16','bfloat16']:
            with self.assertRaises(ValueError):SAM2Engine(dict(device='cpu',sam_precision=precision),'unused',baseline,predictor=FakePredictor())

    def test_lru_bound_and_immutable_copies(self):
        c=PropagationCache(8);x=np.array([1,2],np.float32);c.put('a',x);x[0]=10
        self.assertEqual(c.get('a')[0],1)
        with self.assertRaises(ValueError):c.get('a')[0]=2
        c.put('b',x);self.assertIsNone(c.get('a'));self.assertLessEqual(c.bytes,8)

    def test_cached_replay_matches_and_does_not_leak_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'mock.pt';checkpoint.write_bytes(b'mock-only')
            cfg=dict(sam_checkpoint=str(checkpoint),sam_config='mock',device='cpu')
            baseline=np.full((6,2,4,5),.2,np.float16);cache=PropagationCache(100000)
            predictor=FakePredictor();mask=np.zeros((4,5),bool);mask[1:3]=True
            for t in range(6):(Path(tmp)/f'{t:05d}.jpg').write_bytes(b'mock-only JPEG bytes for fake predictor')
            cold=SAM2Engine(cfg,tmp,baseline,cache,predictor);a=cold.correct(0,0,mask)
            self.assertEqual(predictor.propagations,2)
            warm=SAM2Engine(cfg,tmp,baseline,cache,predictor);b=warm.correct(0,0,mask)
            self.assertTrue(np.array_equal(a,b));self.assertEqual(predictor.propagations,2)
            self.assertTrue(np.array_equal(b[:,1],baseline[:,1]))
            # Additional charged prompt changes the key, forcing new propagation.
            warm.correct(2,0,~mask);self.assertEqual(predictor.propagations,4)
            # Baseline and clip contexts cannot reuse another context's results.
            other=SAM2Engine(cfg,tmp,baseline+.1,cache,predictor);other.correct(0,0,mask)
            self.assertEqual(predictor.propagations,6)
            (Path(tmp)/'00000.jpg').write_bytes(b'changed mock pixels')
            changed=SAM2Engine(cfg,tmp,baseline,cache,predictor);changed.correct(0,0,mask)
            self.assertEqual(predictor.propagations,8)

    def test_frame_count_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):SAM2Engine(dict(device='cpu'),tmp,np.zeros((6,2,4,5)),predictor=FakePredictor())

if __name__=='__main__':unittest.main()
