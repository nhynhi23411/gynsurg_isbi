import unittest
import numpy as np
from src.source_identity import refine_window,identity_decision

class SourceIdentityTest(unittest.TestCase):
    def test_temporal_offset_recovered_with_compression_noise(self):
        rng=np.random.default_rng(42)
        source=rng.integers(10,245,size=(180,4,6,3),dtype=np.uint8)
        offsets=np.array([0,24,49,74,99,124]);start=31
        query=np.clip(source[start+offsets].astype(int)+rng.integers(-2,3,size=(6,4,6,3)),0,255).astype(np.uint8)
        error,recovered,per_frame=refine_window(source,query)
        self.assertEqual(recovered,start);self.assertLess(error,.01);self.assertEqual(len(per_frame),6)

    def test_source_tie_and_missing_alternate_fail(self):
        a=dict(source_video_id='01',six_frame_rgb_mae=.001,max_frame_rgb_mae=.002,per_frame_rgb_correlation=[.999]*6)
        self.assertFalse(identity_decision([a])['pixel_identity_pass'])
        self.assertFalse(identity_decision([a,dict(a,source_video_id='02')])['pixel_identity_pass'])
        b=dict(source_video_id='02',six_frame_rgb_mae=.04,max_frame_rgb_mae=.05)
        self.assertTrue(identity_decision([b,a])['pixel_identity_pass'])
        a['per_frame_rgb_correlation']=[.8]*6
        self.assertFalse(identity_decision([b,a])['pixel_identity_pass'])

    def test_wrong_encoding_and_truncated_window_rejected(self):
        with self.assertRaises(ValueError):refine_window(np.zeros((124,4,6,3),np.uint8),np.zeros((6,4,6,3),np.uint8))
        with self.assertRaises(ValueError):refine_window(np.zeros((180,4,6,3)),np.zeros((6,4,6,3)))

if __name__=='__main__':unittest.main()
