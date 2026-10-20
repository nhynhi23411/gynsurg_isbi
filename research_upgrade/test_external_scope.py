import unittest
import numpy as np
from research_upgrade.external_scope import decode_release_mask,valid_overlap,build_windows,QUERY,SCORE,AUC_WEIGHTS

class ExternalScopeTests(unittest.TestCase):
    def test_actual_decimal_codes_and_void(self):
        a=np.array([[31,32,50],[11,0,255]],np.uint8)
        fg,valid=decode_release_mask(np.repeat(a[:,:,None],3,axis=2))
        self.assertEqual(fg.sum(),2);self.assertEqual(valid.sum(),4)
        self.assertFalse(fg[0,2])
        with self.assertRaises(ValueError):decode_release_mask(np.array([[49]],np.uint8))
    def test_void_is_excluded_not_negative(self):
        p=np.array([[1,1,0]],bool);r=np.array([[1,0,0]],bool);v=np.array([[1,0,1]],bool)
        self.assertEqual(valid_overlap(p,r,v)['dice'],1.)
        self.assertIsNone(valid_overlap(p,r,np.zeros_like(v))['dice'])
    def test_no_visualization_as_reference(self):
        with self.assertRaises(ValueError):decode_release_mask(np.array([[[31,32,31]]],np.uint8))
    def test_numeric_order_contiguity_and_disjoint_pools(self):
        rows=[dict(source_video='video01',window='video01/clip',source_frame_index=str(i),image_member=f'video01/clip/{i}.png',mask_member=f'video01/clip/{i}_mask.png') for i in reversed(range(80,160))]
        w=build_windows(rows)[0];self.assertEqual(w['query_source_indices'],[80,95,111]);self.assertFalse(set(QUERY)&set(SCORE))
        self.assertAlmostEqual(sum(AUC_WEIGHTS),1.)
        with self.assertRaises(ValueError):build_windows(rows[:-1]+[rows[0]])

if __name__=='__main__':unittest.main()
