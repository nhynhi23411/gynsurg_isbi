import types
import unittest
from research_upgrade.precision_controls_v5 import apply_matched_flags, pair_factor


class PrecisionControlsTests(unittest.TestCase):
    def test_both_arms_disable_tf32_without_altering_other_flags(self):
        matmul = types.SimpleNamespace(allow_tf32=True)
        cudnn = types.SimpleNamespace(allow_tf32=True, deterministic=False)
        fake = types.SimpleNamespace(backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=matmul), cudnn=cudnn))
        apply_matched_flags(fake)
        self.assertFalse(matmul.allow_tf32)
        self.assertFalse(cudnn.allow_tf32)
        self.assertFalse(cudnn.deterministic)

    def test_precision_contrast_is_within_registered_pair(self):
        a = dict(clip_id='132', pair_id='p0', precision='bfloat16', repeat=0)
        self.assertEqual(pair_factor(a, dict(a, precision='float32')), 'within_device_precision')
        self.assertEqual(pair_factor(a, dict(a, pair_id='p1', repeat=1)), 'same_precision_repeat')
        self.assertIsNone(pair_factor(a, dict(a, pair_id='p1', repeat=1, precision='float32')))
        self.assertIsNone(pair_factor(a, dict(a, clip_id='148')))
