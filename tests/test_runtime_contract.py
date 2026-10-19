import unittest
from unittest.mock import patch
from src.runtime_contract import validate_hardware
class RuntimeContract(unittest.TestCase):
    def test_explicit_gpu_name_is_enforced(self):
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.get_device_name',return_value='mock GPU A'):
            self.assertEqual(validate_hardware(dict(device='cuda',required_gpu_name='mock GPU A'))['gpu'],'mock GPU A')
            with self.assertRaises(RuntimeError):validate_hardware(dict(device='cuda',required_gpu_name='mock GPU B'))
    def test_cpu_cannot_satisfy_gpu_contract(self):
        with self.assertRaises(RuntimeError):validate_hardware(dict(device='cpu',required_gpu_name='mock GPU A'))
        with self.assertRaises(ValueError):validate_hardware(dict(device='cpu',required_gpu_name=123))
if __name__=='__main__':unittest.main()
