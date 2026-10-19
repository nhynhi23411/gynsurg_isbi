import importlib.util,unittest,sys,tempfile,zipfile,hashlib
from pathlib import Path
spec=importlib.util.spec_from_file_location('transport',Path(__file__).resolve().parents[1]/'scripts/compact_train_expansion_transport.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)

class TransportTests(unittest.TestCase):
    def test_archive_paths_restricted_to_known_clips_metrics(self):
        self.assertTrue(transport.valid_member('001/suite/uncertainty_raw/per_frame.csv',{'001'}))
        for name in ['../001/a.json','001/../../a.json','/001/a.json','001\\a.json','002/a.json','001/final_predictions.npz']:
            self.assertFalse(transport.valid_member(name,{'001'}),name)

    def test_hash_mismatch_blocks_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'metrics.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('001/execution.json',b'{}')
            manifest={'archive_sha256':transport.sha(archive),'sha256':{'001/execution.json':'0'*64}}
            with self.assertRaisesRegex(ValueError,'Member hash'):transport.extract_verified(archive,manifest,root/'stage',{'001'})
            self.assertFalse((root/'stage/001/execution.json').exists())
