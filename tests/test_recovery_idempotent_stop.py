import importlib.util,subprocess,unittest,tempfile
from pathlib import Path
spec=importlib.util.spec_from_file_location('resume',Path(__file__).resolve().parents[1]/'scripts/resume_train_expansion_recovery.py')
resume=importlib.util.module_from_spec(spec);spec.loader.exec_module(resume)

class StopTests(unittest.TestCase):
    def test_explicit_already_stopped_and_unknown_error(self):
        command=['python','-m','modal','app','stop','ap-own','--yes']
        with tempfile.TemporaryDirectory() as directory:
            previous=resume.BASE;resume.BASE=Path(directory)
            try:
                def stopped(*args,**kwargs):return subprocess.CompletedProcess(command,1,'App is already stopped.','')
                self.assertEqual(resume.safe_run(stopped,command,check=True).returncode,1)
                def error(*args,**kwargs):return subprocess.CompletedProcess(command,1,'','Authentication failed')
                with self.assertRaises(subprocess.CalledProcessError):resume.safe_run(error,command,check=True)
            finally:resume.BASE=previous

    def test_other_command_retains_check(self):
        def original(command,**kwargs):
            self.assertEqual(command,['python','analysis.py']);self.assertTrue(kwargs['check']);return 'untouched'
        self.assertEqual(resume.safe_run(original,['python','analysis.py'],check=True),'untouched')
