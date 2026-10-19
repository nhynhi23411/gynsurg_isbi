import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_train_expansion_recovery as driver

class AnalysisEnvironmentTest(unittest.TestCase):
    def test_analysis_subprocess_gets_original_site_without_mutating_environment(self):
        original={'EXAMPLE':'value'}
        with patch.object(driver,'original_run',return_value=7) as mocked:
            self.assertEqual(driver.run_with_original_analysis_site([driver.ANALYSIS,'-c','pass'],env=original),7)
        sent=mocked.call_args.kwargs['env']
        self.assertEqual(sent['PYTHONUSERBASE'],'C:/Users/pc/AppData/Roaming/Python')
        self.assertEqual(original,{'EXAMPLE':'value'})

    def test_modal_cli_python_site_is_not_changed(self):
        with patch.object(driver,'original_run',return_value=0) as mocked:
            driver.run_with_original_analysis_site(['python3.11','-m','modal','app','list'])
        self.assertNotIn('env',mocked.call_args.kwargs)
