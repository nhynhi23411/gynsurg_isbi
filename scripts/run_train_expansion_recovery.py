"""Local driver: preserve original Python3.12 analysis site from Store Python3.11.

No change to the frozen serialized GPU worker or scientific inputs. The Store
Python redirects child user-site discovery unless its original base is explicit.
"""
from pathlib import Path
import json,os,subprocess,sys

ANALYSIS='C:/Users/pc/.venvs/isbi_gynsurg/Scripts/python.exe'
original_run=subprocess.run

def run_with_original_analysis_site(command,*args,**kwargs):
    if isinstance(command,(list,tuple)) and command and str(command[0]).replace('\\','/').lower()==ANALYSIS.lower():
        environment=dict(kwargs.get('env') or os.environ)
        environment['PYTHONUSERBASE']='C:/Users/pc/AppData/Roaming/Python'
        kwargs['env']=environment
    return original_run(command,*args,**kwargs)

def main():
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Use py -3.11')
    subprocess.run=run_with_original_analysis_site
    probe="import sys,site,json,numpy,pandas,scipy;print(json.dumps(dict(python=sys.version,site=site.getusersitepackages(),numpy=numpy.__version__,pandas=pandas.__version__,scipy=scipy.__version__)))"
    result=subprocess.run([ANALYSIS,'-c',probe],check=True,capture_output=True,text=True)
    record=json.loads(result.stdout)
    expected='C:/Users/pc/AppData/Roaming/Python/Python312/site-packages'
    if record['site'].replace('\\','/').lower()!=expected.lower():raise ValueError('Wrong analysis environment')
    out=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1/recovery_20261019')
    out.mkdir(parents=True,exist_ok=True)
    (out/'local_analysis_environment.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print('ORIGINAL_ANALYSIS_ENVIRONMENT_PASS',record,flush=True)
    if '--check-env' in sys.argv:return
    from modal_train_expansion_recovery import main as execute_main
    import modal
    with modal.enable_output():execute_main()

if __name__=='__main__':main()
