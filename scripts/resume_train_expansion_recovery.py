"""Resume frozen driver; tolerate only Modal's explicit already-stopped reply."""
from pathlib import Path
import datetime,json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('C:/Users/pc/ISBI_GynSurg_runs/train_expansion_20261017_v1/recovery_20261019')

def safe_run(original,command,*args,**kwargs):
    is_stop=(isinstance(command,(list,tuple)) and len(command)==7
             and list(command[1:5])==['-m','modal','app','stop']
             and str(command[5]).startswith('ap-') and command[6]=='--yes')
    if not is_stop:return original(command,*args,**kwargs)
    request=dict(kwargs);check=request.pop('check',False)
    result=original(command,*args,**request,capture_output=True,text=True)
    output=(result.stdout or '')+(result.stderr or '')
    print(output,flush=True)
    stopped='App is already stopped.' in output
    entry=dict(command=list(command),returncode=result.returncode,already_stopped=stopped,output=output)
    path=BASE/'idempotent_stop_history.json'
    history=json.loads(path.read_text()) if path.exists() else []
    history.append(entry);path.write_text(json.dumps(history,indent=2),encoding='utf8')
    if result.returncode and not stopped and check:
        raise subprocess.CalledProcessError(result.returncode,command,output=result.stdout,stderr=result.stderr)
    return result

def main():
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Use py -3.11')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive=BASE/('resume_history_'+stamp);archive.mkdir()
    for name in ['calls.json','attempts.json','active_app.json','runtime_preflight.json','pilot_passed.json','local_analysis_environment.json']:
        if (BASE/name).exists():shutil.copy2(BASE/name,archive/name)
    import run_train_expansion_recovery as driver
    original=driver.original_run
    driver.original_run=lambda command,*args,**kwargs:safe_run(original,command,*args,**kwargs)
    driver.main()

if __name__=='__main__':main()
