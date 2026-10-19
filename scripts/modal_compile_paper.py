"""Compile/render the clearly labelled protocol draft, not a submission or result."""
from pathlib import Path
import datetime,json
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('isbi-gynsurg-paper-draft-v1',tags={'project':'isbi-gynsurg'})
volume=modal.Volume.from_name('isbi-gynsurg-results-v1',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.11')
       .apt_install('texlive-latex-base','texlive-fonts-recommended','poppler-utils')
       .add_local_dir(str(ROOT/'paper'),'/opt/paper',ignore=['__pycache__','*.png','*.log','*.aux','*.bbl','*.blg']))

@app.function(image=image,cpu=1,memory=2048,timeout=600,max_containers=1,volumes={'/results':volume})
def compile_draft(run_id:str):
    import subprocess,shutil,hashlib
    out=Path('/results')/run_id;shutil.copytree('/opt/paper',out)
    sources={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.iterdir() if p.suffix in ['.tex','.bib','.sty','.bst']}
    log=[]
    commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['bibtex','main'],
              ['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
              ['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex']]
    for cmd in commands:
        r=subprocess.run(cmd,cwd=out,capture_output=True,text=True);log.append(r.stdout+'\n'+r.stderr)
        if r.returncode:
            (out/'compile_stdout.txt').write_text('\n'.join(log));volume.commit()
            return dict(success=False,stage=cmd[0],returncode=r.returncode,diagnostics=log[-1][-5000:])
    (out/'compile_stdout.txt').write_text('\n'.join(log))
    info=subprocess.check_output(['pdfinfo',str(out/'main.pdf')],text=True)
    subprocess.run(['pdftoppm','-scale-to','1400','-png',str(out/'main.pdf'),str(out/'page')],check=True)
    report=dict(success=True,run_id=run_id,draft_only=True,submission_ready=False,primary_results_available=False,
        source_hashes=sources,pdf_sha256=hashlib.file_digest((out/'main.pdf').open('rb'),'sha256').hexdigest(),
        pdf_bytes=(out/'main.pdf').stat().st_size,pdfinfo=info,
        rendered_pages=[p.name for p in sorted(out.glob('page-*.png'))],
        tex_version=subprocess.check_output(['pdflatex','--version'],text=True).splitlines()[0])
    (out/'compilation_report.json').write_text(json.dumps(report,indent=2));volume.commit();return report

@app.local_entrypoint()
def main():
    run_id='paper_protocol_draft_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report=compile_draft.remote(run_id);out=ROOT/'results'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'compilation_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    names=['compile_stdout.txt']+(['main.pdf','main.log']+report['rendered_pages'] if report['success'] else [])
    for name in names:
        (out/name).write_bytes(b''.join(volume.read_file(run_id+'/'+name)))
    print(json.dumps(report,indent=2))
