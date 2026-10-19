"""Verify downloaded replay evidence and record actual curated visual review."""
from pathlib import Path
import argparse,datetime,hashlib,json,re
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);args=a.parse_args()
    base=ROOT/'results'/args.run;accepted=json.loads((base/'analysis/debug_acceptance.json').read_text())
    if not accepted['passed'] or len(accepted['checks'])!=4:raise ValueError('Replay not accepted')
    panels=list(base.glob('*/suite/*/debug/*.png'))
    if len(panels)!=60:raise ValueError('Expected 60 before/after panels')
    for path in panels:
        with Image.open(path) as img:img.verify()
    report=ROOT/'REPORT_DEBUG_REPLAY_20261017.md'
    links=re.findall(r'\]\(<(D:/[^>]+)>\)',report.read_text(encoding='utf8'))
    if not all(Path(link).is_file() for link in links):raise FileNotFoundError('Broken report image link')
    reviewed=[ROOT/p for p in accepted['curated_panels']]+[base/'analysis/correction_trajectories.png']
    files=[p for p in base.rglob('*') if p.is_file()]+[report]
    result=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,run_id=args.run,png_integrity_checked=len(panels),reviewed_images={p.relative_to(ROOT).as_posix():digest(p) for p in reviewed},visual_review='Curated report panels and plot were visually inspected: text and overlays readable, footer outside image; before/after/reference aligned. Clip059 later-frame uterus coverage decreases after first query; clip086 tools missing distal coverage after second query at frame124. Engineering observations only, no clinical review.',artifact_hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(files)},clinical_review=False,final_test_open=False)
    (ROOT/'materials/verification/debug_artifact_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('Debug artifact audit passed:',len(files),'hashes;',len(panels),'PNG files decode;',len(reviewed),'curated images reviewed')

if __name__=='__main__':main()
