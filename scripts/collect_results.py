import argparse,csv
from pathlib import Path
a=argparse.ArgumentParser();a.add_argument('--glob',required=True);a.add_argument('--output',required=True);args=a.parse_args();out=Path(args.output)
if out.exists():raise FileExistsError(out)
rows=[]
for f in sorted(Path('.').glob(args.glob)):
    for r in csv.DictReader(f.open(encoding='utf-8-sig')):r['run_source']=str(f);rows.append(r)
if not rows:raise ValueError('No results found')
with out.open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Collected',len(rows),'rows')
