"""Build one sensitivity score set from the union of ALL compared query logs.

Keep exclusion_seconds=0 in rerun configs: exclusion is already fixed in the NPZ.
"""
from pathlib import Path
import argparse,csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.common import approved,sha
from src.scoring import common_scoring
from src.surgery import validate_sequence
ROOT=Path(__file__).resolve().parents[1]
def main():
    a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--queries',nargs='+',required=True);a.add_argument('--seconds',type=float,default=1.);a.add_argument('--output',required=True);args=a.parse_args()
    with np.load(args.input,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    approved(ROOT,test=str(d['split'])=='test');validate_sequence(d)
    if str(d['group_verified'])!='true' or str(d['exposure_clean'])!='true':raise ValueError('Unresolved source gates')
    out=Path(args.output)
    if out.exists() or out.resolve().is_relative_to((ROOT/'data/raw').resolve()):raise ValueError('Use a new derived/output path')
    q=[]
    for p in args.queries:
        rows=list(csv.DictReader(open(p,encoding='utf-8-sig')))
        for r in rows:
            t,k=int(r['frame']),int(r['structure'])
            if t not in d['query_ids'] or not 0<=k<d['gold'].shape[1]:raise ValueError('Query log does not match input sequence')
            q.append(t)
    old=d['scoring_ids'].copy();d['scoring_ids']=common_scoring(old,q,float(d['fps']),args.seconds)
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**d)
    out.with_suffix('.scoring_provenance.json').write_text(json.dumps(dict(input_sha256=sha(args.input),query_sources={p:sha(p) for p in args.queries},seconds=args.seconds,original_scoring_ids=old.tolist(),common_scoring_ids=d['scoring_ids'].tolist(),union_query_frames=sorted(set(q)),rerun_config_exclusion_seconds=0,note='Include every compared policy and seed; no per-policy changing score set'),indent=2),encoding='utf8')
if __name__=='__main__':main()
