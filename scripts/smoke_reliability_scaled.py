"""Synthetic CPU wiring smoke for six configs; explicitly no SAM/clinical result."""
from pathlib import Path
import datetime,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.surgery import run
from src.debug_trace import DebugTrace
from src.common import write_csv

class ToyEngine:
    def __init__(self,cfg,jpeg_dir,baseline):self.baseline=baseline.copy();self.prompted=set()
    def correct(self,t,k,mask):
        self.prompted.add(k);raw=self.baseline.copy()
        for structure in self.prompted:raw[:,structure]=.95
        raw[t,k]=mask
        return raw

def main():
    out=ROOT/'results'/('synthetic_reliability_scaled_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
    rng=np.random.default_rng(20261017);p=rng.uniform(.02,.2,(125,2,8,12)).astype('float32');gold=np.zeros_like(p,dtype=bool);gold[:,:,2:6,3:9]=True
    d=dict(probabilities=p,framewise=p.copy(),gold=gold,query_ids=np.array([0,24,49]),scoring_ids=np.array([74,99,124]),fps=25,jpeg_dir='not_used_synthetic',procedure_id='SYNTHETIC_ONLY',clip_id='SYNTHETIC_ONLY')
    reports=[]
    for recipe in json.loads((ROOT/'configs/reliability_scaled_v1/experiment_registry.json').read_text()):
        cfg=json.loads((ROOT/recipe['config']).read_text());folder=out/recipe['run_id'];folder.mkdir();frames=[]
        trace=DebugTrace(folder/'debug',d,cfg);trace.panels=lambda *args:None
        rows,queries,pred=run(d,cfg,ToyEngine,frame_sink=frames.extend,trace_sink=trace);trace.finish()
        write_csv(folder/'per_clip.csv',rows);write_csv(folder/'per_frame.csv',frames);write_csv(folder/'queries.csv',queries)
        if len(rows)!=8 or len(frames)!=24 or len(queries)!=5:raise ValueError('Synthetic wiring incomplete')
        if not np.isfinite(pred).all() or pred.min()<0 or pred.max()>1:raise ValueError('Invalid blended output')
        reports.append(dict(recipe=recipe['run_id'],aggregate_rows=len(rows),frame_rows=len(frames),charged_queries=len(queries),debug_score_reconstruction=True))
    report=dict(passed=True,synthetic_only=True,foundation_model_executed=False,gpu_executed=False,clinical_accuracy_claim=False,recipes=reports)
    (out/'smoke_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (ROOT/'materials/verification/reliability_scaled_cpu_smoke.json').write_text(json.dumps(dict(output=out.relative_to(ROOT).as_posix(),**report),indent=2),encoding='utf8')
    print(out);print('Six recipes passed synthetic CPU wiring; no GPU or clinical inference')

if __name__=='__main__':main()
