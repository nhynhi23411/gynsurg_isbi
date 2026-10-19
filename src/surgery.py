import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from .common import approved,fresh_out,overlap,write_csv,sha
from .policy import choose
from .sam_engine import SAM2Engine
from .reference_scope import reference_scope_ready
ROOT=Path(__file__).resolve().parents[1]
def validate_sequence(d):
    p=d['probabilities'];g=d['gold'];f=d['framewise'];q=np.asarray(d['query_ids']);s=np.asarray(d['scoring_ids'])
    if any(ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) for ids in [q,s]):raise ValueError('Frame IDs must be one-dimensional integers')
    q=q.astype(int);s=s.astype(int)
    if p.ndim!=4 or p.shape!=g.shape or p.shape!=f.shape:raise ValueError('Expected matching T,K,H,W')
    if not p.size or not np.isfinite(float(d['fps'])) or float(d['fps'])<=0:raise ValueError('Empty sequence or invalid fps')
    if g.dtype!=np.bool_ and (not np.isfinite(g).all() or ((g!=0)&(g!=1)).any()):raise ValueError('Gold references must be binary 0/1')
    if not np.isfinite(p).all() or not np.isfinite(f).all() or p.min()<0 or p.max()>1 or f.min()<0 or f.max()>1:raise ValueError('Invalid probabilities')
    if set(q)&set(s):raise ValueError('Query and scoring frames overlap')
    if not len(q) or not len(s) or min(q.min(),s.min())<0 or max(q.max(),s.max())>=len(p):raise ValueError('Invalid query/scoring pool')
    if len(set(q))!=len(q) or len(set(s))!=len(s):raise ValueError('Duplicate frame IDs')
    if not np.allclose(p,f):raise ValueError('Budget zero must start from framewise predictions with no free gold prompt')
    return p,g,f,q,s
def run(d,c,engine_factory=SAM2Engine,frame_sink=None,trace_sink=None):
    p,g,f,qi,si=validate_sequence(d);p=p.copy();c=dict(c);c['fps']=float(d['fps']);rng=np.random.default_rng(c['seed']);queries=[];query_log=[];rows=[];engine=engine_factory(c,str(d['jpeg_dir']),p.copy());candidates=[(int(t),k) for t in qi for k in range(p.shape[1])]
    from .reliability import wrap_engine
    engine=wrap_engine(engine,c,p)
    requested=[int(np.floor(x*len(p)/100)) for x in c['densities']]
    if not requested or requested!=sorted(requested) or requested[0]!=0:raise ValueError('Invalid budget grid')
    budgets=[min(x,len(candidates)) for x in requested]
    trace_ids=np.asarray(sorted(set(map(int,qi))|set(map(int,si))),dtype=int)
    def emit(stage,selected=None):
        if trace_sink is not None:
            snapshot=p[trace_ids].copy();snapshot.setflags(write=False)
            trace_sink(dict(stage=stage,query_count=len(queries),queries=tuple(queries),selected=selected,frame_ids=tuple(map(int,trace_ids)),probabilities=snapshot,reliability=tuple(dict(r) for r in getattr(engine,'last_reliability_summary',[]))))
    emit('initial')
    for density,budget in zip(c['densities'],budgets):
        start=time.perf_counter();selection_seconds=0.;propagation_seconds=0.
        while len(queries)<budget:
            select_start=time.perf_counter();t,k=choose(p,f,candidates,queries,c['policy'],c,rng);selection_seconds+=time.perf_counter()-select_start
            emit('decision',(t,k))
            # Oracle mask is accessed only AFTER selection; queries cannot be scoring frames.
            prop_start=time.perf_counter();p=engine.correct(t,k,g[t,k]);propagation_seconds+=time.perf_counter()-prop_start;queries.append((t,k));query_log.append(dict(query=len(queries),frame=t,structure=k,mask_sha256=hashlib.sha256(np.asarray(g[t,k],dtype=np.uint8).tobytes()).hexdigest()))
            emit('after',(t,k))
        scored=[int(t) for t in si if all(abs(t-a)>c['exclusion_seconds']*c['fps'] for a,_ in queries)]
        if not scored:raise ValueError('No scoring frames after temporal exclusion')
        scoring_hash=hashlib.sha256(np.asarray(sorted(scored),dtype='<i8').tobytes()).hexdigest()
        frame_rows=[]
        gold_distances={}
        # Reference geometry is computed only in evaluation, never passed to choose().
        from scipy.ndimage import distance_transform_edt
        for t in scored:
            organ=np.any(g[t,c['organ_ids']],axis=0);tool=np.any(g[t,c['instrument_ids']],axis=0)
            gold_distances[t]=(float(distance_transform_edt(~tool)[organ].min()/np.hypot(*organ.shape)) if organ.any() and tool.any() else None)
        for k in range(p.shape[1]):
            vals=[overlap(p[t,k]>.5,g[t,k]) for t in scored]
            for t,v in zip(scored,vals):
                same=[abs(t-a)/c['fps'] for a,j in queries if j==k]
                frame_rows.append(dict(procedure_id=str(d['procedure_id']),clip_id=str(d['clip_id']),policy=c['policy'],seed=c['seed'],density=density,actual_density=100*len(queries)/len(p),n_queries=len(queries),structure=k,frame_index=t,timestamp_s=t/c['fps'],dice=v['dice'],iou=v['iou'],gold_present=bool(g[t,k].any()),prediction_present=bool((p[t,k]>.5).any()),nearest_same_structure_query_s=min(same) if same else None,gold_proximity_normalized=gold_distances[t],scoring_frame_sha256=scoring_hash))
            present=[v['dice'] for t,v in zip(scored,vals) if g[t,k].any()]
            threshold=c.get('failure_threshold')
            rows.append(dict(procedure_id=str(d['procedure_id']),clip_id=str(d['clip_id']),policy=c['policy'],seed=c['seed'],density=density,actual_density=100*len(queries)/len(p),n_queries=len(queries),structure=k,dice=float(np.mean([v['dice'] for v in vals])),iou=float(np.mean([v['iou'] for v in vals])),dice_present_only=float(np.mean(present)) if present else None,gold_present_frames=len(present),failure_rate=float(np.mean([v['dice']<threshold for v in vals])) if threshold is not None else None,n_scored=len(scored),update_runtime_s=time.perf_counter()-start))
        for row in rows[-p.shape[1]:]:
            row.update(scoring_frame_sha256=scoring_hash,budget_capped=requested[len(rows)//p.shape[1]-1]>budget,selection_and_feature_seconds=selection_seconds,propagation_seconds=propagation_seconds)
        if frame_sink is not None:frame_sink(frame_rows)
    return rows,query_log,p
def validate_execution_gates(d,split):
    if str(d['split'])!=split or str(d['group_verified'])!='true' or str(d['exposure_clean'])!='true':raise ValueError('Sequence grouping/split/exposure gate failed')
    if not reference_scope_ready(d) or any(key not in d or str(d[key])!=value for key,value in [('human_annotation_complete','true'),('permission_status','verified')]):raise ValueError('Sequence reference/scope/permission gate failed; regenerate from reviewed metadata')

def main():
    a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--config',required=True);a.add_argument('--output',required=True);a.add_argument('--split',choices=['val','test'],default='val');a.add_argument('--debug-trace',action='store_true');args=a.parse_args()
    approved(ROOT,test=args.split=='test');c=json.loads(Path(args.config).read_text());d=np.load(args.input,allow_pickle=False)
    validate_execution_gates(d,args.split)
    out=fresh_out(args.output);frame_rows=[]
    from .debug_trace import DebugTrace
    trace=DebugTrace(out/'debug',d,c) if args.debug_trace else None
    rows,queries,p=run(d,c,frame_sink=frame_rows.extend,trace_sink=trace);write_csv(out/'per_clip.csv',rows);write_csv(out/'per_frame.csv',frame_rows)
    if trace is not None:trace.finish()
    if queries:write_csv(out/'queries.csv',queries)
    import subprocess,sys,platform,datetime
    np.savez_compressed(out/'final_predictions.npz',probabilities=p)
    (out/'environment_lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (out/'provenance.json').write_text(json.dumps(dict(config=c,input_sha256=sha(args.input),sam_checkpoint_sha256=sha(c['sam_checkpoint']),split=args.split,python=platform.python_version(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes={str(x.relative_to(ROOT)):sha(x) for x in (ROOT/'src').glob('*.py')}),indent=2))
if __name__=='__main__':main()
