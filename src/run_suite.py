"""Run a matched policy suite for one reviewed clip with bounded shared caching.

Quality outputs use identical charged-prompt replay; runtimes are explicitly
amortized across the suite, never claimed as standalone per-policy speed.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from .common import approved,fresh_out,sha,write_csv
from .surgery import run,validate_sequence,validate_execution_gates
from .sam_engine import SAM2Engine
from .propagation_cache import PropagationCache
ROOT=Path(__file__).resolve().parents[1]

def main():
    a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--output',required=True)
    a.add_argument('--registry',default=str(ROOT/'configs/experiment_registry.json'))
    a.add_argument('--split',choices=['val','test'],default='val');a.add_argument('--cache-gib',type=float,default=10)
    a.add_argument('--save-final-probabilities',action='store_true');a.add_argument('--debug-trace',action='store_true');args=a.parse_args()
    approved(ROOT,test=args.split=='test')
    with np.load(args.input,allow_pickle=False) as loaded:d={k:loaded[k] for k in loaded.files}
    validate_execution_gates(d,args.split);validate_sequence(d)
    d['framewise']=d['probabilities'] # equality already checked; reduce memory duplication
    cache=PropagationCache(int(args.cache_gib*1024**3));predictor=None
    def factory(cfg,jpeg_dir,baseline):
        nonlocal predictor
        engine=SAM2Engine(cfg,jpeg_dir,baseline,cache=cache,predictor=predictor)
        if predictor is None:predictor=engine.predictor
        return engine
    recipes=json.loads(Path(args.registry).read_text());out=fresh_out(args.output);runs=[]
    # A single immutable propagation engine configuration is required for sharing.
    configs=[json.loads((ROOT/r['config']).read_text()) for r in recipes]
    keys=[(c['sam_config'],str(Path(c['sam_checkpoint']).resolve()),c['device'],c.get('sam_postprocessing',False),c.get('sam_precision','float32'),c.get('required_gpu_name')) for c in configs]
    if not keys or len(set(keys))!=1:raise ValueError('Suite propagation engine configuration differs')
    for recipe,cfg in zip(recipes,configs):
        before_hits,before_misses=cache.hits,cache.misses;start=time.perf_counter()
        folder=fresh_out(out/recipe['run_id'])
        from .debug_trace import DebugTrace
        trace=DebugTrace(folder/'debug',d,cfg) if args.debug_trace else None
        frame_rows=[];rows,queries,pred=run(d,cfg,factory,frame_sink=frame_rows.extend,trace_sink=trace)
        if trace is not None:trace.finish()
        for row in rows:row.update(propagation_cache_mode='shared_structure_prompt_set',runtime_interpretation='amortized_suite_execution')
        write_csv(folder/'per_clip.csv',rows)
        write_csv(folder/'per_frame.csv',frame_rows)
        if queries:write_csv(folder/'queries.csv',queries)
        if args.save_final_probabilities:np.savez_compressed(folder/'final_predictions.npz',probabilities=pred)
        record=dict(run_id=recipe['run_id'],config=cfg,seconds=time.perf_counter()-start,
            cache_hits=cache.hits-before_hits,cache_misses=cache.misses-before_misses,
            actual_annotation_budget_unchanged=True,runtime_interpretation='amortized_suite_execution')
        (folder/'provenance.json').write_text(json.dumps(record,indent=2));runs.append(record)
        print(recipe['run_id'],'complete; cache hits/misses',record['cache_hits'],record['cache_misses'],flush=True)
    report=dict(input_sha256=sha(args.input),registry_sha256=sha(args.registry),split=args.split,
        sam_checkpoint_sha256=sha(configs[0]['sam_checkpoint']),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').glob('*.py')},
        runs=runs,cache_max_bytes=cache.max_bytes,cache_retained_bytes=cache.bytes,
        runtime_note='Order-dependent warm/shared runtime. Use a separate uncached fixed-case benchmark for independent policy runtime comparisons.')
    import torch,platform
    report.update(torch=torch.__version__,cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,python=platform.python_version())
    (out/'suite_provenance.json').write_text(json.dumps(report,indent=2))
    import subprocess,sys
    (out/'environment_lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))

if __name__=='__main__':main()
