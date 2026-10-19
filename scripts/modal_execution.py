"""Versioned ISBI jobs. Existing dataset volume is read only by convention.

Run with MODAL_PROFILE set in the invoking process. No old launcher is imported.
The first stage is technical/synthetic; it never evaluates a clinical cohort.
"""
from pathlib import Path
import datetime,json
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('isbi-gynsurg-execution-v1',tags={'project':'isbi-gynsurg'})
data_volume=modal.Volume.from_name('gynsurg',create_if_missing=False)
results_volume=modal.Volume.from_name('isbi-gynsurg-results-v1',create_if_missing=True)
sam=ROOT/'materials/code/extracted/sam2/sam2-main'
image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.7.1','torchvision==0.22.1','numpy==1.26.4','scipy==1.15.2','pandas==2.2.3','matplotlib==3.10.1','Pillow==11.2.1','transformers==4.53.3','hydra-core==1.3.2','omegaconf==2.3.0','iopath==0.1.10','opencv-python-headless==4.11.0.86','tqdm==4.67.1')
    .env({'SAM2_BUILD_CUDA':'0','PYTHONPATH':'/opt/isbi','PYTHONHASHSEED':'20261016','HF_HUB_OFFLINE':'1'})
    .add_local_dir(str(sam),'/opt/sam2',copy=True,ignore=['.git','__pycache__'])
    .run_commands('pip install --no-deps --no-build-isolation /opt/sam2')
    .add_local_dir(str(ROOT/'src'),'/opt/isbi/src',ignore=['__pycache__'])
    .add_local_dir(str(ROOT/'configs'),'/opt/isbi/configs')
    .add_local_file(str(ROOT/'materials/checkpoints/sam2.1_hiera_tiny.pt'),'/opt/isbi/materials/checkpoints/sam2.1_hiera_tiny.pt')
    .add_local_file(str(ROOT/'materials/checkpoints/dinov2-base_config.json'),'/opt/isbi/materials/checkpoints/dinov2-base_config.json')
    .add_local_file(str(ROOT/'materials/local/followup/baseline/phase53_source.py'),'/opt/isbi/materials/local/followup/baseline/phase53_source.py')
    .add_local_file(str(ROOT/'materials/local/followup/baseline/baseline_final.pth'),'/opt/isbi/materials/local/followup/baseline/baseline_final.pth'))
for name in ['001001','058001','086001']:
    image=image.add_local_file(str(ROOT/f'data/raw/AutoLaparo_Task3/imgs/{name}.jpg'),f'/opt/development/{name}.jpg')
    image=image.add_local_file(str(ROOT/f'data/raw/AutoLaparo_Task3/masks/{name}.png'),f'/opt/development/{name}.png')

@app.function(image=image,gpu='A10G',cpu=4,memory=24576,timeout=1800,max_containers=1,volumes={'/datasets':data_volume,'/results':results_volume})
def technical_smoke(run_id:str,matched_scope:bool=False):
    import os,sys,time,hashlib,subprocess,platform
    import numpy as np
    import torch
    from PIL import Image
    from src.sam_engine import SAM2Engine
    os.chdir('/opt/isbi');out=Path('/results')/run_id;out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(20261016);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    # Saved source snapshot and full environment are attached to every run.
    checks={str(p.relative_to('/opt/isbi')):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in Path('/opt/isbi/src').glob('*.py')}
    (out/'source_hashes.json').write_text(json.dumps(checks,indent=2))
    (out/'environment_lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    frames=out/'synthetic_frames';frames.mkdir();baseline=np.full((12,2,256,256),.1,np.float32)
    yy,xx=np.mgrid[:256,:256]
    for t in range(12):
        rgb=np.zeros((256,256,3),np.uint8);rgb[:]=[35,25,30];rgb[(xx-90-t)**2+(yy-100)**2<35**2]=[175,75,90];rgb[170:190,120+t:220+t]=[150,155,150]
        Image.fromarray(rgb).save(frames/f'{t:05d}.jpg')
    cfg=json.loads(Path('configs/surgery.json').read_text());cfg.update(device='cuda',sam_postprocessing=False)
    engine=SAM2Engine(cfg,str(frames),baseline)
    p=engine.correct(4,0,(xx-94)**2+(yy-100)**2<35**2)
    assert p.shape==baseline.shape and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    p=engine.correct(8,1,(yy>=170)&(yy<190)&(xx>=128)&(xx<228))
    assert p.shape==baseline.shape and np.isfinite(p).all()
    sam_seconds=time.perf_counter()-start;del engine;torch.cuda.empty_cache()
    export=out/'exports/uterus_primary_tools.ts'
    cmd=[sys.executable,'-m','src.export_joint','--instrument-classes',('1,5,6' if matched_scope else 'union'),'--expected-sha256','c8c763312cba9cff5bb02d492e548cce0cd0975d13c54c1a40f93e8f578bf192','--device','cuda','--output',str(export)]
    if matched_scope:cmd.append('--merge-selected')
    result=subprocess.run(cmd,capture_output=True,text=True)
    (out/'export_stdout.txt').write_text(result.stdout);(out/'export_stderr.txt').write_text(result.stderr)
    if result.returncode:
        (out/'failed_status.json').write_text(json.dumps({'stage':'joint_export','returncode':result.returncode,'sam2_bidirectional_smoke_pass':True,'sam2_seconds':sam_seconds},indent=2));results_volume.commit()
        raise RuntimeError('Joint export failed: '+result.stderr[-4000:])
    report={'run_id':run_id,'scope':'synthetic GPU API/shape/export verification only','clinical_inference':False,'training':False,'gpu':torch.cuda.get_device_name(0),'python':platform.python_version(),'torch':torch.__version__,'sam2_bidirectional_corrections':2,'structures':2,'synthetic_frames':12,'sam_postprocessing':False,'sam2_seconds':sam_seconds,'total_seconds':time.perf_counter()-start,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'joint_export':json.loads(export.with_suffix('.ts.provenance.json').read_text()),'source_hashes':checks}
    (out/'technical_smoke.json').write_text(json.dumps(report,indent=2));results_volume.commit();return report

@app.function(image=image,gpu='A10G',cpu=4,memory=16384,timeout=600,max_containers=1,volumes={'/results':results_volume})
def development_frame_smoke(run_id:str,export_run:str):
    import os,time,hashlib,subprocess,sys
    import numpy as np
    import torch
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.autolaparo import decode_targets
    os.chdir('/opt/isbi');out=Path('/results')/run_id;out.mkdir(parents=True,exist_ok=False)
    source=Path('/results')/export_run/'exports/uterus_primary_tools.ts'
    metadata=json.loads(source.with_suffix('.ts.provenance.json').read_text())
    assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==metadata['export_sha256']
    model=torch.jit.load(str(source),map_location='cuda').eval();mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    rows=[];fig,axes=plt.subplots(3,3,figsize=(12,7.5))
    for row,name in enumerate(['001001','058001','086001']):
        assert int(name[:3])<=170 # official development pool only
        p=Path('/opt/development')/(name+'.jpg');mp=p.with_suffix('.png')
        with Image.open(p) as im:rgb=np.asarray(im.convert('RGB').resize((720,405)))
        with Image.open(mp) as im:gold=decode_targets(np.asarray(im.resize((720,405),Image.Resampling.NEAREST)))
        x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float().cuda()/255
        x=torch.nn.functional.interpolate(x,size=(518,518),mode='bilinear',align_corners=False)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():pred=torch.nn.functional.interpolate(model((x-mean)/std).sigmoid(),size=(405,720),mode='bilinear',align_corners=False)[0]
        torch.cuda.synchronize();seconds=time.perf_counter()-start;pred=pred.cpu().numpy();assert pred.shape==(2,405,720) and np.isfinite(pred).all()
        for k,label in enumerate(['uterus','primary_tools_union']):
            Image.fromarray((pred[k]>.5).astype(np.uint8)*255).save(out/f'{name}_{label}_prediction.png')
        for ax in axes[row]:ax.imshow(rgb);ax.axis('off')
        for k,color in enumerate(['cyan','yellow']):
            if (pred[k]>.5).any():axes[row,1].contour(pred[k]>.5,levels=[.5],colors=[color],linewidths=.7)
            if gold[k].any():axes[row,2].contour(gold[k],levels=[.5],colors=[color],linewidths=.7)
        axes[row,0].set_title(name+' — source');axes[row,1].set_title('Frozen model contours');axes[row,2].set_title('Official mask contours (scope review pending)')
        rows.append(dict(image=name,source_image_sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),source_mask_sha256=hashlib.file_digest(mp.open('rb'),'sha256').hexdigest(),inference_seconds=seconds,probability_range=[float(pred.min()),float(pred.max())]))
    fig.suptitle('DEVELOPMENT TECHNICAL PREVIEW — cyan: uterus; yellow: primary tools; no accuracy estimate')
    fig.tight_layout();fig.savefig(out/'development_preview.png',dpi=150);plt.close(fig)
    report=dict(run_id=run_id,scope='3 official-training-pool frames for pipeline/coordinate/qualitative checks only',clinical_inference=True,training=False,accuracy_evaluated=False,correction_evaluation=False,procedure_grouping_verified=False,gpu=torch.cuda.get_device_name(0),export_sha256=metadata['export_sha256'],source_checkpoint_sha256=metadata['checkpoint_sha256'],frames=rows)
    (out/'development_frame_smoke.json').write_text(json.dumps(report,indent=2));results_volume.commit();return report

clip_image=image.add_local_file(str(ROOT/'data/raw/local/AutoLaparo/autolaparo_task2/clips/clips/001.mp4'),'/opt/development/001.mp4')

@app.function(image=clip_image,gpu='A10G',cpu=4,memory=24576,timeout=1800,max_containers=1,volumes={'/results':results_volume})
def full_clip_technical(run_id:str,export_run:str,use_cache:bool=False,sam_precision:str='float32'):
    import os,sys,time,hashlib,subprocess
    import numpy as np,cv2,torch
    from PIL import Image
    from src.sam_engine import SAM2Engine
    from src.propagation_cache import PropagationCache
    os.chdir('/opt/isbi');out=Path('/results')/run_id;out.mkdir(parents=True,exist_ok=False)
    frames=out/'frames';frames.mkdir();source=Path('/results')/export_run/'exports/uterus_primary_tools.ts'
    provenance=json.loads(source.with_suffix('.ts.provenance.json').read_text())
    assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==provenance['export_sha256']
    model=torch.jit.load(str(source),map_location='cuda').eval()
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    cap=cv2.VideoCapture('/opt/development/001.mp4');assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS)==25
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();prob=[]
    while True:
        ok,bgr=cap.read()
        if not ok:break
        rgb=cv2.cvtColor(cv2.resize(bgr,(720,405),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(frames/f'{len(prob):05d}.jpg',quality=95)
        x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float().cuda()/255
        x=torch.nn.functional.interpolate(x,size=(518,518),mode='bilinear',align_corners=False)
        with torch.inference_mode():p=torch.nn.functional.interpolate(model((x-mean)/std).sigmoid(),size=(405,720),mode='bilinear',align_corners=False)[0]
        prob.append(p.cpu().numpy().astype(np.float16))
    cap.release();torch.cuda.synchronize();feature_seconds=time.perf_counter()-start
    baseline=np.stack(prob);assert baseline.shape==(250,2,405,720) and np.isfinite(baseline).all()
    del model,prob,p,x;torch.cuda.empty_cache();cfg=json.loads(Path('configs/surgery.json').read_text());cfg.update(device='cuda',sam_postprocessing=False)
    cache=PropagationCache() if use_cache else None
    cfg['sam_precision']=sam_precision
    engine=SAM2Engine(cfg,str(frames),baseline,cache=cache);timing=[];prefix_hashes=[]
    # Deliberately prediction masks, not oracle reference labels. No accuracy/budget curve.
    for t,k in [(0,0),(124,1)]:
        mask=baseline[t,k]>.5;begin=time.perf_counter();pred=engine.correct(t,k,mask);torch.cuda.synchronize()
        assert pred.shape==baseline.shape and np.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1
        prefix_hashes.append(hashlib.sha256(pred.tobytes()).hexdigest())
        timing.append(dict(frame=t,structure=k,prompt_source='frozen_framewise_prediction_only',
            prompt_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),seconds=time.perf_counter()-begin))
    warm_timings=[];warm_setup_seconds=None
    if use_cache:
        begin=time.perf_counter();warm=SAM2Engine(cfg,str(frames),baseline,cache=cache,predictor=engine.predictor);warm_setup_seconds=time.perf_counter()-begin
        for index,(t,k) in enumerate([(0,0),(124,1)]):
            begin=time.perf_counter();again=warm.correct(t,k,baseline[t,k]>.5);torch.cuda.synchronize()
            assert hashlib.sha256(again.tobytes()).hexdigest()==prefix_hashes[index], 'Warm cache changed output'
            warm_timings.append(time.perf_counter()-begin)
    report=dict(run_id=run_id,scope='full 250-frame development engineering benchmark; predicted-mask prompts only',
        clinical_inference=True,clip_id='001',official_pool='train',gold_masks_accessed=False,
        clinical_accuracy_evaluated=False,correction_policy_evaluated=False,training=False,
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,working_shape=[250,2,405,720],
        clip_sha256=hashlib.file_digest(open('/opt/development/001.mp4','rb'),'sha256').hexdigest(),
        export_sha256=provenance['export_sha256'],checkpoint_sha256=provenance['checkpoint_sha256'],
        source_hashes={str(p.relative_to('/opt/isbi')):hashlib.file_digest(open(p,'rb'),'sha256').hexdigest() for p in Path('/opt/isbi/src').glob('*.py')},
        decode_framewise_and_save_seconds=feature_seconds,prompt_replay_timings=timing,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),peak_gpu_memory_scope='whole single-process full-clip benchmark',
        sam_postprocessing=False,sam_precision=sam_precision,primary_evaluation_complete=False,propagation_cache_enabled=use_cache,
        cache_hits=cache.hits if cache else 0,cache_misses=cache.misses if cache else 0,
        cache_cpu_bytes=cache.bytes if cache else 0,warm_prefix_bitwise_identical=bool(use_cache),
        warm_engine_setup_seconds=warm_setup_seconds,warm_replay_seconds=warm_timings,
        runtime_note='Shared-cache warm runtime is amortized throughput, not an independent policy runtime comparison')
    (out/'full_clip_technical.json').write_text(json.dumps(report,indent=2))
    (out/'environment_lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    results_volume.commit();return report

@app.function(image=clip_image,gpu='L4',cpu=4,memory=24576,timeout=1800,max_containers=1,volumes={'/results':results_volume})
def full_clip_technical_l4(run_id:str,export_run:str,use_cache:bool=False,sam_precision:str='bfloat16'):
    return full_clip_technical.local(run_id,export_run,use_cache,sam_precision)

@app.local_entrypoint()
def main(stage:str='technical',export_run:str='technical_smoke_20261016T084403Z',matched_scope:bool=False,use_cache:bool=False,sam_precision:str='float32'):
    prefix='technical_smoke_' if stage=='technical' else ('full_clip_technical_' if stage in ['clip','clip_l4'] else 'development_frame_smoke_')
    run_id=prefix+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if stage=='technical':report=technical_smoke.remote(run_id,matched_scope)
    elif stage=='frame':report=development_frame_smoke.remote(run_id,export_run)
    elif stage=='clip':report=full_clip_technical.remote(run_id,export_run,use_cache,sam_precision)
    elif stage=='clip_l4':report=full_clip_technical_l4.remote(run_id,export_run,use_cache,sam_precision)
    else:raise ValueError('Unknown stage')
    out=ROOT/'results'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/(prefix.rstrip('_')+'.json')).write_text(json.dumps(report,indent=2),encoding='utf8')
    if stage=='frame':
        with (out/'development_preview.png').open('wb') as f:
            for chunk in results_volume.read_file(run_id+'/development_preview.png'):f.write(chunk)
    print(json.dumps(report,indent=2))
