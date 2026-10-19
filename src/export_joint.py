"""Export the archived frozen joint DINOv2 model without historical launchers.
Only audited model classes are loaded. Encoder architecture comes from the saved
official config; no pretrained weights are fetched or trained by this exporter.
Export is fixed to batch 1 and normalized RGB 518x518, verified on synthetic inputs.
"""
import argparse,ast,json,math
from pathlib import Path
from unittest import mock
import torch
from torch import nn
import torch.nn.functional as F
from .common import approved,sha
ROOT=Path(__file__).resolve().parents[1]
INSTRUMENT_NAMES=['background','grasper','scissors','irrigator','bipolar-forceps','sealer-divider','hook','suturing-instrument']
class BinaryAdapter(nn.Module):
    def __init__(self,model,classes,union=False,merge_selected=False):super().__init__();self.model=model;self.classes=tuple(classes);self.union=union;self.merge_selected=merge_selected
    def forward(self,x):
        logits=self.model(x,['instrument','anatomy']);org=logits['anatomy'].softmax(1)[:,1:2];ins=logits['instrument'].softmax(1)
        tools=1-ins[:,0:1] if self.union else (ins[:,list(self.classes)].sum(1,keepdim=True) if self.merge_selected else torch.cat([ins[:,k:k+1] for k in self.classes],1))
        p=torch.cat([org,tools],1).clamp(1e-5,1-1e-5)
        return p.log()-torch.log1p(-p)
def main():
    a=argparse.ArgumentParser();a.add_argument('--checkpoint',default='materials/local/followup/baseline/baseline_final.pth');a.add_argument('--expected-sha256',required=True);a.add_argument('--source',default='materials/local/followup/baseline/phase53_source.py');a.add_argument('--encoder-config',default='materials/checkpoints/dinov2-base_config.json');a.add_argument('--instrument-classes',required=True,help='Explicit comma-separated primary class IDs 1..7, or union for one all-primary-tools channel');a.add_argument('--merge-selected',action='store_true',help='Merge only the explicit instrument IDs into one binary channel');a.add_argument('--output',required=True);a.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=a.parse_args()
    approved(ROOT);checkpoint=Path(args.checkpoint);source=Path(args.source);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    if sha(checkpoint)!=args.expected_sha256:raise ValueError('Checkpoint fingerprint mismatch')
    union=args.instrument_classes=='union';classes=list(range(1,8)) if union else [int(x) for x in args.instrument_classes.split(',')]
    if not classes or len(set(classes))!=len(classes) or any(k<1 or k>7 for k in classes):raise ValueError('Instrument classes must be unique IDs 1..7')
    allowed={'MultiLevelHead','LastLayerHead','DinoScene'};tree=ast.parse(source.read_text(encoding='utf8'));nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in allowed]
    if {n.name for n in nodes}!=allowed:raise ValueError('Missing audited model definitions')
    ns={'nn':nn,'torch':torch,'F':F,'math':math,'NCLS':{'instrument':8,'anatomy':4}}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    from transformers import Dinov2Config,Dinov2Model
    config=Dinov2Config.from_dict(json.loads(Path(args.encoder_config).read_text()))
    with mock.patch.object(Dinov2Model,'from_pretrained',side_effect=lambda *a,**k:Dinov2Model(config)):
        model=ns['DinoScene'](['instrument','anatomy'],head_types={'anatomy':'last'})
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload.get('smoke') or payload.get('epoch')!=20:raise ValueError('Expected the documented full frozen baseline, not smoke or another recipe')
    if args.merge_selected and union:raise ValueError('Choose explicit class IDs with --merge-selected')
    model.load_state_dict(payload['model'],strict=True);wrapped=BinaryAdapter(model,classes,union,args.merge_selected).to(args.device).eval();torch.manual_seed(20261016)
    example=torch.zeros(1,3,518,518,device=args.device)
    checks=[]
    with torch.inference_mode():
        traced=torch.jit.trace(wrapped,example,strict=False)
        for sample in [example,torch.randn_like(example)]:
            eager=wrapped(sample);compiled=traced(sample)
            ep,cp=eager.sigmoid(),compiled.sigmoid()
            check=dict(logit_max_abs=float((eager-compiled).abs().max()),probability_max_abs=float((ep-cp).abs().max()),binary_disagreement_pixels=int(((ep>.5)!=(cp>.5)).sum()),probability_allclose=bool(torch.allclose(ep,cp,atol=1e-5,rtol=1e-4)))
            checks.append(check)
            if not torch.isfinite(compiled).all() or not check['probability_allclose']:raise ValueError('Synthetic trace probability equivalence failed: '+json.dumps(check))
    out.parent.mkdir(parents=True,exist_ok=True);traced.save(str(out))
    scopes=['uterus']+(['primary-instrument-union'] if union else (['selected-union:'+','.join(INSTRUMENT_NAMES[k] for k in classes)] if args.merge_selected else [INSTRUMENT_NAMES[k] for k in classes]))
    report=dict(checkpoint_sha256=sha(checkpoint),source_sha256=sha(source),encoder_config_sha256=sha(args.encoder_config),export_sha256=sha(out),structure_names=scopes,input_shape=[1,3,518,518],input_normalization='ImageNet mean/std from source',output='independent binary logits constructed from categorical softmax',synthetic_trace_check=True,trace_checks=checks,trace_tolerance='probability space atol=1e-5 rtol=1e-4; logits may amplify tiny probability errors near clamp limits',clinical_inference=False,training=False,exposure_note='This baseline was trained on all source GynSurg cases; only separately verified unseen procedures can be final test')
    out.with_suffix(out.suffix+'.provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
