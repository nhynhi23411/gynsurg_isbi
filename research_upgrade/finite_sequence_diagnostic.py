"""EVALUATOR-ONLY retrospective finite-space oracle; never an inference policy."""
from pathlib import Path
import csv,hashlib,itertools,json
import numpy as np
from research_upgrade.bank import QUERY,SCORE,AUC_WEIGHTS
from research_upgrade.acceptance_v2 import bootstrap
ROOT=Path(__file__).resolve().parents[1]
BANK=ROOT/'results/research_upgrade_20261020_v13'
OUT=ROOT/'results/research_upgrade_finite_sequence_diagnostic_20261020_v1'
SEQUENCES=tuple(itertools.permutations(range(6),5))

def charged_states(sequence):
    if len(sequence)!=5 or len(set(sequence))!=5 or any(a not in range(6) for a in sequence):raise ValueError('Five distinct charged actions required')
    paid=[0,0];states=[tuple(paid)]
    for n,a in enumerate(sequence,1):
        j,k=divmod(a,2);paid[k]|=1<<j
        if n in [1,2,5]:states.append(tuple(paid))
    return states

def evaluate(labels,sequence):
    curves=[]
    for state in charged_states(sequence):
        curves.append(float(np.mean([labels[k,state[k],t] for k in range(2) for t in SCORE])))
    return float(np.dot(curves,AUC_WEIGHTS)),curves

def exhaustive(labels,raw_sequence):
    values=[];curves=[]
    for sequence in SEQUENCES:
        value,curve=evaluate(labels,sequence);values.append(value);curves.append(curve)
    raw,raw_curve=evaluate(labels,raw_sequence);hi=int(np.argmax(values));lo=int(np.argmin(values))
    if not values[lo]-1e-12<=raw<=values[hi]+1e-12:raise ValueError('Finite-space bound failure')
    return dict(best_auc=values[hi],worst_auc=values[lo],raw_auc=raw,exact_random_auc=float(np.mean(values)),
        raw_fraction_permutations_at_or_below=float(np.mean(np.asarray(values)<=raw+1e-12)),
        best_sequence=SEQUENCES[hi],worst_sequence=SEQUENCES[lo],raw_sequence=list(raw_sequence),
        best_curve=curves[hi],worst_curve=curves[lo],raw_curve=raw_curve,exact_random_curve=np.mean(curves,axis=0).tolist(),
        sequences=720,evaluator_gold_used=True,deployable=False)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def write(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    protocol=ROOT/'protocols/20261020_finite_sequence_diagnostic_v1/FREEZE.json'
    freeze=json.loads(protocol.read_text())
    for name,h in freeze['hashes'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen diagnostic mismatch')
    if OUT.exists():raise FileExistsError('Preserve diagnostic outputs')
    OUT.mkdir();reports=[];budget=[]
    for cid in [f'{i:03d}' for i in range(1,171)]:
        folder=BANK/'banks'/cid;meta=json.loads((folder/'execution.json').read_text())
        if meta['official_split']!='train' or meta['clip_id']!=cid or sha(folder/'execution.json')!=freeze['training_receipt_hashes'][cid]:raise ValueError('TRAIN/input identity mismatch')
        for name in ['evaluation_labels.csv','policy_queries.csv','policy_curves.csv']:
            if sha(folder/name)!=meta['files'][name]:raise ValueError('Artifact identity mismatch')
        labels={(int(r['head']),int(r['subset']),int(r['frame'])):float(r['dice']) for r in read(folder/'evaluation_labels.csv')}
        if len(labels)!=48 or any(not np.isfinite(v) or not 0<=v<=1 for v in labels.values()):raise ValueError('Incomplete or invalid gold scoring table')
        queries=sorted([r for r in read(folder/'policy_queries.csv') if r['policy']=='uncertainty'],key=lambda r:int(r['query']))
        sequence=[QUERY.index(int(r['frame']))*2+int(r['head']) for r in queries]
        result=exhaustive(labels,sequence)
        saved=sorted([r for r in read(folder/'policy_curves.csv') if r['policy']=='uncertainty'],key=lambda r:int(r['n_queries']))
        np.testing.assert_allclose(result['raw_curve'],[float(r['dice']) for r in saved],rtol=0,atol=1e-12)
        result.update(clip_id=cid,procedure_id=meta['procedure_id']);reports.append(result)
        for method in ['best','worst','raw','exact_random']:
            for n,value in zip([0,1,2,5],result[method+'_curve']):budget.append(dict(clip_id=cid,procedure_id=meta['procedure_id'],diagnostic=method,n_queries=n,dice=value))
    sources=[]
    for source in sorted({r['procedure_id'] for r in reports}):
        group=[r for r in reports if r['procedure_id']==source]
        sources.append(dict(procedure_id=source,clips=len(group),**{name:float(np.mean([r[name] for r in group])) for name in ['best_auc','worst_auc','raw_auc','exact_random_auc','raw_fraction_permutations_at_or_below']}))
    summary=dict(clips=170,sources=9,sequences_per_clip=720,posthoc=True,evaluator_only=True,deployable=False,
        final_test=False,method_promoted=False,scope='Only the frozen six-action/five-mask independent-head finite space',
        mean_auc={m:float(np.mean([r[m+'_auc'] for r in sources])) for m in ['best','worst','raw','exact_random']},
        contrasts={m+'_minus_raw':bootstrap([r[m+'_auc']-r['raw_auc'] for r in sources]) for m in ['best','exact_random']},
        freeze_sha256=sha(protocol))
    (OUT/'clip_details.json').write_text(json.dumps(reports,indent=2,allow_nan=False));write(OUT/'source_diagnostic.csv',sources);write(OUT/'budget_diagnostic.csv',budget)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
