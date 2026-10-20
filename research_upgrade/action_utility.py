"""Expected marginal correction utility from paid-state observables only."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from research_upgrade.bank import QUERY,SCORE,BUDGETS,popcount,score_paths,trajectory
from research_upgrade.acceptance_v2 import source_auc,bootstrap
ROOT=Path(__file__).resolve().parents[1];BANK=ROOT/'results/research_upgrade_20261020_v13';OUT=ROOT/'results/research_upgrade_20261020_action_utility_v1'
STATIC=['target_uterus','paid_count','candidate_seconds','nearest_paid_seconds','mean_paid_seconds','paid_available']
DYNAMIC=['entropy','baseline_entropy','entropy_delta','entropy_minus_eligible_mean','entropy_minus_eligible_max','entropy_times_paid_count','entropy_times_target_uterus']

def features(selector,head,paid,action):
    if paid&(1<<action):raise ValueError('Candidate already purchased')
    available=[j for j in range(3) if not paid&(1<<j)];purchased=[QUERY[j] for j in range(3) if paid&(1<<j)]
    # Only current paid state and baseline are accessible; no future prompt state is looked up.
    e=float(selector[head,paid,QUERY[action]]);base=float(selector[head,0,QUERY[action]])
    eligible=[float(selector[head,paid,QUERY[j]]) for j in available]
    values=[int(head==0),len(purchased),QUERY[action]/25.,min([abs(QUERY[action]-q)/25. for q in purchased],default=10.),np.mean(purchased)/25. if purchased else 0.,int(bool(purchased)),
        e,base,e-base,e-float(np.mean(eligible)),e-max(eligible),e*len(purchased),e*int(head==0)]
    if not np.isfinite(values).all():raise ValueError('Nonfinite observable')
    return dict(zip(STATIC+DYNAMIC,map(float,values)))

class UtilityModel:
    def __init__(self,ablation='full'):self.names=STATIC+(DYNAMIC if ablation=='full' else [])
    def fit(self,X,y,groups):
        x=np.asarray(X,float);y=np.asarray(y,float);groups=np.asarray(groups)
        w=np.array([1/np.sum(groups==s) for s in groups]);w/=w.mean()
        self.mean=np.average(x,axis=0,weights=w);var=np.average((x-self.mean)**2,axis=0,weights=w)
        self.scale=np.sqrt(var);self.scale[self.scale<1e-12]=1.
        self.model=Ridge(alpha=10.).fit((x-self.mean)/self.scale,y,sample_weight=w);return self
    def predict(self,X):
        p=self.model.predict((np.asarray(X,float)-self.mean)/self.scale)
        if not np.isfinite(p).all():raise ValueError('Nonfinite predicted utility')
        return p

def choose_paths(selector,model):
    paid=[0,0];paths={0:tuple(paid)};history=[]
    for n in range(1,6):
        eligible=[(j,k) for j in range(3) for k in range(2) if not paid[k]&(1<<j)]
        X=[[features(selector,k,paid[k],j)[name] for name in model.names] for j,k in eligible]
        values=model.predict(X);idx=max(range(len(eligible)),key=lambda i:(float(values[i]),-QUERY[eligible[i][0]],-eligible[i][1]))
        j,k=eligible[idx];paid[k]|=1<<j;paths[n]=tuple(paid);history.append((QUERY[j],k))
    return paths,history

def load():
    proof=json.loads((BANK/'full_acceptance.json').read_text())
    if not proof['passed'] or len(proof['reports'])!=170:raise ValueError('Complete accepted bank required')
    clips={};rows=[];labels=[]
    for record in proof['reports']:
        cid=record['clip_id'];folder=BANK/'banks'/cid;e=json.loads((folder/'execution.json').read_text())
        for name,h in e['files'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=h:raise ValueError('Bank hash mismatch')
        ss=pd.read_csv(folder/'selector_observables.csv');ll=pd.read_csv(folder/'evaluation_labels.csv')
        selector={(int(r['head']),int(r['subset']),int(r['frame'])):float(r['entropy']) for _,r in ss.iterrows()}
        gold={(int(r['head']),int(r['subset']),int(r['frame'])):r.to_dict() for _,r in ll.iterrows()}
        clips[cid]=dict(cid=cid,source=e['procedure_id'],selector=selector,labels=gold)
        for k in range(2):
            for paid in range(8):
                for j in range(3):
                    if paid&(1<<j):continue
                    key=dict(clip_id=cid,procedure_id=e['procedure_id'],head=k,paid=paid,action=j)
                    rows.append(dict(**key,**features(selector,k,paid,j)))
                    before=np.mean([gold[k,paid,t]['dice'] for t in SCORE]);after=np.mean([gold[k,paid|(1<<j),t]['dice'] for t in SCORE])
                    labels.append(dict(**key,marginal_head_dice=float(after-before)))
    return clips,pd.DataFrame(rows),pd.DataFrame(labels)

def main():
    if OUT.exists():raise FileExistsError('Preserve existing hypothesis output; no overwrite/tuning')
    freeze=ROOT/'protocols/20261020_action_utility_v1/FREEZE.json'
    frozen=json.loads(freeze.read_text())
    for f,h in frozen['hashes'].items():
        if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h:raise ValueError('Action-utility freeze mismatch')
    clips,F,L=load();sources=sorted(F['procedure_id'].unique());assert len(sources)==9 and len(F)==4080
    OUT.mkdir();F.to_csv(OUT/'training_features.csv',index=False);L.to_csv(OUT/'training_utilities.csv',index=False)
    raw_rows=[]
    for clip in clips.values():
        path,_=trajectory(clip['selector'],'uncertainty');raw_rows.extend(score_paths(path,clip['labels'],'uncertainty',20261017,clip['cid'],clip['source']))
    raw=source_auc(raw_rows);results={};curves=[];queries=[];folds=[]
    for ablation in ['static','full']:
        rows=[]
        for held in sources:
            train=F['procedure_id']!=held;model=UtilityModel(ablation)
            model.fit(F.loc[train,model.names].values,L.loc[train,'marginal_head_dice'].values,F.loc[train,'procedure_id'].values)
            folds.append(dict(ablation=ablation,held_source=held,training_sources=[s for s in sources if s!=held],pairs=int(train.sum()),ridge_alpha=10.))
            for clip in clips.values():
                if clip['source']!=held:continue
                paths,history=choose_paths(clip['selector'],model)
                rows.extend(score_paths(paths,clip['labels'],'utility_'+ablation,20261020,clip['cid'],held))
                queries.extend(dict(method=ablation,clip_id=clip['cid'],procedure_id=held,query=n,frame=t,head=k) for n,(t,k) in enumerate(history,1))
            print('UTILITY OOF',ablation,held,flush=True)
        values=source_auc(rows);delta=values-raw
        results[ablation]=dict(mean_auc=float(values.mean()),candidate_minus_raw=bootstrap(delta),positive_sources=int((delta>0).sum()),per_source_delta=delta.to_dict());curves.extend(rows)
    diff=np.array([results['full']['per_source_delta'][s]-results['static']['per_source_delta'][s] for s in sources])
    gate=results['full']['candidate_minus_raw']['delta']>=.005 and results['full']['positive_sources']>=6
    report=dict(clips=170,sources=9,training_action_pairs=len(F),results=results,full_minus_static=bootstrap(diff),internal_development_gate_pass=bool(gate),
        one_fixed_family_no_tuning=True,extra_offline_outcome_supervision=True,exploratory_after_prior_negative=True,final_test=False,validation_performed=False,clinical_or_confirmatory_claim=False,
        freeze_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest())
    pd.DataFrame(curves).to_csv(OUT/'oof_curves.csv',index=False);pd.DataFrame(queries).to_csv(OUT/'oof_queries.csv',index=False)
    (OUT/'folds.json').write_text(json.dumps(folds,indent=2));(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
