"""Observable, paid-anchor acceptance models and source-grouped evaluation.

Fitting is separate from inference. The inference gate never receives evaluator
labels; they are used only after decisions to summarize development outcomes.
"""
from pathlib import Path
import argparse,hashlib,json,datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.preprocessing import StandardScaler
from research_upgrade.bank import trajectory,SCORE,BUDGETS,AUC_WEIGHTS,popcount

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research_upgrade_20261019_v1'
META={'clip_id','procedure_id','head','paid','subset','frame'}

def bootstrap(delta):
    delta=np.asarray(delta,float);rng=np.random.default_rng(20261019)
    boot=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(axis=1)
    return dict(delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n_sources=len(delta),exploratory=True)

class PairModel:
    def __init__(self,family='ridge',ablation='anchors'):
        self.family=family;self.ablation=ablation;self.scaler=StandardScaler()
    def fit(self,X,y,groups,columns):
        self.columns=list(columns);self.keep=[i for i,n in enumerate(columns) if self.ablation=='anchors' or 'anchor_' not in n]
        a=np.asarray(X,float)[:,self.keep];target=np.asarray(y,float)
        groups=np.asarray(groups);counts={g:int((groups==g).sum()) for g in np.unique(groups)}
        weights=np.array([1/counts[g] for g in groups]);weights/=weights.mean()
        a=self.scaler.fit_transform(a,sample_weight=weights)
        if self.family=='ridge':
            self.model=Ridge(alpha=10.);self.model.fit(a,target,sample_weight=weights)
        elif self.family=='logistic':
            classification=(target>1e-8).astype(int)
            if len(np.unique(classification))<2:
                self.model=None;self.constant=float(np.average(classification,weights=weights));return self
            self.model=LogisticRegression(C=1.,max_iter=300,random_state=20261019)
            weights*=np.maximum(np.abs(target),.005);weights/=weights.mean()
            self.model.fit(a,classification,sample_weight=weights)
        else:raise ValueError('Unknown family')
        return self
    def predict(self,X):
        a=self.scaler.transform(np.asarray(X,float)[:,self.keep])
        if self.model is None:return np.full(len(a),self.constant)
        return self.model.predict(a) if self.family=='ridge' else self.model.predict_proba(a)[:,1]

def paired_features(new,old,names):
    a=np.array([new[k] for k in names],float);b=np.array([old[k] for k in names],float)
    return np.concatenate([a,b,a-b])

def load_banks(ids):
    clips={};X=[];y=[];groups=[];columns=None
    for cid in ids:
        folder=OUT/'banks'/cid;receipt=json.loads((folder/'execution.json').read_text())
        for name,h in receipt['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h
        ff=pd.read_csv(folder/'features.csv');ll=pd.read_csv(folder/'evaluation_labels.csv');ss=pd.read_csv(folder/'selector_observables.csv')
        names=[n for n in ff.columns if n not in META]
        if {'dice','iou','gold_present','prediction_present'}&set(names):raise ValueError('Scoring label in feature columns')
        cols=[prefix+n for prefix in ['new_','old_','delta_'] for n in names]
        if columns is not None and cols!=columns:raise ValueError('Feature schema changed')
        columns=cols
        f={(int(r['head']),int(r.paid),int(r.subset),int(r.frame)):r.to_dict() for _,r in ff.iterrows()}
        labels={(int(r['head']),int(r.subset),int(r.frame)):r.to_dict() for _,r in ll.iterrows()}
        selector={(int(r['head']),int(r.subset),int(r.frame)):float(r.entropy) for _,r in ss.iterrows()}
        source=receipt['procedure_id'];paths,history=trajectory(selector,'uncertainty')
        clip=dict(cid=cid,source=source,features=f,labels=labels,selector=selector,names=names,paths=paths,history=history)
        clips[cid]=clip
        for head in range(2):
            for paid in range(1,8):
                if popcount(paid)<2:continue
                for old in range(8):
                    if old==paid or old&~paid:continue
                    for frame in SCORE:
                        X.append(paired_features(f[head,paid,paid,frame],f[head,paid,old,frame],names))
                        y.append(labels[head,paid,frame]['dice']-labels[head,old,frame]['dice']);groups.append(source)
    return clips,np.array(X),np.array(y),np.array(groups),columns

def acceptance_decisions(clip,model,threshold):
    """Only observable features and raw query history; no labels read here."""
    accepted={(k,t):0 for k in range(2) for t in SCORE};decisions=[];states={0:dict(accepted)}
    for n,(_,head) in enumerate(clip['history'],1):
        paid=clip['paths'][n][head]
        for t in SCORE:
            old=accepted[head,t];new=paid;use=True;value=None
            if popcount(paid)>=2 and old!=paid:
                feature=paired_features(clip['features'][head,paid,new,t],clip['features'][head,paid,old,t],clip['names'])
                value=float(model.predict(feature[None,:])[0]);use=value>threshold
            if use:accepted[head,t]=new
            decisions.append(dict(clip_id=clip['cid'],procedure_id=clip['source'],query=n,head=head,frame=t,paid=paid,old_subset=old,new_subset=new,accepted=bool(use),score=value))
        states[n]=dict(accepted)
    return states,decisions

def evaluate_states(clip,states):
    rows=[]
    for budget in BUDGETS:
        cells=[clip['labels'][k,states[budget][k,t],t] for k in range(2) for t in SCORE]
        rows.append(dict(clip_id=clip['cid'],procedure_id=clip['source'],n_queries=budget,
            dice=float(np.mean([r['dice'] for r in cells])),iou=float(np.mean([r['iou'] for r in cells]))))
    return rows

def source_auc(rows,metric='dice'):
    f=pd.DataFrame(rows);f['weighted']=f[metric]*f.n_queries.map(dict(zip(BUDGETS,AUC_WEIGHTS)))
    return f.groupby(['procedure_id','clip_id']).weighted.sum().groupby('procedure_id').mean()

def fit_subset(X,y,groups,cols,train_sources,family,ablation):
    mask=np.isin(groups,list(train_sources))
    if not mask.any():raise ValueError('No training sources')
    return PairModel(family,ablation).fit(X[mask],y[mask],groups[mask],cols)

def nested_oof(ids,output=None):
    clips,X,y,groups,cols=load_banks(ids);sources=sorted(set(groups));assert len(sources)>=4
    output=Path(output or OUT/'analysis');output.mkdir(parents=True,exist_ok=True)
    candidate_configs=[('ridge',0.),('ridge',.01),('logistic',.5),('logistic',.6)]
    raw_rows=[]
    for clip in clips.values():
        states={n:{(k,t):clip['paths'][n][k] for k in range(2) for t in SCORE} for n in BUDGETS}
        raw_rows.extend(evaluate_states(clip,states))
    raw=source_auc(raw_rows);summaries={};all_folds=[];all_decisions=[]
    for ablation in ['controls','anchors']:
        final_rows=[]
        for outer in sources:
            train=[s for s in sources if s!=outer];scores=[]
            # Three fixed source folds inside each outer training fold.
            inner_folds=[train[i::3] for i in range(3)]
            for family,threshold in candidate_configs:
                rows=[]
                for held in inner_folds:
                    fit_sources=[s for s in train if s not in held]
                    model=fit_subset(X,y,groups,cols,fit_sources,family,ablation)
                    for clip in clips.values():
                        if clip['source'] in held:
                            state,_=acceptance_decisions(clip,model,threshold);rows.extend(evaluate_states(clip,state))
                gain=float((source_auc(rows)-raw.reindex(train)).mean());scores.append((gain,family,threshold))
            # Stable order determines ties; no outer-source metrics are used.
            choice=max(range(len(scores)),key=lambda i:(scores[i][0],-i));gain,family,threshold=scores[choice]
            model=fit_subset(X,y,groups,cols,train,family,ablation)
            for clip in clips.values():
                if clip['source']==outer:
                    state,decisions=acceptance_decisions(clip,model,threshold);final_rows.extend(evaluate_states(clip,state))
                    all_decisions.extend(dict(ablation=ablation,**r) for r in decisions)
            all_folds.append(dict(ablation=ablation,held_source=outer,training_sources=train,inner_folds=inner_folds,family=family,threshold=threshold,inner_delta=gain,candidate_scores=scores))
            print('OOF',ablation,outer,family,threshold,flush=True)
        selected=source_auc(final_rows);delta=selected-raw
        summaries[ablation]=dict(mean_auc=float(selected.mean()),raw_auc=float(raw.mean()),candidate_minus_raw=bootstrap(delta),positive_sources=int((delta>0).sum()),per_source_delta=delta.to_dict())
        pd.DataFrame(final_rows).to_csv(output/(ablation+'_oof_curves.csv'),index=False)
    pd.DataFrame(raw_rows).to_csv(output/'raw_curves.csv',index=False);pd.DataFrame(all_decisions).to_csv(output/'oof_decisions.csv',index=False)
    (output/'folds.json').write_text(json.dumps(all_folds,indent=2),encoding='utf8')
    result=dict(cohort_clips=len(clips),source_count=len(sources),training_pairs=len(y),feature_columns=cols,
        sources_already_development_exposed=True,independent_validation=False,final_test=False,results=summaries,
        trained_model_is_not_a_risk_guarantee=True,generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['representatives','full'],default='representatives');a=p.parse_args()
    acceptance=OUT/('representative_acceptance.json' if a.stage=='representatives' else 'full_acceptance.json')
    if not acceptance.exists() or not json.loads(acceptance.read_text())['passed']:raise ValueError('Accepted banks required')
    ids=[r['clip_id'] for r in json.loads(acceptance.read_text())['reports']]
    print(json.dumps(nested_oof(ids,OUT/('analysis_'+a.stage)),indent=2))
