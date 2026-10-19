"""Retrospective diagnosis; preserve empty-reference convention and main endpoint."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scipy.integrate import trapezoid
ROOT=Path(__file__).resolve().parents[1]

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);args=a.parse_args();base=ROOT/'results'/args.run
    auc=pd.read_csv(base/'analysis/recipe_auc.csv',dtype={'clip_id':str}).pivot(index=['procedure_id','clip_id'],columns='recipe',values='dice_auc')
    auc['prototype_delta']=auc.scaled_fallback-auc.uncertainty_raw;auc.sort_values('prototype_delta').to_csv(base/'analysis/source_deltas_ranked.csv')
    rows=[];curves=[];weights=[]
    for pid,cid in auc.index:
        for recipe in ['uncertainty_raw','scaled_raw','scaled_fallback','uncertainty_fallback']:
            folder=base/cid/'suite'/recipe;f=pd.read_csv(folder/'per_frame.csv');f['gold_present']=f.gold_present.astype(str).str.lower()=='true';f['prediction_present']=f.prediction_present.astype(str).str.lower()=='true'
            for (density,k),b in f.groupby(['density','structure']):
                rows.append(dict(procedure_id=pid,clip_id=cid,recipe=recipe,density=density,structure=k,n_frames=len(b),empty_reference_frames=int((~b.gold_present).sum()),empty_reference_foreground_predictions=int((~b.gold_present&b.prediction_present).sum()),mean_dice=float(b.dice.mean())))
            present=f[f.gold_present].groupby(['actual_density','structure']).dice.mean().groupby('actual_density').mean()
            if len(present)==4:curves.append(dict(procedure_id=pid,clip_id=cid,recipe=recipe,present_only_auc=float(trapezoid(present.values,present.index)/(present.index[-1]-present.index[0]))))
            path=folder/'debug/reliability_weights.csv'
            if path.exists():
                w=pd.read_csv(path);w=w[(w.frame.isin([74,99,124]))&(~w.charged_prompt_override)]
                for k,b in w.groupby('structure'):weights.append(dict(clip_id=cid,recipe=recipe,structure=k,median_weight=float(b.weight.median()),median_baseline_entropy=float(b.framewise_entropy.median()),median_sam_entropy=float(b.sam_entropy.median()),median_disagreement=float(b.disagreement.median())))
    table=pd.DataFrame(rows);table.to_csv(base/'analysis/empty_reference_diagnostics.csv',index=False)
    pd.DataFrame(weights).to_csv(base/'analysis/scoring_weight_diagnostics.csv',index=False)
    present=pd.DataFrame(curves);present.to_csv(base/'analysis/present_only_recipe_auc.csv',index=False)
    paired=present.pivot(index=['procedure_id','clip_id'],columns='recipe',values='present_only_auc')
    delta=float((paired.scaled_fallback-paired.uncertainty_raw).mean())
    worst=auc.sort_values('prototype_delta').head(2).reset_index();result=dict(posthoc=True,worst_sources=worst[['procedure_id','clip_id','prototype_delta']].to_dict('records'),present_only_mean_prototype_delta=delta,present_only_n_sources=len(paired),primary_endpoint_changed=False,clinical_label_review=False)
    (base/'analysis/failure_diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2));print(table[(table.clip_id.isin(worst.clip_id))&(table.structure==0)].to_string(index=False));print(pd.DataFrame(weights).query('structure == 0').to_string(index=False))

if __name__=='__main__':main()
