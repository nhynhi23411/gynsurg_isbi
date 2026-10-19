"""Reconstruct OOF selection/decisions and verify frozen evidence."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from group_gate import keep_raw,candidates,macro_auc,select_from_training

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    protocol=ROOT/'protocols/20261017_group_gate_v1';out=ROOT/'results/group_gate_development_20261017_v1'
    freeze=json.loads((protocol/'FREEZE.json').read_text())
    for name,digest in freeze['sha256'].items():
        assert sha(ROOT/name)==digest,name
    report=json.loads((out/'report.json').read_text())
    assert report['freeze_sha256']==sha(protocol/'FREEZE.json')
    assert not json.loads((ROOT/'configs/review_status.json').read_text()).get('approved_for_test',False)
    assert not report['final_test_open'] and not report['model_rerun'] and not report['gpu_used']
    assert not report['independent_validation'] and report['default_policy']=='uncertainty_raw'
    obs=pd.read_csv(ROOT/'results/oracle_diagnostics_20261017T075644Z/observations.csv',dtype={'clip_id':str})
    obs=obs[obs.query_count.isin([0,1,2,5])].copy()
    decisions=pd.read_csv(out/'oof_decisions.csv',dtype={'clip_id':str})
    scores=pd.read_csv(out/'all_candidate_source_auc.csv',index_col='procedure_id')
    per=pd.read_csv(out/'per_source_oof_auc.csv',dtype={'clip_id':str}).set_index('procedure_id')
    folds=json.loads((out/'folds.json').read_text());rules={r['name']:r for r in candidates()}
    assert scores.shape==(9,26) and len(decisions)==216 and len(folds)==9
    assert set(scores.index)==set(obs.procedure_id)==set(per.index)
    keys=['procedure_id','clip_id','query_count','frame','structure']
    assert not decisions.duplicated(keys).any()
    original=obs.sort_values(keys).reset_index(drop=True)
    saved=decisions.sort_values(keys).reset_index(drop=True)
    for col in original.columns:
        pd.testing.assert_series_equal(original[col],saved[col],check_names=False,check_exact=False,rtol=0,atol=1e-12)
    for pid,b in decisions.groupby('procedure_id'):
        rule_name,means,inner=select_from_training(scores,pid)
        f=next(f for f in folds if f['outer_held_out']==pid)
        assert f['selected_rule']==rule_name and f['inner_folds']==inner
        assert set(f['selection_sources'])==set(scores.index)-{pid}
        assert all(np.isclose(means[n],f['inner_mean_candidate_auc'][n],rtol=0,atol=1e-12) for n in scores.columns)
        assert b.selected_rule.eq(rule_name).all()
        prompt=b.structure_prompted.astype(str).str.lower().eq('true')
        keep=keep_raw(prompt,b.nearest_query_s,b.sam_entropy,rules[rule_name])
        np.testing.assert_array_equal(keep,b.keep_raw)
        dice=np.where(keep,b.raw_dice,b.baseline_dice)
        np.testing.assert_allclose(dice,b.gated_dice,rtol=0,atol=1e-12)
        assert np.isclose(macro_auc(b,dice),per.loc[pid,'gated_auc'],rtol=0,atol=1e-12)
        for name,rule in rules.items():
            k=keep_raw(prompt,b.nearest_query_s,b.sam_entropy,rule)
            assert np.isclose(macro_auc(b,np.where(k,b.raw_dice,b.baseline_dice)),scores.loc[pid,name],rtol=0,atol=1e-12)
    delta=(per.gated_auc-per.raw_auc).to_numpy()
    np.testing.assert_allclose(delta,per.delta,rtol=0,atol=1e-12)
    rng=np.random.default_rng(20261017)
    ci=np.quantile(delta[rng.integers(0,9,size=(2000,9))].mean(axis=1),[.025,.975])
    np.testing.assert_allclose(ci,report['descriptive_ci95'],rtol=0,atol=1e-12)
    assert np.isclose(delta.mean(),report['mean_delta'],rtol=0,atol=1e-12)
    without=float(per[per.clip_id!='160'].delta.mean())
    assert np.isclose(without,report['mean_delta_excluding_160'],rtol=0,atol=1e-12)
    passes=bool(ci[0]>0 and int((delta>1e-12).sum())>=6 and without>0)
    assert passes==report['development_signal_criterion_passed']
    files=[p for p in out.iterdir() if p.is_file()]+[ROOT/'REPORT_GROUP_GATE_20261017.md',protocol/'FREEZE.json']
    receipt=dict(passed=True,source_groups=9,outer_folds=9,inner_folds=72,candidates=26,oof_rows=216,all_candidate_metrics_reproduced=True,outer_source_exclusion_verified=True,gate_decisions_reproduced=True,final_test_open=False,artifact_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    (ROOT/'materials/verification/group_gate_audit.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
