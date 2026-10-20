import unittest
import numpy as np
from research_upgrade.bank import FRAMES,QUERY,SCORE,trajectory,entropy,observable_features,PaidView
from research_upgrade.online_replay import run
from research_upgrade.acceptance_v2 import acceptance_decisions

class ToyEngine:
    def __init__(self,p):self.baseline=p;self.prompts=[];self.calls=[]
    def correct(self,t,k,mask):
        self.prompts.append((t,k,mask.copy()));self.calls.append([(a,b) for a,b,_ in self.prompts]);p=self.baseline.copy()
        subset=sum(1<<QUERY.index(a) for a,b,_ in self.prompts if b==k)
        p[:,k]=.12+.07*subset
        for a,b,m in self.prompts:p[a,b]=m
        return p

class OnlineReplayTests(unittest.TestCase):
    def test_same_raw_queries_as_factorized_bank(self):
        p=np.full((125,2,2,2),.2,np.float32);states={}
        for k in range(2):
            for s in range(8):
                states[k,s]=np.full((len(FRAMES),2,2),.12+.07*s,np.float32) if s else p[list(FRAMES),k].copy()
                for j,t in enumerate(QUERY):
                    if s&(1<<j):states[k,s][FRAMES.index(t)]=1
        selector={(k,s,t):entropy(states[k,s][FRAMES.index(t)]) for k in range(2) for s in range(8) for t in QUERY}
        for policy in ['uncertainty','uniform','random']:
            with self.subTest(policy=policy):
                purchased=[];engine=ToyEngine(p)
                def buy(t,k):purchased.append((t,k));return np.ones((2,2),bool)
                out=run(p,engine,buy,policy=policy,seed=100)
                paths,expected=trajectory(selector,policy,100)
                self.assertEqual(purchased,expected);self.assertEqual(len(set(purchased)),5)
                for n,a in out['predictions'].items():
                    expected_array=np.stack([np.stack([states[k,paths[n][k]][FRAMES.index(t)] for k in range(2)]) for t in SCORE])
                    np.testing.assert_allclose(a,expected_array)
                self.assertEqual(len(out['propagations']),5)
    def test_gate_only_uses_paid_masks_and_reconstructs_support(self):
        p=np.full((125,2,2,2),.2,np.float32);engine=ToyEngine(p);purchased=[];features=[]
        states={(k,0):p[list(FRAMES),k].copy() for k in range(2)}
        dummy=PaidView(states,{},0,0);names=list(observable_features(dummy,0,SCORE[0]))
        class Reject:
            def predict(self,a):features.append(a);return np.array([-1.])
        def buy(t,k):purchased.append((t,k));return np.ones((2,2),bool)
        out=run(p,engine,buy,model=Reject(),feature_names=names)
        self.assertTrue(features)
        self.assertEqual(out['paid_masks'],5)
        self.assertTrue(any(not d['accepted'] for d in out['decisions']))
        for c in out['propagations']:self.assertFalse(c['subset']&~c['paid_at_execution'])
        self.assertEqual(len(out['propagations']),len({(c['head'],c['subset']) for c in out['propagations']}))
        self.assertFalse(out['precomputed_future_prompt_states'])
    def test_invalid_reference_callback_fails(self):
        p=np.full((125,2,2,2),.2,np.float32)
        with self.assertRaises(ValueError):run(p,ToyEngine(p),lambda t,k:np.full((2,2),255))

    def test_on_demand_features_and_decisions_equal_finite_bank(self):
        p=np.full((125,2,2,2),.2,np.float32);states={};anchors={(k,j):np.ones((2,2),bool) for k in range(2) for j in range(3)}
        for k in range(2):
            for subset in range(8):
                if subset:
                    engine=ToyEngine(p);prompts=[(QUERY[j],k,anchors[k,j]) for j in range(3) if subset&(1<<j)]
                    engine.prompts=prompts[:-1];states[k,subset]=engine.correct(*prompts[-1])[list(FRAMES),k]
                else:states[k,0]=p[list(FRAMES),k].copy()
        selector={(k,s,t):entropy(states[k,s][FRAMES.index(t)]) for k in range(2) for s in range(8) for t in QUERY}
        paths,history=trajectory(selector,'uncertainty');features={};names=None
        for k in range(2):
            for paid in range(8):
                for subset in range(8):
                    if subset&~paid:continue
                    for t in SCORE:
                        f=observable_features(PaidView(states,anchors,k,paid),subset,t);features[k,paid,subset,t]=f;names=list(f)
        class Gate:
            def __init__(self):self.inputs=[]
            def predict(self,x):self.inputs.append(x.copy());return x[:,names.index('anchor_reconstruction_mean')]-.5
        a=Gate();b=Gate();clip=dict(cid='toy',source='synthetic',features=features,names=names,paths=paths,history=history)
        expected,decisions=acceptance_decisions(clip,a,0.)
        actual=run(p,ToyEngine(p),lambda t,k:np.ones((2,2),bool),model=b,feature_names=names)
        np.testing.assert_allclose(np.concatenate(a.inputs),np.concatenate(b.inputs))
        self.assertEqual([r['accepted'] for r in decisions],[r['accepted'] for r in actual['decisions']])
        for n,result in actual['predictions'].items():
            target=np.stack([np.stack([states[k,expected[n][k,t]][FRAMES.index(t)] for k in range(2)]) for t in SCORE])
            np.testing.assert_allclose(result,target)

if __name__=='__main__':unittest.main()
