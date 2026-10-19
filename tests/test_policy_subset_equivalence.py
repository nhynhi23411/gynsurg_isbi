import unittest,numpy as np
from src.policy import choose,features
class PolicySubsetEquivalence(unittest.TestCase):
    def test_eligible_frame_optimization_matches_full_frame_reference(self):
        for seed in range(5):
            rng=np.random.default_rng(seed);p=rng.random((125,2,8,10));f=rng.random(p.shape)
            candidates=[(t,k) for t in [0,24,49] for k in [0,1]];queries=[(24,1)]
            cfg=dict(organ_ids=[0],instrument_ids=[1],sigma=.03,alpha=1,beta=.25,tau_s=2,fps=25)
            ent,dis,prox=features(p,f,[0],[1],.03)
            for policy in ['uncertainty','error','error_diversity','error_proximity','full']:
                e=ent if policy=='uncertainty' else .5*(ent+dis);alpha=cfg['alpha'] if policy in ['full','error_proximity'] else 0;beta=cfg['beta'] if policy in ['full','error_diversity'] else 0
                def score(q):
                    t,k=q;same=[a for a,j in queries if j==k];red=max([np.exp(-abs(t-a)/(cfg['tau_s']*cfg['fps'])) for a in same],default=0)
                    return e[t,k]*(1+alpha*prox[t,k])-beta*red
                expected=max([q for q in candidates if q not in queries],key=lambda q:(score(q),-q[0],-q[1]))
                self.assertEqual(choose(p,f,candidates,queries,policy,cfg,rng),expected)
if __name__=='__main__':unittest.main()
