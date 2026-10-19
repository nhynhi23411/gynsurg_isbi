"""Observational debug output. Never supplies a reference to the selector."""
from pathlib import Path
import json
import numpy as np
from .common import overlap,write_csv
from .policy import features,scaled_candidate_scores

class DebugTrace:
    def __init__(self,out,d,c):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
        self.d=d;self.c=dict(c,fps=float(d['fps']));self.metrics=[];self.decisions=[];self.previous=None;self.reliability=[]
        if c['policy'] not in ['full','uncertainty','error','error_diversity','error_proximity','scaled_diversity_v1']:raise ValueError('Debug score trace supports scored policies only')

    def __call__(self,event):
        p=event['probabilities'];ids=list(event['frame_ids']);n=event['query_count'];lookup={t:i for i,t in enumerate(ids)}
        if event['stage']=='decision':
            if self.c['policy']=='scaled_diversity_v1':
                # Sparse storage, but original raw-frame IDs for temporal distances.
                candidate=[(int(t),k) for t in self.d['query_ids'] for k in range(p.shape[1])]
                rows=scaled_candidate_scores(p,self.d['framewise'][ids],candidate,event['queries'],self.c,frame_ids=ids)
                best=max(rows,key=lambda r:(r['score'],-r['frame'],-r['structure']))
                if (best['frame'],best['structure'])!=event['selected']:raise ValueError('v2 debug selector mismatch')
                self.decisions.extend(dict(query=n+1,**r,selected=(r['frame'],r['structure'])==event['selected']) for r in rows);return
            available=[(int(t),k) for t in self.d['query_ids'] for k in range(p.shape[1]) if (int(t),k) not in event['queries']]
            ent,dis,prox=features(p,self.d['framewise'][ids],self.c['organ_ids'],self.c['instrument_ids'],self.c['sigma'])
            scored=[]
            for t,k in available:
                i=lookup[t];e=ent[i,k] if self.c['policy']=='uncertainty' else .5*(ent[i,k]+dis[i,k])
                same=[a for a,j in event['queries'] if j==k]
                r=max([np.exp(-abs(t-a)/(self.c['tau_s']*self.c['fps'])) for a in same],default=0.)
                alpha=self.c['alpha'] if self.c['policy'] in ['full','error_proximity'] else 0
                beta=self.c['beta'] if self.c['policy'] in ['full','error_diversity'] else 0
                scored.append(dict(query=n+1,frame=t,structure=k,entropy=float(ent[i,k]),disagreement=float(dis[i,k]),proximity=float(prox[i,k]),redundancy=float(r),error_proxy=float(e),proximity_boost=float(e*alpha*prox[i,k]),redundancy_penalty=float(beta*r),score=float(e*(1+alpha*prox[i,k])-beta*r),selected=(t,k)==event['selected']))
            best=max(scored,key=lambda row:(row['score'],-row['frame'],-row['structure']))
            if (best['frame'],best['structure'])!=event['selected']:raise ValueError('Debug reconstruction disagrees with selector')
            self.decisions.extend(scored);return
        np.savez_compressed(self.out/f'state_{n:02d}.npz',frame_ids=np.asarray(ids),probabilities=p)
        self.reliability.extend(dict(query_count=n,**r) for r in event.get('reliability',()))
        for t in ids:
            for k in range(p.shape[1]):
                metric=overlap(p[lookup[t],k]>.5,self.d['gold'][t,k])
                self.metrics.append(dict(query_count=n,frame=t,structure=k,pool='score' if t in self.d['scoring_ids'] else 'query',**metric))
        if self.previous is not None:
            self.panels(self.previous,p,ids,n,event['selected'])
        self.previous=p.copy()

    def panels(self,before,after,ids,n,selected):
        from PIL import Image,ImageDraw
        # Show all official scoring snapshots; other raw frames have no gold here.
        for t in map(int,self.d['scoring_ids']):
            i=ids.index(t);raw=Image.open(Path(str(self.d['jpeg_dir']))/f'{t:05d}.jpg').convert('RGB').resize((320,180))
            panel=Image.new('RGB',(1280,246),'white');draw=ImageDraw.Draw(panel)
            for col,(label,masks) in enumerate([('Raw',None),('Before correction',before[i]>.5),('After correction',after[i]>.5),('Official reference',self.d['gold'][t].astype(bool))]):
                image=np.asarray(raw).copy()
                if masks is not None:
                    for k,color in enumerate([(255,90,30),(40,210,255)]):
                        mask=np.asarray(Image.fromarray(masks[k].astype('uint8')*255).resize((320,180),Image.Resampling.NEAREST))>0
                        image[mask]=(.55*image[mask]+.45*np.asarray(color)).astype('uint8')
                panel.paste(Image.fromarray(image),(col*320,35));draw.text((col*320+5,10),label,fill='black')
            draw.text((5,225),f"{self.d['clip_id']} {self.c['policy']} query #{n}: frame/structure {selected}; score frame {t}; orange uterus / cyan tools",fill='black')
            panel.save(self.out/f'query_{n:02d}_score_{t:03d}.png')

    def finish(self):
        write_csv(self.out/'state_metrics.csv',self.metrics)
        if self.decisions:write_csv(self.out/'candidate_scores.csv',self.decisions)
        if self.reliability:write_csv(self.out/'reliability_weights.csv',self.reliability)
        (self.out/'trace_manifest.json').write_text(json.dumps(dict(scope='Development debug; snapshots only at official query/scoring frames',clinical_review=False,query_gold_used_in_selector=False,state_count=len({r['query_count'] for r in self.metrics}),decision_count=len({r['query'] for r in self.decisions}),probability_storage='float32 at annotated snapshots; full masks after each charged correction',config=self.c),indent=2))
