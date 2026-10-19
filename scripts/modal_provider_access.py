"""Check user-provided NAS links from two networks without logging share tokens."""
from pathlib import Path
import json
import modal
ROOT=Path(__file__).resolve().parents[1]
image=modal.Image.debian_slim(python_version='3.11').pip_install('requests==2.32.3')
app=modal.App('isbi-gynsurg-provider-access-v1',tags={'project':'isbi-gynsurg'})
def check(label,links):
    import concurrent.futures,os,re,time
    import requests
    from urllib.parse import urlsplit
    def one(task):
        start=time.perf_counter()
        try:
            r=requests.get(links[str(task)],timeout=(12,25))
            return dict(task=task,status=r.status_code,bytes=len(r.content),path=urlsplit(r.url).path,scripts=re.findall(r'<script[^>]+src=["\x27]([^"\x27]+)',r.text)[:20],html=r.text[:300000],seconds=time.perf_counter()-start)
        except requests.RequestException as e:return dict(task=task,error_type=type(e).__name__,seconds=time.perf_counter()-start)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:rows=list(ex.map(one,[1,2,3]))
    return dict(network=label,tasks=rows)
@app.function(image=image,cpu=.25,memory=512,timeout=120)
def check_default(links:dict):return check('Modal default region',links)
@app.function(image=image,cpu=.25,memory=512,timeout=120,region='ap')
def check_asia(links:dict):return check('Modal Asia-Pacific',links)
@app.local_entrypoint()
def main():
    links=json.loads((ROOT/'materials/private_access/autolaparo_provider_links.json').read_text(encoding='utf8'))
    handles=[check_default.spawn(links),check_asia.spawn(links)];results=[h.get() for h in handles];public=[]
    for i,r in enumerate(results):
        for row in r['tasks']:
            html=row.pop('html',None)
            if html:(ROOT/f"materials/private_access/network{i}_task{row['task']}.html").write_text(html,encoding='utf8')
        public.append(r)
    (ROOT/'materials/verification/provider_remote_access_check.json').write_text(json.dumps(public,indent=2),encoding='utf8');print(json.dumps(public,indent=2))
