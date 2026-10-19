from pathlib import Path
import requests,concurrent.futures,hashlib,json,zipfile,time,shutil,threading
R=Path(__file__).resolve().parents[1];DEST=R/'data/raw/archives'
CHUNK=8*1024*1024
LOCAL=threading.local()
def session():
    if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session()
    return LOCAL.session
def download(name):
    url='http://ftp.itec.aau.at/datasets/GynSurge/Semantic_Segmentation_Dataset/'+name+'.zip'
    dst=DEST/(name+'.zip');part=dst.with_suffix('.zip.part');state=dst.with_suffix('.ranges.json')
    if dst.exists() and dst.with_suffix('.verified.json').exists():return json.loads(dst.with_suffix('.verified.json').read_text())
    if state.exists():size=json.loads(state.read_text())['size']
    else:
        size=None
    for attempt in range(5):
        if size is not None:break
        try:
            r=requests.get(url,headers={'Range':'bytes=0-0'},timeout=(15,30));r.raise_for_status();size=int(r.headers['Content-Range'].split('/')[-1]);break
        except Exception:
            if attempt==4:raise
            time.sleep(2)
    if dst.exists():return dict(name=name,status='present',bytes=dst.stat().st_size)
    completed=set(json.loads(state.read_text()).get('completed',[])) if state.exists() else set()
    if not completed:
        existing_size=part.stat().st_size if part.exists() else 0
        required=max(0,size-existing_size)
        if shutil.disk_usage(R).free<required+900_000_000:raise RuntimeError('Not enough free disk to finish '+name)
        with part.open('r+b' if part.exists() else 'wb') as f:f.truncate(size)
        state.write_text(json.dumps(dict(size=size,completed=[])))
    chunks=list(range((size+CHUNK-1)//CHUNK))
    def worker(i):
        lo=i*CHUNK;hi=min(size,(i+1)*CHUNK)-1
        for attempt in range(5):
            try:
                res=session().get(url,headers={'Range':f'bytes={lo}-{hi}'},timeout=(15,60));res.raise_for_status()
                if res.status_code!=206 or res.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{size}' or len(res.content)!=hi-lo+1:raise ValueError('Range response mismatch')
                with part.open('r+b') as f:f.seek(lo);f.write(res.content)
                return i
            except Exception:
                if attempt==4:raise
                time.sleep(attempt+1)
    failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        futures=[ex.submit(worker,i) for i in chunks if i not in completed]
        for future in concurrent.futures.as_completed(futures):
            try:completed.add(future.result())
            except Exception as e:
                failures.append(str(e));print('Chunk retry exhausted:',str(e),flush=True);continue
            tmp=state.with_suffix(state.suffix+'.tmp');tmp.write_text(json.dumps(dict(size=size,completed=sorted(completed))));tmp.replace(state)
            if len(completed)%20==0:print(name,len(completed),'/',len(chunks),flush=True)
    if failures:raise RuntimeError(f'{len(failures)} chunks still missing; resume is safe. First error: {failures[0]}')
    with zipfile.ZipFile(part) as z:
        bad=z.testzip()
        if bad:raise ValueError('CRC failed '+bad)
        entries=len(z.infolist());uncompressed=sum(x.file_size for x in z.infolist())
    part.replace(dst)
    row=dict(url=url,path=str(dst.relative_to(R)),status='downloaded_crc_verified',bytes=size,sha256=hashlib.file_digest(dst.open('rb'),'sha256').hexdigest(),entries=entries,uncompressed_bytes=uncompressed)
    dst.with_suffix('.verified.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True);return row
if __name__=='__main__':
    rows=[]
    for name in ['GynSurg_Anatomy_Dataset','GynSurg_Auxiliary_Tool_Dataset','GynSurg_Instrument_Dataset','Instrument_Anatomy_Original_Dataset']:
        try:rows.append(download(name))
        except Exception as e:rows.append(dict(name=name,status='blocked',error=str(e)));print(rows[-1],flush=True)
        (R/'range_download_status.json').write_text(json.dumps(rows,indent=2))
