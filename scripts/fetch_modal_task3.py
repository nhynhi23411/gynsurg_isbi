"""Resume a read-only download of existing AutoLaparo labels from the user's volume."""
from pathlib import Path,PurePosixPath
import concurrent.futures,hashlib,json,threading
import modal
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'data/raw/AutoLaparo_Task3'
def main():
    volume=modal.Volume.from_name('gynsurg',create_if_missing=False)
    entries=list(volume.iterdir('/AutoLaparo_Dataset',recursive=True))
    files=[x for x in entries if x.path.lower().endswith(('.png','.jpg','.jpeg','.json','.txt'))]
    print('Remote files:',len(files),flush=True)
    rows=[];lock=threading.Lock()
    def download(e):
        rel=PurePosixPath(e.path.lstrip('/')).relative_to('AutoLaparo_Dataset')
        if '..' in rel.parts:raise ValueError('Unsafe remote path')
        dst=DEST.joinpath(*rel.parts);dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.is_file() or dst.stat().st_size!=e.size:
            tmp=dst.with_suffix(dst.suffix+'.part')
            with tmp.open('wb') as f:
                for chunk in volume.read_file(e.path):f.write(chunk)
            if tmp.stat().st_size!=e.size:raise ValueError('Incomplete transfer '+e.path)
            tmp.replace(dst)
        row={'remote_path':e.path,'path':str(dst.relative_to(ROOT)),'bytes':dst.stat().st_size,'sha256':hashlib.file_digest(dst.open('rb'),'sha256').hexdigest()}
        with lock:
            rows.append(row)
            if len(rows)%200==0:print('Downloaded/verified',len(rows),'/',len(files),flush=True)
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:list(ex.map(download,files))
    (ROOT/'data/manifests/modal_task3_files.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    print('Complete:',len(rows),'files',sum(x['bytes'] for x in rows),'bytes',flush=True)
if __name__=='__main__':main()
