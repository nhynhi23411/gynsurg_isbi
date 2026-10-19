"""Local, model-blind human annotation workspace. Never auto-approves references.

Run with the project venv; serves only 127.0.0.1. Saves versioned working masks
outside raw data. Clinical review/adjudication and scope approval stay separate.
"""
from pathlib import Path
import argparse,base64,csv,datetime,hashlib,io,json,re
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'clinical_annotation/tail_reference_v1'

def validate_submission(body,tasks):
    task_id=body.get('task_id')
    if type(task_id) is not int or not 0<=task_id<len(tasks):raise ValueError('Invalid task')
    task=tasks[task_id];annotator=str(body.get('annotator','')).strip()
    if not annotator:raise ValueError('Enter the actual annotator name')
    state=body.get('visibility');complete=body.get('annotation_complete')
    if state not in ['visible','absent','indeterminate','unannotated'] or type(complete) is not bool:raise ValueError('Invalid visibility/completeness')
    if complete and state not in ['visible','absent']:raise ValueError('Uncertain/unannotated cannot be complete')
    encoded=body.get('mask_png','')
    if not encoded.startswith('data:image/png;base64,'):raise ValueError('Expected PNG mask')
    raw=base64.b64decode(encoded.split(',',1)[1],validate=True)
    if len(raw)>8_000_000:raise ValueError('Mask too large')
    with Image.open(io.BytesIO(raw)) as im:
        if im.format!='PNG' or im.size!=(int(task['width']),int(task['height'])):raise ValueError('Mask must preserve native image size')
        a=np.asarray(im)
    if a.ndim==3:
        if a.shape[2] not in [3,4] or not np.array_equal(a[:,:,0],a[:,:,1]) or not np.array_equal(a[:,:,0],a[:,:,2]):raise ValueError('Expected grayscale binary mask')
        if a.shape[2]==4 and not (a[:,:,3]==255).all():raise ValueError('Transparent mask pixels are ambiguous')
        a=a[:,:,0]
    if a.ndim!=2 or not set(np.unique(a).tolist())<={0,255}:raise ValueError('Mask must contain only 0/255')
    if state=='absent' and a.any():raise ValueError('Absent requires an explicitly empty mask')
    if state=='visible' and complete and not a.any():raise ValueError('Visible complete mask is empty')
    clean=io.BytesIO();Image.fromarray(a).save(clean,format='PNG')
    return task,clean.getvalue(),dict(annotator=annotator,visibility=state,
        annotation_complete=complete,notes=str(body.get('notes',''))[:10000])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8768);args=parser.parse_args()
    with (BASE/'annotation_tasks.csv').open(encoding='utf8') as f:tasks=list(csv.DictReader(f))
    working=BASE/'working_annotations_v1';working.mkdir(exist_ok=True)
    class Handler(BaseHTTPRequestHandler):
        def send(self,data,mime='application/json',status=200):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode('utf8')
            self.send_response(status);self.send_header('Content-Type',mime+'; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/':return self.send((ROOT/'clinical_annotation/annotation_workspace.html').read_bytes(),'text/html')
            if self.path=='/api/tasks':return self.send([dict(id=i,clip_id=t['clip_id'],frame_index=t['frame_index'],structure=t['structure_name'],image='/image/'+Path(t['image']).name) for i,t in enumerate(tasks)])
            if re.fullmatch(r'/api/latest/\d+',self.path):
                task_id=int(self.path.split('/')[-1])
                if not 0<=task_id<len(tasks):return self.send({'error':'Invalid task'},status=400)
                t=tasks[task_id];prefix=f"{t['clip_id']}_{int(t['frame_index']):05d}_k{t['structure']}_"
                records=sorted(working.glob(prefix+'*.json'))
                if not records:return self.send({'exists':False})
                record=json.loads(records[-1].read_text(encoding='utf8'));record['mask_url']='/mask/'+Path(record['mask']).name
                return self.send(dict(exists=True,record=record))
            if re.fullmatch(r'/mask/\d{3}_\d{5}_k[01]_\d{8}T\d{12}Z\.png',self.path):
                path=working/self.path.split('/')[-1]
                if path.is_file():return self.send(path.read_bytes(),'image/png')
            if re.fullmatch(r'/image/\d{3}_\d{5}\.jpg',self.path):
                path=BASE/'images'/self.path.split('/')[-1]
                if path.is_file():return self.send(path.read_bytes(),'image/jpeg')
            return self.send({'error':'Not found'},status=404)
        def do_POST(self):
            if self.path!='/api/save':return self.send({'error':'Not found'},status=404)
            origin=self.headers.get('Origin')
            if origin and origin not in [f'http://127.0.0.1:{args.port}',f'http://localhost:{args.port}']:return self.send({'error':'Origin rejected'},status=403)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=12_000_000:raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(length));task,png,fields=validate_submission(body,tasks)
                timestamp=datetime.datetime.now(datetime.timezone.utc)
                token=timestamp.strftime('%Y%m%dT%H%M%S%fZ')
                stem=f"{task['clip_id']}_{int(task['frame_index']):05d}_k{task['structure']}_{token}"
                mask=working/(stem+'.png');mask.write_bytes(png)
                record=dict(task_id=body['task_id'],clip_id=task['clip_id'],frame_index=int(task['frame_index']),
                    structure=int(task['structure']),structure_name=task['structure_name'],image=task['image'],
                    image_sha256=hashlib.file_digest(open(task['image'],'rb'),'sha256').hexdigest(),
                    mask=str(mask.resolve()),mask_sha256=hashlib.sha256(png).hexdigest(),
                    saved_at_utc=timestamp.isoformat(),**fields,adjudicator='',clinical_review_status='pending',
                    clinical_scope_reviewed=False,reference_approved=False,provenance='human-drawn in local workspace; identity self-reported')
                (working/(stem+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
                self.send({'saved':True,'record':str((working/(stem+'.json')).resolve()),'reference_approved':False})
            except (ValueError,KeyError,TypeError,Image.UnidentifiedImageError) as e:self.send({'error':str(e)},status=400)
        def log_message(self,format,*values):pass
    print(f'Annotation workspace: http://127.0.0.1:{args.port} ; {len(tasks)} structure tasks; no clinical approvals are granted',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__':main()
