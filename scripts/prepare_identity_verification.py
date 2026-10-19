"""Match the video verifier's exact OpenCV RGB/area resampling convention."""
from pathlib import Path
import hashlib,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    cv2.setNumThreads(1);queries=[]
    for clip in range(1,301):
        seq=[]
        for frame in [1,25,50,75,100,125]:
            p=ROOT/f'data/raw/AutoLaparo_Task3/imgs/{clip:03}{frame:03}.jpg'
            bgr=cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_COLOR)
            if bgr.shape!=(1080,1920,3):raise ValueError('Unexpected reference coordinates')
            seq.append(cv2.cvtColor(cv2.resize(bgr,(128,72),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
        queries.append(seq)
    path=ROOT/'data/manifests/task3_identity_verification_cv2_v2.npz'
    np.savez_compressed(path,rgb=np.array(queries),clip_ids=np.arange(1,301))
    path.with_suffix('.provenance.json').write_text(json.dumps(dict(sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
        resize='OpenCV INTER_AREA',channels='RGB',width=128,height=72,frames=1800,
        source_hash_manifest='data/manifests/modal_task3_files.json',
        note='v1 PIL bilinear query differs from video AREA resampling; v2 matches both sides without changing acceptance thresholds'),indent=2))
    print('Prepared consistent area-resampled verification queries')

if __name__=='__main__':main()
