import argparse,csv,hashlib
from pathlib import Path
from PIL import Image
import numpy as np
def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--output',required=True);args=a.parse_args()
    root=Path(args.root).resolve();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    with out.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['path','bytes','sha256','width','height','mode','mask_values_if_named_mask','error'])
        for p in sorted(root.rglob('*')):
            if not p.is_file():continue
            h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();width=height=0;mode=values=err=''
            if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.tif','.tiff']:
                try:
                    with Image.open(p) as im:
                        width,height=im.size;mode=im.mode;im.load()
                        if 'mask' in str(p).lower():values=str(np.unique(np.asarray(im)).tolist())[:1000]
                except Exception as e:err=str(e)
            w.writerow([str(p.relative_to(root)),p.stat().st_size,h,width,height,mode,values,err])
if __name__=='__main__':main()
