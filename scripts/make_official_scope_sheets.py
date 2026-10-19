"""Prepare reproducible official-training-pool overlays without model predictions."""
from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    ids=['001001','038125','058001','059125','079001','086125','105001','116125','138001','146125','160001','170125']
    out=ROOT/'materials/verification/official_sparse_scope_v1';out.mkdir(exist_ok=True);rows=[]
    for page in range(3):
        sheet=Image.new('RGB',(960,4*202),'white');draw=ImageDraw.Draw(sheet)
        for row,name in enumerate(ids[page*4:(page+1)*4]):
            ip=ROOT/f'data/raw/AutoLaparo_Task3/imgs/{name}.jpg';mp=ROOT/f'data/raw/AutoLaparo_Task3/masks/{name}.png'
            with Image.open(ip) as im:rgb=np.asarray(im.convert('RGB').resize((320,180)))
            with Image.open(mp) as im:raw=np.asarray(im.resize((320,180),Image.Resampling.NEAREST))[...,0]
            draw.text((5,row*202+4),name+' | source / uterus=180 / all tool parts=20..160',fill='black')
            sheet.paste(Image.fromarray(rgb),(0,row*202+22))
            for col,(mask,color) in enumerate([(raw==180,[0,255,255]),((raw>0)&(raw<180),[255,255,0])],1):
                blend=rgb.copy();blend[mask]=(rgb[mask].astype(float)*.5+np.array(color)*.5).astype(np.uint8)
                sheet.paste(Image.fromarray(blend),(col*320,row*202+22))
            rows.append(dict(image=name,image_sha256=hashlib.file_digest(ip.open('rb'),'sha256').hexdigest(),mask_sha256=hashlib.file_digest(mp.open('rb'),'sha256').hexdigest(),palette=np.unique(raw).tolist()))
        sheet.save(out/f'page_{page+1}.png')
    (out/'selection.json').write_text(json.dumps(dict(selection='fixed representative official training images; no model/metric selection',images=rows,visual_review_complete=False,clinical_review_complete=False),indent=2))
    print('Prepared 3 engineering target-scope sheets, 12 official train images; review pending')
if __name__=='__main__':main()
