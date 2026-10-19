"""Verified pixel palettes. Annotation completeness is a separate manifest field."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
PALETTES={'instrument':[0,36,73,109,146,182,219,255],'anatomy':[0,85,170,255],'auxiliary':[0,60,120,190,255]}
CLASS_NAMES={'instrument':['background','grasper','scissors','irrigator','bipolar-forceps','sealer-divider','hook','suturing-instrument'],'anatomy':['background','uterus','fallopian-tube','ovary'],'auxiliary':['background','morcellator','thread','trocar-sleeve','cannula']}
def decode(raw,head):
    raw=np.asarray(raw)
    if raw.ndim!=2:raise ValueError('Expected single-channel label raster')
    palette=PALETTES[head];unknown=set(np.unique(raw).tolist())-set(palette)
    if unknown:raise ValueError(f'Unknown {head} values: {sorted(unknown)}')
    lut=np.full(256,255,np.uint8)
    for i,value in enumerate(palette):lut[value]=i
    return lut[raw.astype(np.uint8)]
def main():
    a=argparse.ArgumentParser();a.add_argument('--mask',required=True);a.add_argument('--head',choices=list(PALETTES),required=True);a.add_argument('--class-id',type=int,required=True);a.add_argument('--output',required=True);args=a.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    raw_root=Path(__file__).resolve().parents[1]/'data/raw'
    if out.resolve().is_relative_to(raw_root.resolve()):raise ValueError('Write derived labels outside raw data')
    if args.class_id<=0 or args.class_id>=len(PALETTES[args.head]):raise ValueError('Choose a foreground class ID')
    with Image.open(args.mask) as im:y=decode(np.asarray(im),args.head)
    out.parent.mkdir(parents=True,exist_ok=True);Image.fromarray((y==args.class_id).astype(np.uint8)*255).save(out)
if __name__=='__main__':main()
