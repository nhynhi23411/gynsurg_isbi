"""Fetch small query/source evidence panels for actual visual source QA."""
from pathlib import Path
import csv,json,concurrent.futures,argparse
import modal
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]

def main():
    a=argparse.ArgumentParser();a.add_argument('--base',default='data/manifests/source_identity_recovery_v2');a.add_argument('--output',default='materials/verification/task1_source_visual_review');a.add_argument('--prior-failures');args=a.parse_args()
    base=ROOT/args.base
    rows=list(csv.DictReader((base/'candidate_mapping.csv').open(encoding='utf8')))
    by_video={}
    for r in rows:by_video.setdefault(r['source_video_id'],[]).append(r)
    selected=[];selection={}
    for vid,items in sorted(by_video.items()):
        ranked=sorted(items,key=lambda r:float(r['mean_rgb_mae']))
        chosen=[ranked[-1],ranked[len(ranked)//2]]
        unique={r['clip_id']:r for r in chosen};selection[vid]=list(unique);selected.extend(unique.values())
    # Include every failed identity candidate for resolution, not just visual exemplars.
    failed=[r for r in rows if r['pixel_identity_pass'].lower()!='true']
    if args.prior_failures:
        old=list(csv.DictReader((ROOT/args.prior_failures).open(encoding='utf8')))
        ids={r['clip_id'] for r in old if r['pixel_identity_pass'].lower()!='true'}
        failed=list({r['clip_id']:r for r in failed+[r for r in rows if r['clip_id'] in ids]}.values())
    selected=list({r['clip_id']:r for r in selected+failed}.values())
    out=ROOT/args.output;panels=out/'panels';panels.mkdir(parents=True,exist_ok=True)
    volume=modal.Volume.from_name('isbi-gynsurg-task1-recovery-v1',create_if_missing=False)
    def fetch(row):
        path=panels/(row['source_video_id']+'_'+row['clip_id']+'.png')
        if not path.exists():path.write_bytes(b''.join(volume.read_file(row['evidence_path'])))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(fetch,selected))
    for vid,ids in selection.items():
        sheet=Image.new('RGB',(768,174*len(ids)),(245,245,245));draw=ImageDraw.Draw(sheet)
        for index,cid in enumerate(ids):
            r=next(x for x in rows if x['clip_id']==cid)
            draw.text((5,index*174+5),f"Source {vid} / clip {cid} | query top, mirror bottom | MAE {float(r['mean_rgb_mae']):.5f}",fill='black')
            with Image.open(panels/(vid+'_'+cid+'.png')) as im:sheet.paste(im.convert('RGB'),(0,index*174+28))
        sheet.save(out/(vid+'_review.png'))
    manifest=dict(source_panels=selection,failed_clip_ids=[r['clip_id'] for r in failed],
        selection='highest raw MAE and median raw MAE per inferred source, plus all failed identities',
        clinical_accuracy_evaluation=False,visual_audit_complete=False)
    (out/'review_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    # Fixed outlier gallery includes earlier failures even when new verification passes.
    for page in range((len(failed)+3)//4):
        block=failed[page*4:(page+1)*4];sheet=Image.new('RGB',(768,174*len(block)),(245,245,245));draw=ImageDraw.Draw(sheet)
        for index,r in enumerate(block):
            draw.text((5,index*174+5),f"Prior outlier clip {r['clip_id']} / source {r['source_video_id']} | query top, source bottom",fill='black')
            with Image.open(panels/(r['source_video_id']+'_'+r['clip_id']+'.png')) as im:sheet.paste(im.convert('RGB'),(0,index*174+28))
        sheet.save(out/f'outliers_{page+1}.png')
    print('Prepared',len(selection),'source review sheets and',len(failed),'failed-candidate panels')

if __name__=='__main__':main()
