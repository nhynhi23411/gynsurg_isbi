"""Prepare official reference labels for review, without inventing procedure IDs."""
from pathlib import Path
import sys,csv,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
import numpy as np
from src.autolaparo import decode_targets
ROOT=Path(__file__).resolve().parents[1]
COUNTS=[37,21,27,19,11,22,8,14,0,11,19,0,21,0,0,17,14,26,13,0,20]
def main():
    from src.common import approved
    approved(ROOT);raw=ROOT/'data/raw/AutoLaparo_Task3';out=ROOT/'data/derived/task3_official_v1';out.mkdir(parents=True,exist_ok=True)
    images=sorted((raw/'imgs').glob('*.jpg'));masks=sorted((raw/'masks').glob('*.png'))
    if len(images)!=1800 or len(masks)!=1800:raise ValueError('Wait for complete 1800+1800 transfer')
    rows=[]
    for p in images:
        mp=raw/'masks'/(p.stem+'.png')
        with Image.open(mp) as im:targets=decode_targets(np.asarray(im))
        clip=p.stem[:3];one_based=int(p.stem[3:]);pool='development_source_01_10' if int(clip)<=170 else ('heldout_source_11_16' if int(clip)<=227 else 'heldout_source_17_21')
        for k,name in enumerate(['uterus','primary_tools_union']):
            dest=out/(p.stem+'_'+name+'.png');expected=targets[k].astype(np.uint8)*255
            if dest.exists():
                with Image.open(dest) as im:
                    if not np.array_equal(np.asarray(im),expected):raise ValueError('Existing derived label differs')
            else:Image.fromarray(expected).save(dest)
            rows.append(dict(clip_id=clip,procedure_id='',group_verified='false',source_pool=pool,frame_number_one_based=one_based,frame_index=one_based-1,frame_alignment_status='pending_pixel_alignment_audit',structure=k,structure_name=name,image=str(p.resolve()),mask=str(dest.resolve()),mask_sha256=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest(),annotation_origin='official_Task3_semantic_annotation',annotation_complete='pending_scope_review',clinical_review_status='pending',permission_status='academic_noncommercial_terms_recorded'))
    with (ROOT/'data/manifests/task3_references_draft.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (ROOT/'data/manifests/autolaparo_supplementary_case_counts.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['video_id','task2_clip_count','source']);w.writerows([(f'{i:02d}',n,'materials/papers/AutoLaparo_Supplementary.pdf Table1') for i,n in enumerate(COUNTS,1)])
    summary=dict(images=len(images),binary_target_files=len(rows),target_names=['uterus','primary_tools_union'],raw_palette=sorted(range(0,181,20)),contributing_procedures=sum(n>0 for n in COUNTS),known_source_pool_counts={'01_10':sum(COUNTS[:10]),'11_16':sum(COUNTS[10:16]),'17_21':sum(COUNTS[16:])},exact_clip_procedure_mapping=False,main_evaluation_ready=False)
    (ROOT/'data/manifests/task3_reference_preparation.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
