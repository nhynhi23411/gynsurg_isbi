"""Generate review-only sequence metadata and missing-input checklists."""
from pathlib import Path
import csv,json,hashlib,collections
ROOT=Path(__file__).resolve().parents[1]
def write_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    refs=list(csv.DictReader((ROOT/'data/manifests/task3_references_draft.csv').open(encoding='utf8')));groups=collections.defaultdict(list)
    for r in refs:groups[r['clip_id']].append(r)
    out=ROOT/'data/manifests/sequence_review_v1';out.mkdir(parents=True,exist_ok=True);mapping=[]
    for clip,rows in sorted(groups.items()):
        pool=list(range(1,11)) if int(clip)<=170 else (list(range(11,17)) if int(clip)<=227 else list(range(17,22)))
        meta=json.loads((ROOT/'configs/sequence_metadata_template.json').read_text());meta.update(clip_id=clip,procedure_id='',query_ids=[0,24,49,74,99],scoring_ids=[124,149,174,199,224],structure_names=['uterus','autolaparo_annotated_tools_union'],source_dataset='AutoLaparo',official_split='train' if int(clip)<=170 else ('val' if int(clip)<=227 else 'test'),split='test' if int(clip)>=228 else 'val',group_verified=False,exposure_clean=True,exposure_evidence='data/manifests/frozen_checkpoint_exposure_audit_v1.json; data/manifests/task3_transfer_integrity.json',permission_status='verified',permission_scope='academic noncommercial research only',permission_evidence='https://autolaparo.github.io/ and official repository README',ethics_statement_status='institutional confirmation pending; no waiver invented',clinical_scope_reviewed=False,matched_gynsurg_instrument_classes=[1,5,6],human_annotation_complete=False)
        (out/f'{clip}.metadata.draft.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
        reference_rows=[dict(frame_index=r['frame_index'],structure=r['structure'],mask=r['mask'],annotation_complete=r['annotation_complete']) for r in rows]
        reference_rows.extend(dict(frame_index=t,structure=k,mask='',annotation_complete='false') for t in [149,174,199,224] for k in [0,1])
        write_csv(out/f'{clip}.references.draft.csv',reference_rows)
        mapping.append(dict(clip_id=clip,source_dataset='AutoLaparo',source_video_id='',patient_id='',procedure_id='',possible_source_video_ids=';'.join(f'{p:02d}' for p in pool),evidence_path='',verified_by='',verified_date='',mapping_verified='false'))
    write_csv(ROOT/'data/manifests/procedure_mapping_tasks_v1.csv',mapping)
    images=[];master={r['clip_id']:r for r in csv.DictReader((ROOT/'materials/local/followup/manifests/master_manifest_v1.csv').open(encoding='utf8'))}
    for p in sorted((ROOT/'clinical_annotation/tail_reference_v1/images').glob('*.jpg')):
        clip,t=p.stem.split('_');images.append(dict(image=str(p.relative_to(ROOT)),clip_id=clip,frame_index=int(t),bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),source_video_sha256=master[clip]['sha256']))
    if len(images)!=1200:raise ValueError('Tail frame package is incomplete')
    write_csv(ROOT/'clinical_annotation/tail_reference_v1/image_provenance.csv',images)
    print('Prepared',len(mapping),'metadata/reference drafts and',len(images),'annotation image hashes')
if __name__=='__main__':main()
