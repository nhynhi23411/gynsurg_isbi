"""Record the actual engineering review; never impersonate a clinician."""
from pathlib import Path
import json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1]
def main():
    folder=ROOT/'materials/verification/official_sparse_scope_v1'
    selection=json.loads((folder/'selection.json').read_text())
    # All three sheets were viewed in this chat before this record was created.
    proof=dict(mode='official_release_computational',reviewer='Codex — engineering target/encoding/coordinate QA',
      checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      engineering_scope_review_complete=True,clinical_review_complete=False,new_annotations_created=0,
      targets=['uterus','autolaparo_annotated_tools_union'],
      decoder={'uterus':180,'tool_parts':list(range(20,180,20))},
      model_adapter={'anatomy_classes':[1],'instrument_classes':[1,5,6],'merge_selected':True},
      official_source='https://autolaparo.github.io/',
      alignment_evidence='data/manifests/task3_frame_alignment.summary.json',
      integrity_evidence='data/manifests/task3_transfer_integrity.json',
      engineering_visual_sample=selection['images'],
      visually_reviewed_sheets={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sorted(folder.glob('page_*.png'))},
      completeness_basis='Released semantic annotations for the dataset-defined targets; no new independent completeness/adjudication assessment',
      observations='Overlays align with visible released target regions in 12 fixed training images, including empty-target examples. This is technical target QA, not clinical boundary validation.',
      limitations=['Sparse snapshots only','Cross-dataset harmonization may differ at ambiguous instruments','No current clinical lead review or inter-reader assessment','No claim that unannotated/out-of-scope anatomy is absent'])
    out=folder/'scope_evidence.json'
    if out.exists():raise FileExistsError('Preserve scope review version')
    out.write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf8')
    print('Recorded engineering official-release scope; clinical review remains false')
if __name__=='__main__':main()
