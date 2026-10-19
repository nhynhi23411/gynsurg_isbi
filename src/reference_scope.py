"""Separate clinical review from a documented official-release benchmark scope."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def truth(value):return str(value).lower()=='true'
def reference_scope_ready(meta,root=ROOT):
    if truth(meta.get('clinical_scope_reviewed',False)):return True
    if str(meta.get('reference_review_mode',''))!='official_release_computational':return False
    if not truth(meta.get('official_reference_scope_verified',False)):return False
    if str(meta.get('official_annotation_completeness',''))!='dataset_defined':return False
    try:
        path=(Path(root)/str(meta['reference_scope_evidence'])).resolve()
        if not path.is_relative_to(Path(root).resolve()):return False
        if hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=str(meta['reference_scope_evidence_sha256']):return False
        proof=json.loads(path.read_text(encoding='utf8'))
        return (proof.get('mode')=='official_release_computational' and proof.get('engineering_scope_review_complete') is True
                and proof.get('clinical_review_complete') is False and proof.get('new_annotations_created')==0
                and proof.get('targets')==['uterus','autolaparo_annotated_tools_union'])
    except (KeyError,OSError,ValueError):return False
