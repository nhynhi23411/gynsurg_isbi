import unittest,json,tempfile,hashlib
from pathlib import Path
from src.reference_scope import reference_scope_ready
class OfficialScope(unittest.TestCase):
    def test_evidence_hash_and_explicit_mode_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);proof=root/'proof.json'
            proof.write_text(json.dumps(dict(mode='official_release_computational',engineering_scope_review_complete=True,clinical_review_complete=False,new_annotations_created=0,targets=['uterus','autolaparo_annotated_tools_union'])))
            meta=dict(clinical_scope_reviewed=False,reference_review_mode='official_release_computational',official_reference_scope_verified=True,official_annotation_completeness='dataset_defined',reference_scope_evidence='proof.json',reference_scope_evidence_sha256=hashlib.file_digest(proof.open('rb'),'sha256').hexdigest())
            self.assertTrue(reference_scope_ready(meta,root))
            self.assertFalse(reference_scope_ready(dict(meta,reference_scope_evidence_sha256='wrong'),root))
            self.assertFalse(reference_scope_ready(dict(meta,reference_review_mode='unreviewed'),root))
            self.assertFalse(reference_scope_ready(dict(meta,official_reference_scope_verified=False),root))
            self.assertFalse(reference_scope_ready(dict(meta,reference_scope_evidence='../outside.json'),root))
if __name__=='__main__':unittest.main()
