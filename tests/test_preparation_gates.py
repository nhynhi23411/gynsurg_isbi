import unittest,tempfile,json
from pathlib import Path
from src.prepare_sequence import validate_metadata,validate_references
from src.common import approved
from src.surgery import validate_execution_gates

class PreparationGates(unittest.TestCase):
    def metadata(self):
        return dict(procedure_id='synthetic-source',clip_id='synthetic-clip',fps=25,
            n_structures=1,input_size=518,mean=[.5]*3,std=[.5]*3,query_ids=[0],scoring_ids=[1],
            group_verified=True,exposure_clean=True,permission_status='verified',
            reference_format='binary_per_structure',split='val',clinical_scope_reviewed=True,
            human_annotation_complete=True)

    def test_unreviewed_scope_and_test_relabelling_rejected(self):
        validate_metadata(self.metadata())
        for key in ['clinical_scope_reviewed','human_annotation_complete']:
            m=self.metadata();m[key]=False
            with self.assertRaises(ValueError):validate_metadata(m)
        m=self.metadata();m['official_split']='test'
        with self.assertRaises(ValueError):validate_metadata(m)

    def test_reference_alias_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            mask=Path(tmp)/'placeholder.png';mask.write_bytes(b'existence-only preflight')
            refs=[dict(frame_index=str(t),structure='0',mask=str(mask),annotation_complete='true') for t in [0,1]]
            validate_references(refs,self.metadata())
            with self.assertRaises(ValueError):validate_references(refs+[refs[0]],self.metadata())
            with self.assertRaises(ValueError):validate_references([dict(refs[0],frame_index='-1'),refs[1]],self.metadata())
            with self.assertRaises(ValueError):validate_references(refs[:1],self.metadata())

    def test_development_authorization_does_not_unlock_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir()
            (root/'configs/review_status.json').write_text(json.dumps(dict(approved_for_training=True,
                approved_for_test=False,reviewer='synthetic',protocol_sha256='synthetic')))
            approved(root)
            with self.assertRaises(RuntimeError):approved(root,test=True)

    def test_suite_and_single_run_share_clinical_gates(self):
        d=dict(split='val',group_verified='true',exposure_clean='true',clinical_scope_reviewed='true',human_annotation_complete='true',permission_status='verified')
        validate_execution_gates(d,'val')
        with self.assertRaises(ValueError):validate_execution_gates(d,'test')
        d['human_annotation_complete']='false'
        with self.assertRaises(ValueError):validate_execution_gates(d,'val')

if __name__=='__main__':unittest.main()
