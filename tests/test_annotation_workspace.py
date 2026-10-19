import unittest,io,base64
import numpy as np
from PIL import Image
from scripts.annotation_workspace import validate_submission

class AnnotationWorkspace(unittest.TestCase):
    def body(self,mask,state='visible',complete=True):
        out=io.BytesIO();Image.fromarray(mask).save(out,format='PNG')
        return dict(task_id=0,annotator='synthetic test operator',visibility=state,annotation_complete=complete,
                    mask_png='data:image/png;base64,'+base64.b64encode(out.getvalue()).decode())
    def tasks(self):return [dict(width='8',height='6')]
    def test_visible_and_absent_consistency(self):
        positive=np.ones((6,8),np.uint8)*255
        _,png,fields=validate_submission(self.body(positive),self.tasks())
        self.assertTrue(fields['annotation_complete']);self.assertTrue(png.startswith(b'\x89PNG'))
        with self.assertRaises(ValueError):validate_submission(self.body(positive,'absent'),self.tasks())
        with self.assertRaises(ValueError):validate_submission(self.body(np.zeros((6,8),np.uint8)),self.tasks())
        validate_submission(self.body(np.zeros((6,8),np.uint8),'absent'),self.tasks())
    def test_incomplete_and_wrong_coordinates_rejected(self):
        with self.assertRaises(ValueError):validate_submission(self.body(np.zeros((6,8),np.uint8),'indeterminate'),self.tasks())
        with self.assertRaises(ValueError):validate_submission(self.body(np.ones((7,8),np.uint8)*255),self.tasks())
        with self.assertRaises(ValueError):validate_submission(self.body(np.ones((6,8),np.uint8)*128),self.tasks())
    def test_name_and_transparency_required(self):
        b=self.body(np.ones((6,8),np.uint8)*255);b['annotator']=''
        with self.assertRaises(ValueError):validate_submission(b,self.tasks())
        with self.assertRaises(ValueError):validate_submission(self.body(np.zeros((6,8,4),np.uint8),'absent'),self.tasks())

if __name__=='__main__':unittest.main()
