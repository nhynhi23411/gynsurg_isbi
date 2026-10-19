"""Guard the actual missing-sibling-module deployment regression."""
import ast,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class ExpansionPackagingTest(unittest.TestCase):
    def setUp(self):
        self.tree=ast.parse((ROOT/'scripts/modal_train_expansion_recovery.py').read_text(encoding='utf8'))

    def test_no_sibling_launcher_import_and_serialized_workers(self):
        for node in ast.walk(self.tree):
            if isinstance(node,ast.ImportFrom):
                self.assertFalse((node.module or '').startswith('modal_'))
            if isinstance(node,ast.Import):
                self.assertFalse(any(x.name.startswith('modal_') for x in node.names))
        for name in ['execute','runtime_preflight']:
            fn=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            decorator=next(d for d in fn.decorator_list if isinstance(d,ast.Call))
            kwargs={k.arg:k.value for k in decorator.keywords}
            self.assertTrue(ast.literal_eval(kwargs['serialized']))
            if name=='execute':self.assertTrue(ast.literal_eval(kwargs['single_use_containers']))

    def test_worker_does_not_reference_external_digest_helper(self):
        fn=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
        self.assertTrue(any(isinstance(n,ast.FunctionDef) and n.name=='digest' for n in fn.body))
        local=next(n for n in self.tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=='modal.is_local()')
        self.assertTrue(any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='image' for t in n.targets) for n in local.body))
        self.assertEqual(ast.unparse(local.orelse[0]),'image = None')
