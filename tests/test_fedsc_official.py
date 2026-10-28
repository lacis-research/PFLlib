"""CUDA parity against the user's local official checkout, not rederived formulas."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'system'))
from flcore.heterogeneity.prototype_protocol import semantic_losses, build_reference, empty_reference
from flcore.heterogeneity.semantic import Payload


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
class OfficialParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        upstream = ROOT.parent/'FedSC'
        if not upstream.exists():
            raise unittest.SkipTest('Local ../FedSC checkout required for parity')
        spec = importlib.util.spec_from_file_location('official_infonce', upstream/'utils/infonce_lcl.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.loss = module.InfoNCE_LCL(.05)
        tree = ast.parse((upstream/'main_fedsc.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Global')
        namespace = dict(torch=torch, np=np, math=math, copy=copy)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(upstream/'main_fedsc.py'), 'exec'), namespace)
        cls.global_class = namespace['Global']

    def test_values_and_gradients_with_repeated_and_missing_labels(self):
        for labels in ([0, 1, 0, 2], [1, 0], [2, 2]):
            torch.manual_seed(12)
            features = torch.randn(len(labels), 7, device='cuda', requires_grad=True)
            relations = {0: torch.randn(2, 7, device='cuda'), 1: torch.randn(3, 7, device='cuda')}
            prototypes = {k: v.mean(0) for k, v in relations.items()}
            positive, targets, negatives = [], [], []
            for i, label in enumerate(labels):
                if label in relations:
                    positive.append(relations[label].mean(0))
                    targets.append(prototypes[label])
                else:
                    positive.append(features[i].detach())
                    targets.append(features[i].detach())
                negatives.extend(v for k, v in relations.items() if k != label)
            expected_rpcl = self.loss.infonce_lcl_loss(features, positive, negatives)
            expected_cpdr = torch.nn.functional.mse_loss(features, torch.stack(targets))
            actual_cpdr, actual_rpcl = semantic_losses(features, torch.tensor(labels, device='cuda'),
                                                       dict(empty_reference(7), relations=relations, prototypes=prototypes))
            torch.testing.assert_close(actual_cpdr, expected_cpdr)
            torch.testing.assert_close(actual_rpcl, expected_rpcl)
            for actual, expected in [(actual_cpdr, expected_cpdr), (actual_rpcl, expected_rpcl)]:
                ga, = torch.autograd.grad(actual, features, retain_graph=True)
                ge, = torch.autograd.grad(expected, features, retain_graph=True)
                torch.testing.assert_close(ga, ge)

    def test_bank_matches_executed_upstream_including_missing_class_positions(self):
        for counts in ([{0: 5, 1: 2}, {0: 2, 1: 4}, {0: 6, 1: 1}],
                       [{0: 5}, {1: 4}, {0: 2, 1: 5}]):
            torch.manual_seed(4)
            payloads = [Payload(cid, 1, .8, 7, count, {k: torch.randn(7, device='cuda') for k in count},
                                dict.fromkeys('HUDQ', .5), dict.fromkeys('HUDQ', True), 1., 0)
                        for cid, count in zip((4, 19, 23), counts)]
            upstream = self.global_class.__new__(self.global_class)
            upstream.args = SimpleNamespace(num_classes=2, device='cuda')
            re_p = upstream.get_re_p(copy.deepcopy([p.prototypes for p in payloads]))
            discrepancies = upstream.get_dis_weights([[c.get(k, 0) for k in range(2)] for c in counts])
            co_p = upstream.get_co_p(re_p, discrepancies, [sum(c.values()) for c in counts])
            actual = build_reference(payloads, empty_reference(7), 1, 2, 'relational', device='cuda')
            for label in co_p:
                torch.testing.assert_close(actual['prototypes'][label].cuda(), co_p[label])
                torch.testing.assert_close(actual['relations'][label].cuda(), torch.stack(re_p[label]))

    def test_empty_reference_and_disabled_rpcl(self):
        x = torch.randn(3, 7, device='cuda', requires_grad=True)
        y = torch.zeros(3, device='cuda', dtype=torch.long)
        a, b = semantic_losses(x, y, empty_reference(7))
        self.assertEqual(float(a+b), 0.)
        reference = dict(empty_reference(7), prototypes={0: torch.ones(7, device='cuda')},
                         relations={0: torch.ones(1, 7, device='cuda')})
        a, b = semantic_losses(x, y, reference, compute_contrastive=False)
        self.assertEqual(float(b), 0.)
        self.assertTrue(torch.isfinite(a))
        with self.assertRaisesRegex(ValueError, 'nonempty negative pool'):
            semantic_losses(x, y, reference)


if __name__ == '__main__':
    unittest.main()
