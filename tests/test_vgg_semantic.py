"""VGG-specific semantic coordinates, legacy equivalence and CPU lifecycle."""
import contextlib
import gc
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'system'))
from flcore.trainmodel.vgg import VGG11, VGG16
from flcore.trainmodel.mininet_models import MininetCNN
from flcore.heterogeneity.pruning import prune_model, aggregate_submodels
from flcore.heterogeneity.semantic import Bank, extract
from flcore.servers.serverherafl import HERAFL
import test_heterogeneous_algorithms as native


class VGGSemanticTests(unittest.TestCase):
    def test_full_width_features_legacy_path_rng_and_checkpoint_keys(self):
        for cls in (VGG11, VGG16):
            with self.subTest(model=cls.__name__):
                model = cls(num_classes=3)
                x = torch.randn(2, 3, 32, 32)
                self.assertTrue(all(k.startswith('layers.') for k in model.state_dict()))
                for training in (False, True):
                    model.train(training)
                    with torch.no_grad():
                        torch.manual_seed(123)
                        expected = MininetCNN.forward(model, x)
                        torch.manual_seed(123)
                        with patch.object(model, 'augment_inputs', wraps=model.augment_inputs) as augment:
                            actual, features = model.forward_with_features(x)
                        self.assertEqual(augment.call_count, 1)
                        self.assertEqual(tuple(features.shape), (2, 4096))
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                model.eval()
                full, _ = prune_model(model, 1., preserve_representation=True)
                with torch.no_grad():
                    torch.testing.assert_close(full(x), model(x), rtol=0, atol=0)
                full.load_state_dict(model.state_dict(), strict=True)
                del full, model
                gc.collect()

    def test_pruned_feature_identity_flatten_backward_prototypes_and_aggregation(self):
        for cls in (VGG11, VGG16):
            with self.subTest(model=cls.__name__):
                torch.manual_seed(32)
                model = cls(num_classes=3).eval()
                linears = [n for n,l in model.named_modules() if isinstance(l, torch.nn.Linear)]
                convs = [n for n,l in model.named_modules() if isinstance(l, torch.nn.Conv2d)]
                x = torch.randn(2, 3, 32, 32)
                for rate in (.0625, .125):
                    sub, coords = prune_model(model, rate, preserve_representation=True)
                    self.assertEqual(sub.fc.in_features, 4096)
                    self.assertLess(sum(p.numel() for p in sub.parameters()), sum(p.numel() for p in model.parameters()))
                    torch.testing.assert_close(coords[linears[-2]+'.bias'][0], torch.arange(4096))
                    torch.testing.assert_close(coords[linears[-1]+'.weight'][1], torch.arange(4096))
                    # Five pools leave one spatial location per last-convolution channel.
                    torch.testing.assert_close(coords[linears[0]+'.weight'][1], coords[convs[-1]+'.bias'][0])
                    torch.testing.assert_close(coords[linears[-2]+'.weight'][1], coords[linears[0]+'.bias'][0])
                    sub.train()
                    optimizer = torch.optim.Adam(sub.parameters(), lr=1e-4)
                    logits, features = sub.forward_with_features(x)
                    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
                    self.assertTrue(all(p.grad is not None for p in sub.parameters()))
                    optimizer.step()
                    data = list(zip(x, torch.tensor([0, 1])))
                    payload = extract(sub, DataLoader(data, batch_size=2), 3, 0, 1, rate, Bank(dimension=4096), 'cpu')
                    self.assertTrue(sub.training)
                    self.assertEqual(payload.counts, {0:1, 1:1})
                    self.assertEqual(payload.dimension, 4096)
                    sub.eval()
                    with torch.no_grad():
                        expected = sub.forward_features(x)
                    torch.testing.assert_close(payload.prototypes[0], expected[0])
                    self.assertFalse(payload.prototypes[0].requires_grad)
                    result = aggregate_submodels(model, [(sub, coords, 2)])
                    # Check each layer independently, without dense model-sized masks.
                    for name, parameter in model.named_parameters():
                        axes = coords[name]
                        indices = (axes[0][:, None], axes[1][None, :]) if len(axes)==2 else axes
                        torch.testing.assert_close(result.get_parameter(name)[indices], sub.get_parameter(name))
                        mask = torch.zeros_like(parameter, dtype=torch.bool)
                        mask[indices] = True
                        torch.testing.assert_close(result.get_parameter(name)[~mask], parameter[~mask])
                    del sub, optimizer, result, payload, logits, features, mask
                    gc.collect()
                del model
                gc.collect()

    def test_two_round_semantic_server_for_both_vggs(self):
        helper = native.NativeServerTests()
        for cls in (VGG11, VGG16):
            with self.subTest(model=cls.__name__):
                torch.manual_seed(71)
                data = [(torch.randn(3,32,32), torch.tensor(i%3)) for i in range(6)]
                args = helper.make_args(algorithm='HERAFL', model=cls(num_classes=3),
                                        num_clients=2, join_ratio=1., global_rounds=2,
                                        hera_semantic=True, pruning_arms=[.0625,.125],
                                        aligned_comparison=True,
                                        wait_for_all=True, local_learning_rate=1e-4)
                original_assign = HERAFL.assign_submodel
                def assign(server, client, context, pending):
                    original_choose = server.bandit.choose_semantic
                    def choose(c, r, costs, fallback):
                        for a in costs:
                            costs[a]['feasible'] = a == (.0625 if client.id == 0 else .125)
                        return original_choose(c, r, costs, fallback)
                    with patch.object(server.bandit, 'choose_semantic', choose):
                        return original_assign(server, client, context, pending)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch('flcore.servers.serverbase.read_client_data', return_value=data))
                    stack.enter_context(patch('flcore.clients.clientbase.read_client_data', return_value=data))
                    for name in ('save_results','save_global_model','save_policy_results'):
                        stack.enter_context(patch.object(HERAFL,name))
                    stack.enter_context(patch.object(HERAFL,'assign_submodel',assign))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    server=HERAFL(args,0)
                    before=server.global_model.fc.weight.detach().clone()
                    server.train()
                self.assertFalse(torch.equal(before, server.global_model.fc.weight))
                self.assertEqual(server.semantic_bank.dimension,4096)
                self.assertEqual(server.semantic_bank.round,2)
                for report in server.round_reports:
                    self.assertEqual({r['retention'] for r in report['clients']},{.0625,.125})
                    for client in report['clients']:
                        self.assertEqual(client['status'],'accepted')
                        self.assertEqual(client['semantic_payload']['reference_round'],report['round']-1)
                        self.assertEqual(client['selection_context'],client['learning_context'])
                        self.assertGreater(client['round_stats']['semantic_upload_bytes'],0)
                        costs=client['alpha_info']['candidate_costs'][client['retention']]
                        self.assertEqual(costs['predicted_memory'],5*costs['bytes'])
                self.assertTrue(all(c.optimizer is None for c in server.clients))
                del server,args
                gc.collect()

    def test_generic_mininet_remains_unsupported(self):
        with self.assertRaisesRegex(ValueError,'Semantic representation'):
            prune_model(MininetCNN('mnist'),.5,preserve_representation=True)
