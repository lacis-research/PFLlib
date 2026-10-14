"""Regression checks against the local Mininet reference without its runtime dependencies."""
import ast
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'mininetfed_iwcmc'
sys.path.insert(0, str(ROOT / 'system'))
from flcore.heterogeneity.profiles import profiles_for_algorithm
from flcore.heterogeneity.selection import StratifiedRandom
from flcore.heterogeneity.pruning import prune_model, aggregate_submodels
from flcore.trainmodel.mininet_models import MininetCNN
from flcore.trainmodel.models import FedAvgCNN
from flcore.clients.clientfedload import clientFedLoad


def extracted(path, names, namespace=None):
    src = path.read_text()
    nodes = ast.parse(src).body
    code = '\n\n'.join(ast.get_source_segment(src, n) for n in nodes if getattr(n, 'name', None) in names)
    context = dict(np=np, **(namespace or {}))
    exec(code, context)
    return context


class FidelityTests(unittest.TestCase):
    def test_profiles_match_both_experiments_exactly(self):
        for method, hera in [('fedload', False), ('hera_fl', True)]:
            source = extracted(REFERENCE / 'experiments' / method / 'main.py', ['normalize', 'generate_client_profiles'])
            expected = {str(i): e['client_profile'] for i, e in enumerate(source['generate_client_profiles'](30, 42))}
            self.assertEqual(profiles_for_algorithm(30, 42, hera), expected)

    def test_paper_resource_preset_matches_table_one(self):
        profiles = profiles_for_algorithm(300, 42, True, preset='paper')
        self.assertEqual({p['max_throughput_mbps'] for p in profiles.values()}, {75.0, 25.0, 5.0})
        self.assertEqual({p['bandwidth_window'] for p in profiles.values()}, {10})
        self.assertEqual({p['e_cmp_w'] for p in profiles.values()}, {7.0, 4.0, 3.0})
        for field in ('e_up_w', 'e_down_w'):
            self.assertEqual({p[field] for p in profiles.values()}, {2.0, 1.8, 1.7})
        for profile in profiles.values():
            self.assertTrue(1 <= profile['sampled_throughput_mbps'] <= profile['max_throughput_mbps'])
            self.assertTrue(1 <= profile['cpu_frequency_ghz'] <= 2)
            self.assertTrue(5 <= profile['compute_cycles_per_sample'] <= 30)

    def test_runtime_sampling_matches_reference(self):
        src = (REFERENCE / 'mininetfed/client/trainers/mnist.py').read_text()
        cls = next(n for n in ast.parse(src).body if getattr(n, 'name', None) == 'TrainerMNIST')
        method = next(n for n in cls.body if getattr(n, 'name', None) == 'sample_system_state')
        context = {'np': np}
        exec(ast.get_source_segment(src, method), context)
        profile = profiles_for_algorithm(30, 42, True)['0']
        a, b = clientFedLoad.__new__(clientFedLoad), clientFedLoad.__new__(clientFedLoad)
        for client in (a, b):
            client.id = 0
            client.configure_profile(profile)
            client.round_index = 0
        for _ in range(5):
            self.assertEqual(a.sample_system_state(), context['sample_system_state'](b))



    def test_reference_cnn_pruning_and_forward(self):
        model = MininetCNN('mnist')
        layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]
        self.assertEqual([layer.out_channels for layer in layers], [64, 128, 128])
        sub, coordinates = prune_model(model, 0.4)
        self.assertEqual(tuple(sub(torch.rand(2, 1, 28, 28)).shape), (2, 10))
        recovered = aggregate_submodels(model, [(sub, coordinates, 10)])
        for a, b in zip(model.parameters(), recovered.parameters()):
            torch.testing.assert_close(a, b)

    def test_native_cifar100_submodel_trains_with_all_classes(self):
        model = FedAvgCNN(in_features=3, dim=1600, num_classes=100)
        submodel, coordinates = prune_model(model, 0.4)
        output = submodel(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 100))
        loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 99]))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in submodel.parameters()))
        restored = aggregate_submodels(model, [(submodel, coordinates, 2)])
        self.assertEqual(restored.fc.out_features, 100)

    def test_context_reflects_native_partition_size_heterogeneity(self):
        client = clientFedLoad.__new__(clientFedLoad)
        client.id = 0
        client.train_samples = 150
        client.round_stats = {}
        client.configure_profile(profiles_for_algorithm(3, 42)['0'])
        self.assertEqual(client.get_context(300)['dataset_size'], 0.5)
        client.train_samples = 300
        self.assertEqual(client.get_context(300)['dataset_size'], 1.0)

    def test_stratified_selection_preserves_reference_order(self):
        import random, math
        context = extracted(REFERENCE / 'mininetfed/server/selection/stratified_random.py', ['StratifiedRandom'], dict(random=random, math=math))
        selectors = [StratifiedRandom(), context['StratifiedRandom']()]
        profiles = {i: {'cpu_capacity': i/29} for i in range(30)}
        for selector in selectors:
            selector.selection_fraction = 0.1
            selector.set_contexts(profiles, profiles)
        for _ in range(10):
            self.assertEqual(selectors[0].select_trainers_for_round(list(range(30)), {}), selectors[1].select_trainers_for_round(list(range(30)), {}))


if __name__ == '__main__':
    unittest.main()
