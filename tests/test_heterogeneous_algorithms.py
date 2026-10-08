"""Run: python -m unittest discover -s tests -v (from PFLlib)."""
import sys
import unittest
import copy
import contextlib
import io
from types import SimpleNamespace
from unittest.mock import patch, mock_open
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'system'))
from flcore.trainmodel.models import FedAvgCNN, DNN, Mclr_Logistic
from flcore.heterogeneity.pruning import prune_model, recover_model, aggregate_submodels
from flcore.heterogeneity.linucb_fedload import LinUCBBandit as FedLoadBandit
from flcore.heterogeneity.linucb_hera import LinUCBBandit as HeraBandit
from flcore.heterogeneity.straggler import OnlineStragglerPredictor


class StructuredPruningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_full_retention_preserves_predictions_and_parameters(self):
        for model, x in [(FedAvgCNN(), torch.randn(2, 1, 28, 28)),
                         (DNN(), torch.randn(2, 1, 28, 28)),
                         (Mclr_Logistic(), torch.randn(2, 1, 28, 28))]:
            submodel, coordinates = prune_model(model, 1.0)
            torch.testing.assert_close(submodel(x), model(x))
            recovered = aggregate_submodels(model, [(submodel, coordinates, 3)])
            for original, result in zip(model.parameters(), recovered.parameters()):
                torch.testing.assert_close(original, result)

    def test_cnn_flatten_mapping_and_real_size_reduction(self):
        for channels, spatial, dim in [(1, 16, 1024), (3, 25, 1600)]:
            model = FedAvgCNN(in_features=channels, dim=dim)
            submodel, coordinates = prune_model(model, 0.4)
            outputs = coordinates['conv2.0.bias'][0]
            expected = torch.cat([torch.arange(int(c) * spatial, (int(c) + 1) * spatial) for c in outputs])
            self.assertTrue(torch.equal(coordinates['fc1.0.weight'][1], expected))
            self.assertLess(sum(p.numel() for p in submodel.parameters()), sum(p.numel() for p in model.parameters()))
            x = torch.randn(2, channels, 28 if channels == 1 else 32, 28 if channels == 1 else 32)
            loss = submodel(x).square().mean()
            loss.backward()
            self.assertTrue(all(p.grad is not None for p in submodel.parameters()))
            self.assertEqual(submodel.fc.out_features, 10)

    def test_masked_weighted_average_preserves_uncovered_coordinates(self):
        model = DNN(input_dim=4, mid_dim=4, num_classes=2)
        sub, coordinates = prune_model(model, 0.5)
        sub2, _ = prune_model(model, 0.5)
        with torch.no_grad():
            for p in sub.parameters():
                p.fill_(2)
            for p in sub2.parameters():
                p.fill_(6)
        _, masks = recover_model(sub, coordinates, model)
        result = aggregate_submodels(model, [(sub, coordinates, 1), (sub2, coordinates, 3)])
        for name, original in model.named_parameters():
            received = masks[name].bool()
            torch.testing.assert_close(result.get_parameter(name)[received], torch.full_like(original[received], 5))
            torch.testing.assert_close(result.get_parameter(name)[~received], original[~received])
        untouched = aggregate_submodels(model, [])
        for original, result in zip(model.parameters(), untouched.parameters()):
            torch.testing.assert_close(original, result)

    def test_mixed_retention_updates_only_participating_coordinates(self):
        model = DNN(input_dim=4, mid_dim=4, num_classes=2)
        small, coordinates = prune_model(model, 0.5)
        full, full_coordinates = prune_model(model, 1)
        with torch.no_grad():
            for p in small.parameters():
                p.fill_(2)
            for p in full.parameters():
                p.fill_(6)
        _, masks = recover_model(small, coordinates, model)
        result = aggregate_submodels(model, [(small, coordinates, 1), (full, full_coordinates, 3)])
        for name, p in result.named_parameters():
            expected = torch.where(masks[name].bool(), 5., 6.)
            torch.testing.assert_close(p, expected)

    def test_invalid_models_and_rates_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, 'support native'):
            prune_model(torch.nn.Linear(4, 2), 0.5)
        for rate in (0, -1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                prune_model(DNN(), rate)


class PolicyTests(unittest.TestCase):
    def test_ported_linucb_updates_and_reproducible_selection(self):
        context = dict(f_i=0.8, dataset_size=0.5, c_r_i=0.2, b_mean=0.9)
        for cls in (FedLoadBandit, HeraBandit):
            first, second = cls([1., 0.4]), cls([1., 0.4])
            for _ in range(4):
                arm, _ = first.choose_pruning_rate(context, [])
                other, _ = second.choose_pruning_rate(context, [])
                self.assertEqual(arm, other)
                previous = first.A[arm].copy()
                first.update(arm, 0.8, context)
                second.update(arm, 0.8, context)
                self.assertFalse((previous == first.A[arm]).all())

    def test_hera_guides_smaller_models_to_low_capacity_clients(self):
        policy = HeraBandit([1., 0.8, 0.6, 0.4], alpha=0)
        low = dict(cpu_capacity=0, c_r_i=1, bandwidth=0, energy_efficiency=0, straggler_risk=1)
        high = dict(cpu_capacity=1, c_r_i=0, bandwidth=1, energy_efficiency=1, straggler_risk=0)
        self.assertLess(policy.choose_pruning_rate(low)[0], policy.choose_pruning_rate(high)[0])

    def test_predictor_learns_observed_time(self):
        predictor = OnlineStragglerPredictor()
        context = dict(f_i=1, dataset_size=1, c_r_i=1, b_mean=1)
        for _ in range(20):
            predictor.update(context, 0.8)
        self.assertGreater(predictor.predict_round_time(context), 0.7)
        self.assertEqual(predictor.update_count, 20)


class NativeServerTests(unittest.TestCase):
    def make_args(self, **overrides):
        defaults = dict(device='cpu', dataset='synthetic', num_classes=3,
                        model=DNN(input_dim=4, mid_dim=8, num_classes=3),
                        global_rounds=1, local_epochs=1, batch_size=6,
                        local_learning_rate=0.01, num_clients=3, join_ratio=0.67,
                        random_join_ratio=False, few_shot=0, algorithm='FedLoad',
                        time_select=False, goal='test', time_threthold=10000,
                        save_folder_name='items', top_cnt=100, auto_break=False,
                        eval_gap=1, client_drop_rate=0, train_slow_rate=0,
                        send_slow_rate=0, dlg_eval=False, dlg_gap=1,
                        batch_num_per_client=1, num_new_clients=0,
                        fine_tuning_epoch_new=0, learning_rate_decay=True,
                        learning_rate_decay_gamma=0.9,
                        pruning_arms=[0.5], bandit_alpha=0.25, bandit_beta=0.6,
                        policy_seed=42, reward_weights=[0.6, 0.2, 0.2],
                        client_profiles=None, pruning_round_timeout=300)
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def run_server(self, cls, args):
        data = [(torch.randn(4), torch.tensor(i % 3)) for i in range(18)]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('flcore.servers.serverbase.read_client_data', return_value=data))
            stack.enter_context(patch('flcore.clients.clientbase.read_client_data', return_value=data))
            stack.enter_context(patch.object(cls, 'save_results'))
            stack.enter_context(patch.object(cls, 'save_global_model'))
            stack.enter_context(patch('builtins.open', mock_open()))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            server = cls(args, 0)
            original = copy.deepcopy(server.global_model)
            server.train()
        return server, original

    def test_both_servers_train_evaluate_and_update_policy(self):
        from flcore.servers.serverfedload import FedLoad
        from flcore.servers.serverherafl import HERAFL
        for cls in (FedLoad, HERAFL):
            server, original = self.run_server(cls, self.make_args(algorithm=cls.__name__))
            self.assertEqual(len(server.rs_test_acc), 2)
            self.assertEqual(len(server.round_reports), 2)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(original.parameters(), server.global_model.parameters())))
            for report in server.round_reports:
                self.assertEqual(len(report['clients']), 2)
                self.assertTrue(all(r['status'] == 'accepted' and 'reward' in r for r in report['clients']))
            if cls is HERAFL:
                self.assertEqual(server.predictor.update_count, 4)

    def test_fedavg_logging_preserves_original_training(self):
        import random
        import numpy as np
        from flcore.servers.serveravg import FedAvg
        from flcore.servers.serverprofiledavg import ProfiledFedAvg
        servers = []
        for cls in (FedAvg, ProfiledFedAvg):
            torch.manual_seed(23)
            np.random.seed(12)
            random.seed(13)
            server, _ = self.run_server(cls, self.make_args(algorithm='FedAvg'))
            servers.append(server)
        for original, profiled in zip(servers[0].global_model.parameters(), servers[1].global_model.parameters()):
            torch.testing.assert_close(original, profiled, rtol=0, atol=0)
        self.assertEqual(servers[0].rs_test_acc, servers[1].rs_test_acc)
        self.assertEqual(len(servers[1].round_reports), 2)

    def test_no_responses_preserves_global_and_policy(self):
        from flcore.servers.serverfedload import FedLoad
        for options in (dict(pruning_round_timeout=1e-12), dict(client_drop_rate=0.99)):
            server, original = self.run_server(FedLoad, self.make_args(**options))
            for a, b in zip(original.parameters(), server.global_model.parameters()):
                torch.testing.assert_close(a, b)
            self.assertEqual(sum(server.bandit.arm_counts.values()), 0)
            self.assertTrue(all('reward' not in r for report in server.round_reports for r in report['clients']))


if __name__ == '__main__':
    unittest.main()
