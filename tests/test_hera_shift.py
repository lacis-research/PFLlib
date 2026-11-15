"""Class coverage, sample conservation and local-model evaluation."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'system'))
from prepare_hera_shift import partition_indices, prepare_from_partitions
from flcore.clients.clientherafl import clientHERAFL


class ShiftPartitionsTests(unittest.TestCase):
    def test_coverage_reproducibility_and_disjoint_partitions(self):
        train_y = np.repeat(np.arange(10), 1000)
        test_y = np.repeat(np.arange(10), 100)
        train, test = partition_indices(train_y, test_y, num_clients=20)
        self.assertEqual((train, test), partition_indices(train_y, test_y, num_clients=20))
        self.assertEqual(len(set(sum(train, []))), len(sum(train, [])))
        self.assertEqual(sorted(sum(test, [])), list(range(len(test_y))))
        expected = [1000] * 10
        np.testing.assert_array_equal(np.bincount(train_y[sum(train, [])]), expected)
        self.assertLessEqual(max(map(len, train)) - min(map(len, train)), 1)
        observed = [len(set(train_y[t])) for t in train]
        self.assertGreater(len(set(observed)), 1)
        self.assertTrue(any(k < 10 for k in observed))
        for te in test:
            np.testing.assert_array_equal(np.bincount(test_y[te]), [5] * 10)

    def test_exact_training_indices_match_original_fedsc_reference(self):
        # Frozen expectations were generated using the untouched original FedSC
        # source, not this implementation. Includes a partial block (17 clients).
        import hashlib
        fixture = json.loads((ROOT / 'tests/fixtures/fedsc_partition.json').read_text())
        labels = np.repeat(np.arange(10), fixture['train_samples_per_class'])
        heldout = np.repeat(np.arange(10), 100)
        for case in fixture['cases']:
            with self.subTest(clients=case['num_clients']):
                train, _ = partition_indices(labels, heldout, num_clients=case['num_clients'])
                digest = hashlib.sha256(json.dumps(train, separators=(',', ':')).encode()).hexdigest()
                self.assertEqual(digest, case['indices_sha256'])
                self.assertEqual([np.bincount(labels[t], minlength=10).tolist() for t in train], case['class_counts'])

    def test_uniform_baseline(self):
        labels = np.repeat(np.arange(4), 400)
        test = np.repeat(np.arange(4), 40)
        options = dict(num_clients=10, num_classes=4, alpha=1.0)
        uniform, _ = partition_indices(labels, test, allocation='uniform', **options)
        shifted, _ = partition_indices(labels, test, **options)
        self.assertNotEqual(uniform, shifted)
        self.assertTrue(all(len(set(labels[t])) == 4 for t in uniform))
        np.testing.assert_array_equal(np.bincount(labels[sum(shifted, [])]), [400] * 4)

    def test_invalid_or_impossible_scenarios_fail(self):
        labels = np.repeat(np.arange(4), 10)
        for options in [dict(num_clients=2), dict(alpha=0), dict(alpha=float('inf')),
                        dict(num_clients=100), dict(allocation='invalid')]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                partition_indices(labels, labels, num_classes=4, **options)

    def test_local_class_absence_is_observed_not_forced_and_rng_is_isolated(self):
        labels = np.repeat(np.arange(10), 1000)
        test = np.repeat(np.arange(10), 100)
        np.random.seed(123)
        state = np.random.get_state()
        train, _ = partition_indices(labels, test, num_clients=10)
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        counts = [len(set(labels[t])) for t in train]
        self.assertTrue(any(n > 2 for n in counts))
        self.assertTrue(any(n < 10 for n in counts))

    def test_preserves_all_samples_with_unequal_class_totals(self):
        labels = np.repeat(np.arange(10), np.arange(200, 210))
        test = np.repeat(np.arange(10), 100)
        train, _ = partition_indices(labels, test, num_clients=10)
        self.assertEqual(sorted(sum(train, [])), list(range(len(labels))))
        np.testing.assert_array_equal(np.bincount(labels[sum(train, [])]), np.bincount(labels))

    def test_native_source_preserves_every_image_and_test_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'baseline', Path(tmp) / 'new'
            (source / 'train').mkdir(parents=True); (source / 'test').mkdir()
            (source / 'config.json').write_text(json.dumps(dict(num_clients=10, num_classes=10)))
            original_ids = []
            for client in range(10):
                y = np.tile(np.arange(10), 20 + client)
                ids = np.arange(len(y)) + client * 1000
                original_ids.extend(ids.tolist())
                np.savez_compressed(source / 'train' / f'{client}.npz', data={'x': ids[:, None].astype(np.float32), 'y': y})
                np.savez_compressed(source / 'test' / f'{client}.npz', data={'x': np.ones((10, 1)), 'y': np.arange(10)})
            meta = prepare_from_partitions(source, output, num_clients=10)
            observed = []
            for client in range(10):
                self.assertEqual((source / 'test' / f'{client}.npz').read_bytes(), (output / 'test' / f'{client}.npz').read_bytes())
                with np.load(output / 'train' / f'{client}.npz', allow_pickle=True) as f:
                    observed.extend(f['data'].item()['x'][:, 0].tolist())
            self.assertEqual(sorted(observed), sorted(original_ids))
            self.assertEqual(meta['train_samples'], len(original_ids))
            self.assertEqual(meta['test_samples'], 100)
            self.assertTrue(meta['all_source_training_samples_used_once'])
            self.assertEqual(meta['protocol'], 'hera_dirichlet_full_data_v3')
            with self.assertRaises(ValueError):
                prepare_from_partitions(source, output, num_clients=10)



class LocalClassEvaluationTests(unittest.TestCase):
    def make_client(self, train_classes):
        client = clientHERAFL.__new__(clientHERAFL)
        client.device = 'cpu'
        client.model = torch.nn.Identity()
        client.round_stats = {}
        client._training_class_ids = train_classes
        logits = torch.eye(4)[torch.tensor([0, 2, 2, 1])]
        client.load_test_data = lambda: [(logits, torch.arange(4))]
        return client

    def test_trained_local_accuracy_separates_seen_and_absent_labels(self):
        client = self.make_client([0, 1])
        self.assertEqual(client.local_accuracy_metrics(), (2, 4))
        for group in ('seen', 'absent'):
            self.assertEqual(client.round_stats[f'local_{group}_samples'], 2)
            self.assertEqual(client.round_stats[f'local_{group}_correct'], 1)
            self.assertEqual(client.round_stats[f'local_{group}_accuracy'], 0.5)

    def test_empty_absent_group_is_null_not_zero(self):
        client = self.make_client([0, 1, 2, 3])
        client.local_accuracy_metrics()
        self.assertEqual(client.round_stats['local_absent_samples'], 0)
        self.assertIsNone(client.round_stats['local_absent_accuracy'])
        json.dumps(client.round_stats, allow_nan=False)

    def test_complete_hera_round_saves_class_metrics_and_dataset_provenance(self):
        from test_heterogeneous_algorithms import NativeServerTests
        from flcore.servers.serverherafl import HERAFL
        args = NativeServerTests().make_args(algorithm='HERAFL')
        def read(dataset, idx, is_train=True, **kwargs):
            return [(torch.ones(4), torch.tensor(idx if is_train else i % 3)) for i in range(18)]
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            previous = Path.cwd()
            system = Path(tmp) / 'system'; system.mkdir()
            dataset = Path(tmp) / 'dataset/synthetic'; dataset.mkdir(parents=True)
            (dataset / 'config.json').write_text(json.dumps(dict(protocol='hera_fedsc_partition_v2')))
            os.chdir(system)
            stack.callback(os.chdir, previous)
            stack.enter_context(patch('flcore.servers.serverbase.read_client_data', side_effect=read))
            stack.enter_context(patch('flcore.clients.clientbase.read_client_data', side_effect=read))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            server = HERAFL(args, 0)
            server.train()
            for c in server.round_reports[0]['clients']:
                self.assertEqual(c['round_stats']['local_seen_samples'], 6)
                self.assertEqual(c['round_stats']['local_absent_samples'], 12)
                self.assertAlmostEqual(c['accuracy'], (c['round_stats']['local_seen_correct'] + c['round_stats']['local_absent_correct']) / 18)
            config = json.loads((server.result_directory / 'config.json').read_text())
            self.assertEqual(config['local_class_evaluation']['train_classes_by_client'], {'0': [0], '1': [1], '2': [2]})
            self.assertEqual(config['dataset_config']['protocol'], 'hera_fedsc_partition_v2')
            saved = json.loads(next(server.result_directory.glob('*_policy.json')).read_text())
            self.assertIn('local_absent_accuracy', saved[0]['clients'][0]['round_stats'])


if __name__ == '__main__':
    unittest.main()
