"""FedSC reference equivalence and real training on generated partitions."""
import contextlib
import hashlib
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
from prepare_fedsc import prepare_fedsc


def make_source(root):
    source = root / 'dataset' / 'synthetic'
    for split in ('train', 'test'):
        (source / split).mkdir(parents=True)
        for client in range(10):
            y = np.tile(np.arange(3), 100 if split == 'train' else 4)
            x = np.column_stack([np.eye(3)[y], np.ones(len(y))]).astype(np.float32)
            np.savez_compressed(source / split / f'{client}.npz', data=dict(x=x, y=y))
    (source / 'config.json').write_text(json.dumps(dict(num_clients=10, num_classes=3,
                                                       dataset='synthetic', non_iid=False)))
    return source


class FedSCScenariosTests(unittest.TestCase):
    def test_nonconverging_reference_sampler_fails_instead_of_hanging(self):
        from fedsc_partition import build_non_iid_by_dirichlet
        with patch('fedsc_partition.np.random.RandomState') as factory:
            factory.return_value.dirichlet.side_effect = ZeroDivisionError
            with self.assertRaisesRegex(ValueError, '10000 draws'):
                build_non_iid_by_dirichlet(66, [(i, i % 2) for i in range(100)],
                                          0.05, 2, 100, 10)

    def test_indices_match_original_fedsc_long_tail_and_dirichlet(self):
        # Generated from ../FedSC's original train_long_tail + clients_indices;
        # no dependency on that checkout is needed to run these regression tests.
        fixture = json.loads((ROOT / 'tests/fixtures/fedsc_long_tail.json').read_text())
        for case in fixture['cases']:
            with self.subTest(classes=case['num_classes'], factor=case['imb_factor']):
                y = np.repeat(np.arange(case['num_classes']), case['source_counts'])
                options = {k: case[k] for k in ('num_classes', 'num_clients', 'alpha', 'seed', 'imb_factor')}
                state = np.random.get_state()
                train, _ = partition_indices(y, np.tile(np.arange(case['num_classes']), 100), **options)
                after = np.random.get_state()
                np.testing.assert_array_equal(state[1], after[1])
                self.assertEqual(state[2:], after[2:])
                self.assertEqual(hashlib.sha256(json.dumps(train, separators=(',', ':')).encode()).hexdigest(),
                                 case['indices_sha256'])
                np.testing.assert_array_equal(np.bincount(y[sum(train, [])]), case['selected_counts'])

    def test_invalid_imbalance_rejected(self):
        y = np.tile(np.arange(10), 100)
        for factor in (0, -1, 1.1, float('nan'), float('inf'), 1e-20):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                partition_indices(y, y, num_clients=10, imb_factor=factor)
        with self.assertRaisesRegex(ValueError, 'only exponential'):
            partition_indices(y, y, imb_type='step')

    def test_written_samples_exclusions_and_tests_match_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_source(Path(tmp))
            output = source.parent / 'shift'
            metadata = prepare_from_partitions(source, output, imb_factor=0.1, alpha=1.0)
            indices = json.loads((output / 'indices.json').read_text())
            used = sum(indices['train'], [])
            self.assertEqual(sorted(used + indices['excluded_train']), list(range(3000)))
            self.assertEqual(len(set(used)), len(used))
            self.assertEqual(metadata['selected_train_class_counts'], [1000, 316, 100])
            self.assertFalse(metadata['all_source_training_samples_used_once'])
            self.assertFalse(metadata['temporal_drift'])
            pool = [np.load(source / 'train' / f'{i}.npz', allow_pickle=True)['data'].item() for i in range(10)]
            for i in range(10):
                self.assertEqual((source / 'test' / f'{i}.npz').read_bytes(),
                                 (output / 'test' / f'{i}.npz').read_bytes())
                data = np.load(output / 'train' / f'{i}.npz', allow_pickle=True)['data'].item()
                for key in ('x', 'y'):
                    np.testing.assert_array_equal(data[key], np.concatenate([p[key] for p in pool])[indices['train'][i]])

    def test_launcher_preparation_reuses_and_separates_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            source = make_source(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'not been prepared'):
                prepare_fedsc(source, alpha=1.0, reuse_only=True)
            first = prepare_fedsc(source, alpha=1.0)
            self.assertEqual(first, prepare_fedsc(source, alpha=1.0, reuse_only=True))
            second = prepare_fedsc(source, alpha=1.0, imb_factor=1.0)
            self.assertNotEqual(first[2], second[2])
            with self.assertRaisesRegex(ValueError, 'must match'):
                prepare_fedsc(source, num_clients=30)
            (source.parent / first[2] / 'train/0.npz').unlink()
            with self.assertRaisesRegex(ValueError, 'invalid'):
                prepare_fedsc(source, alpha=1.0)

    def test_all_three_algorithms_train_on_generated_data_and_save_provenance(self):
        from test_heterogeneous_algorithms import NativeServerTests
        from flcore.servers.serverherafl import HERAFL
        from flcore.servers.serverfedload import FedLoad, ComparisonFedAvg
        from flcore.trainmodel.models import DNN
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            source = make_source(root)
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            _, _, name = prepare_fedsc(source, alpha=1.0)
            metadata = json.loads((source.parent / name / 'config.json').read_text())
            previous = Path.cwd()
            (root / 'system').mkdir()
            os.chdir(root / 'system')
            stack.callback(os.chdir, previous)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            for algorithm, server_type in [('HERAFL', HERAFL), ('FedLoad', FedLoad), ('FedAvg', ComparisonFedAvg)]:
                with self.subTest(algorithm=algorithm):
                    args = NativeServerTests().make_args(algorithm=algorithm, dataset=name,
                        model=DNN(input_dim=4, mid_dim=8, num_classes=3), num_clients=10,
                        join_ratio=0.2, aligned_comparison=True, record_resources=True,
                        wait_for_all=True, batch_size=16, learning_rate_decay=False)
                    server = server_type(args, 0)
                    before = [p.detach().clone() for p in server.global_model.parameters()]
                    server.train()
                    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, server.global_model.parameters())))
                    saved = json.loads((server.result_directory / 'config.json').read_text())
                    self.assertEqual(saved['dataset_config']['selected_train_class_counts'], [1000, 316, 100])
                    self.assertEqual(saved['dataset_config']['imb_factor'], 0.1)
                    for client in server.clients:
                        self.assertEqual(client.train_samples, sum(metadata['train_class_counts'][client.id]))
                        if algorithm == 'HERAFL':
                            self.assertEqual(client.training_class_ids(),
                                             np.flatnonzero(metadata['train_class_counts'][client.id]).tolist())


if __name__ == '__main__':
    unittest.main()
