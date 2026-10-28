"""Comparison recovery must use a shared goal, never mix experiment sessions."""
import os
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from plot_comparison import latest_common_goal, comparison_directory_name


class ComparisonDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.algorithms = ['HERAFL', 'FedLoad', 'FedAvg']

    def add(self, algorithm, goal, timestamp):
        directory = self.root / algorithm / goal
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'MNIST_{algorithm}_{goal}_0.h5'
        path.touch()
        os.utime(path, (timestamp, timestamp))

    def test_latest_complete_goal_ignores_newer_incomplete_session(self):
        for algorithm in self.algorithms:
            self.add(algorithm, 'start-old', 100)
            self.add(algorithm, 'start-new_with_underscores', 200)
        self.add('HERAFL', 'start-incomplete', 300)
        self.assertEqual(latest_common_goal(self.root, 'MNIST', self.algorithms), 'start-new_with_underscores')

    def test_different_goals_cannot_be_combined(self):
        for i, algorithm in enumerate(self.algorithms):
            self.add(algorithm, f'session-{i}', 100 + i)
        with self.assertRaisesRegex(ValueError, 'No complete comparison'):
            latest_common_goal(self.root, 'MNIST', self.algorithms)

    def test_only_requested_methods_required(self):
        self.add('HERAFL', 'two-methods', 100)
        self.add('FedLoad', 'two-methods', 101)
        self.assertEqual(latest_common_goal(self.root, 'MNIST', ['HERAFL', 'FedLoad']), 'two-methods')

    def test_folder_name_uses_saved_scenario_and_model(self):
        source = self.root/'metrics.h5'
        (self.root/'config.json').write_text(json.dumps(dict(
            model='VGG11', training_config=dict(scenario='iid'),
            partition_metadata=dict(non_iid=True))))
        self.assertEqual(comparison_directory_name(['HERAFL', 'FedLoad'], 'Cifar10', 'start-123', [source]),
                         'HERAFL__vs__FedLoad_Cifar10_non-iid_VGG11_start-123')

    def test_missing_metadata_is_explicit(self):
        self.assertIn('scenario-unknown_model-unknown',
                      comparison_directory_name(['HERAFL'], 'MNIST', 'test', [self.root/'metrics.h5']))


if __name__ == '__main__':
    unittest.main()
