import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from comparison_metrics import mean_accuracy, experiment_total


class ComparisonMetricTests(unittest.TestCase):
    def test_mean_accuracy_uses_responders_and_is_not_sample_weighted(self):
        reports = [{'round': 4, 'clients': [
            {'status': 'accepted', 'accuracy': 0.2, 'num_samples': 100},
            {'status': 'accepted', 'accuracy': 0.8, 'num_samples': 900},
            {'status': 'dropped', 'accuracy': 1.0, 'num_samples': 1000}]}]
        self.assertEqual(mean_accuracy(reports), {4: 0.5})
        self.assertEqual(mean_accuracy([{'round': 0, 'clients': []}]), {})

    def test_total_time_sums_client_times_instead_of_round_maximum(self):
        reports = [{'round': 0, 'clients': [
            {'round_stats': {'round_time': 2, 'energy_consumption': 3, 'total_communication_bytes': 10}},
            {'round_stats': {'round_time': 5, 'energy_consumption': 7, 'total_communication_bytes': 20}}]},
            {'round': 1, 'clients': [
                {'round_stats': {'round_time': 4, 'energy_consumption': 6, 'total_communication_bytes': 30}}]}]
        self.assertEqual(experiment_total(reports, 'round_time'), 11)
        self.assertEqual(experiment_total(reports, 'energy_consumption'), 16)
        self.assertEqual(experiment_total(reports, 'total_communication_bytes'), 60)
