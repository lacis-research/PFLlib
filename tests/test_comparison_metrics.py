import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from comparison_metrics import experiment_total


class ComparisonMetricTests(unittest.TestCase):
    def test_total_time_sums_client_times_instead_of_round_maximum(self):
        reports = [{'round': 0, 'clients': [
            {'round_stats': {'round_time': 2, 'energy_consumption': 3, 'total_communication_bytes': 10}},
            {'round_stats': {'round_time': 5, 'energy_consumption': 7, 'total_communication_bytes': 20}}]},
            {'round': 1, 'clients': [
                {'round_stats': {'round_time': 4, 'energy_consumption': 6, 'total_communication_bytes': 30}}]}]
        self.assertEqual(experiment_total(reports, 'round_time'), 11)
        self.assertEqual(experiment_total(reports, 'energy_consumption'), 16)
        self.assertEqual(experiment_total(reports, 'total_communication_bytes'), 60)

    def test_prototype_totals_include_rejected_clients_and_protocol_overhead(self):
        reports = [{'round': 1, 'initialization_bytes': 100, 'evaluation_telemetry_bytes': 5,
                    'clients': [
                        {'status': 'accepted', 'round_stats': {'total_communication_bytes': 10}},
                        {'status': 'dropped', 'round_stats': {'total_communication_bytes': 20}},
                        {'status': 'infeasible', 'round_stats': {}}]}]
        self.assertEqual(experiment_total(reports, 'total_communication_bytes'), 10)
        self.assertEqual(experiment_total(reports, 'total_communication_bytes', include_all=True), 135)
        reports[0]['evaluation_parameter_bytes'] = 200
        reports[0]['evaluation_communication_bytes'] = 205
        self.assertEqual(experiment_total(reports, 'total_communication_bytes', include_all=True), 335)


class LocalTestComparisonTests(unittest.TestCase):
    def test_common_clients_weighted_including_dropped(self):
        from comparison_metrics import matched_local_accuracy
        def report(correct):
            return [(0, [{'round': 1, 'clients': [
                {'id': 0, 'status': 'dropped', 'local_test': {'correct': correct, 'samples': 2}},
                {'id': 1, 'local_test': {'correct': 8, 'samples': 8}}]}])]
        self.assertEqual(matched_local_accuracy({'HERAFL': report(1), 'FedLoad': report(2)}),
                         {'HERAFL': [{1: .9}], 'FedLoad': [{1: 1.}]})
        self.assertEqual(matched_local_accuracy({'HERAFL': report(1), 'FedLoad': [(0, [])]}), {})


    def test_strict_local_metrics_reject_different_partitions_and_legacy(self):
        from comparison_metrics import matched_local_accuracy
        import copy
        record = dict(correct=1, samples=2, loss_sum=4., auc=.75,
                      partition_sha256='same', protocol='local-test-v2')
        runs = {a: [(0, [dict(round=1, clients=[dict(id=0, local_test=copy.deepcopy(record))])])]
                for a in ('HERAFL', 'FedLoad')}
        self.assertEqual(matched_local_accuracy(runs, 'loss_sum', strict=True),
                         {'HERAFL': [{1: 2.}], 'FedLoad': [{1: 2.}]})
        self.assertEqual(matched_local_accuracy(runs, 'auc', strict=True),
                         {'HERAFL': [{1: .75}], 'FedLoad': [{1: .75}]})
        other = runs['FedLoad'][0][1][0]['clients'][0]['local_test']
        other['partition_sha256'] = 'different'
        self.assertEqual(matched_local_accuracy(runs, strict=True), {})
        other.pop('partition_sha256')
        self.assertEqual(matched_local_accuracy(runs, strict=True), {})

    def test_legacy_shuffled_hashes_do_not_suppress_comparison(self):
        from comparison_metrics import matched_local_accuracy
        runs = {a: [(0, [dict(round=1, clients=[dict(id=0, local_test=dict(
            correct=1, samples=2, loss_sum=1., auc=.75,
            partition_sha256=a, protocol='local-test-v1'))])])]
            for a in ('HERAFL', 'FedLoad')}
        self.assertEqual(matched_local_accuracy(runs, strict=True),
                         {'HERAFL': [{1: .5}], 'FedLoad': [{1: .5}]})


class GlobalMetricTests(unittest.TestCase):
    def test_global_plots_exclude_local_protocol_and_preserve_coordinates(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from plot_comparison import render_global_metrics
        metrics = {key: {'HERAFL': [{2: 80.}], 'FedLoad': [{2: 70.}]}
                   for key in ('rs_test_acc', 'rs_train_loss', 'rs_test_auc')}
        with TemporaryDirectory() as tmp, patch('plot_comparison.draw') as draw:
            render_global_metrics(metrics, {'HERAFL': 'local_models_sample_weighted',
                                           'FedLoad': 'global_model'}, Path(tmp))
            self.assertEqual(draw.call_count, 3)
            for call in draw.call_args_list:
                self.assertEqual(call.args[0], {'FedLoad': [{2: 70.}]})
        with TemporaryDirectory() as tmp, patch('plot_comparison.draw') as draw:
            render_global_metrics(metrics, {'HERAFL': 'global_model', 'FedLoad': 'global_model'}, Path(tmp))
            self.assertEqual(set(draw.call_args_list[0].args[0]), {'HERAFL', 'FedLoad'})
