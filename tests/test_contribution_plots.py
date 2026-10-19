import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from contribution_plots import cumulative_cost, cost_components, render_contributions


class ContributionTests(unittest.TestCase):
    def report(self, round=1, status='rejected'):
        return {'round': round, 'semantic_aggregation_time': .5, 'clients': [
            {'id': 4, 'status': status, 'retention': .5, 'round_stats': {
                'round_time': 10, 'compute_time': 3, 'training_time': 2,
                'semantic_extraction_time': 1, 'download_time': 3, 'upload_time': 4,
                'energy_consumption': 20, 'total_communication_bytes': 100,
                'parameter_download_bytes': 40, 'parameter_upload_bytes': 40,
                'semantic_download_bytes': 5, 'semantic_upload_bytes': 10,
                'parameter_fraction': .4}}]}

    def test_rejected_costs_and_gaps(self):
        self.assertEqual(cumulative_cost([self.report(), self.report(2)], 'round_time'), {0: 0, 1: 10, 2: 20})
        self.assertEqual(cumulative_cost([self.report(), self.report(3)], 'round_time'), {0: 0, 1: 10})
        report = self.report(2)
        del report['clients'][0]['round_stats']['round_time']
        self.assertEqual(cumulative_cost([self.report(), report], 'round_time'), {0: 0, 1: 10})

    def test_components_conserve_cost_without_double_counting(self):
        c = cost_components([self.report()])
        self.assertEqual(c['training']+c['extraction']+c['transfer']+c['unsplit_compute'], 10)
        self.assertEqual(c['parameters']+c['semantic']+c['other_bytes'], 100)
        self.assertEqual(c['server'], .5)
        report = self.report()
        del report['clients'][0]['round_stats']['training_time']
        self.assertEqual(cost_components([report])['unsplit_compute'], 3)

    def test_render_and_common_round_ablation(self):
        runs = {'full': [dict(id='42', reports=[self.report(), self.report(2)], accuracy={1: .5, 2: .8})],
                'control': [dict(id='42', reports=[self.report()], accuracy={1: .4})]}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            render_contributions(runs, output, ablations=True)
            self.assertEqual(len(list(output.glob('*.png'))), 9)
            csv = (output/'ablation_summary.csv').read_text()
            self.assertIn('full,42,1,accuracy,50.0', csv)
            self.assertNotIn('80.0', csv)


if __name__ == '__main__':
    unittest.main()
