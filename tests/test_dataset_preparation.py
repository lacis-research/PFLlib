"""Verify native generator reuse, configuration changes and incomplete partitions."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from prepare_dataset import prepare_dataset

GENERATOR = '''
import json
from pathlib import Path
import numpy as np
num_clients = 3
dir_path = 'Fixture/'
alpha = 0.1
batch_size = 10

def check(*args, **kwargs):
    return True

def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if check():
        return
    directory = Path(dir_path)
    ledger = directory / 'calls.txt'
    ledger.write_text(str(int(ledger.read_text()) + 1 if ledger.exists() else 1))
    for split in ('train', 'test'):
        for client in range(num_clients):
            np.savez_compressed(directory / split / f'{client}.npz',
                                data={'x': np.zeros((2, 4)), 'y': np.array([0, 99])})
    metadata = dict(num_clients=num_clients, num_classes=100, non_iid=niid,
                    balance=balance, partition=partition, alpha=alpha, batch_size=batch_size)
    (directory / 'config.json').write_text(json.dumps(metadata))
'''


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'generate_Fixture.py').write_text(GENERATOR)
        self.directory = self.root / 'Fixture'

    def prepare(self, **kwargs):
        settings = dict(scenario='non-iid', balance=True, partition='dir')
        settings.update(kwargs)
        with contextlib.redirect_stderr(io.StringIO()):
            return prepare_dataset(self.root, 'fixture', **settings)

    def calls(self):
        return int((self.directory / 'calls.txt').read_text())

    def test_missing_dataset_generated_then_reused(self):
        self.assertEqual(self.prepare(), (3, 100, 'Fixture'))
        self.assertEqual(self.prepare(), (3, 100, 'Fixture'))
        self.assertEqual(self.calls(), 1)

    def test_missing_partition_rebuilt_despite_native_metadata_check(self):
        self.prepare()
        (self.directory / 'test/1.npz').unlink()
        self.prepare()
        self.assertEqual(self.calls(), 2)
        self.assertTrue((self.directory / 'test/1.npz').is_file())

    def test_corrupt_partition_rebuilt(self):
        self.prepare()
        (self.directory / 'train/0.npz').write_bytes(b'corrupted')
        self.prepare()
        self.assertEqual(self.calls(), 2)

    def test_parameters_trigger_regeneration(self):
        self.prepare()
        self.prepare(clients=4, balance=False, partition='pat')
        metadata = json.loads((self.directory / 'config.json').read_text())
        self.assertEqual((metadata['num_clients'], metadata['balance'], metadata['partition']), (4, False, 'pat'))
        self.assertEqual(self.calls(), 2)
        self.prepare(scenario='iid', clients=4, balance=False)
        metadata = json.loads((self.directory / 'config.json').read_text())
        self.assertFalse(metadata['non_iid'])
        self.assertIsNone(metadata['partition'])
        self.assertEqual(self.calls(), 3)

    def test_invalid_config_rebuilt(self):
        self.prepare()
        (self.directory / 'config.json').write_text('{broken')
        self.prepare()
        self.assertEqual(self.calls(), 2)

    def test_invalid_arguments_do_not_generate(self):
        for settings in (dict(clients=0), dict(partition='unknown')):
            with self.assertRaises(ValueError):
                self.prepare(**settings)
        self.assertFalse(self.directory.exists())


if __name__ == '__main__':
    unittest.main()
