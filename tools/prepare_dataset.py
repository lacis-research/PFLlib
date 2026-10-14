#!/usr/bin/env python3
"""Verify native PFLlib partitions and invoke their existing generator if needed."""
import argparse
from contextlib import contextmanager, redirect_stdout
import inspect
import json
import os
import pickle
import zipfile
from pathlib import Path
import runpy
import sys
import numpy as np


@contextmanager
def dataset_context(root):
    previous_directory, previous_path = Path.cwd(), sys.path[:]
    os.chdir(root)
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        os.chdir(previous_directory)
        sys.path[:] = previous_path


def read_config(directory):
    try:
        metadata = json.loads((directory / 'config.json').read_text())
        return metadata if isinstance(metadata, dict) else {}
    except (OSError, ValueError):
        return {}


def inspect_partitions(directory, expected):
    metadata = read_config(directory)
    for key, value in expected.items():
        if metadata.get(key) != value:
            return None, f'configuração {key} ausente ou diferente'
    highest_label = -1
    for split in ('train', 'test'):
        for client in range(expected['num_clients']):
            path = directory / split / f'{client}.npz'
            try:
                with np.load(path, allow_pickle=True) as archive:
                    data = archive['data'].item()
                if len(data['x']) != len(data['y']) or not len(data['y']):
                    raise ValueError('partição vazia ou tamanhos inconsistentes')
                labels = np.asarray(data['y'])
                if np.any(labels < 0) or not np.issubdtype(labels.dtype, np.integer):
                    raise ValueError('rótulos inválidos')
                highest_label = max(highest_label, int(labels.max()))
            except (OSError, ValueError, KeyError, EOFError, TypeError, AttributeError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
                return None, f'partição {split}/{client}.npz ausente ou inválida ({error})'
    classes = metadata.get('num_classes', highest_label + 1)
    if not isinstance(classes, int) or classes <= highest_label:
        return None, 'num_classes incompatível com os rótulos'
    return classes, None


def prepare_dataset(root, dataset, scenario, clients=None, balance=True, partition='dir'):
    root = Path(root).resolve()
    normalized = dataset.lower().replace('-', '')
    generator = next((p for p in root.glob('generate_*.py')
                      if p.stem.removeprefix('generate_').lower().replace('-', '') == normalized), None)
    if generator is None:
        raise ValueError(f'Gerador nativo não encontrado para {dataset}.')
    if scenario not in ('iid', 'non-iid'):
        raise ValueError('SCENARIO deve ser iid ou non-iid.')
    if scenario == 'non-iid' and partition not in ('pat', 'dir', 'exdir'):
        raise ValueError('PARTITION deve ser pat, dir ou exdir para non-IID.')
    with dataset_context(root), redirect_stdout(sys.stderr):
        module = runpy.run_path(str(generator), run_name='native_dataset')
        generate = module.get('generate_dataset')
        if not callable(generate) or len(inspect.signature(generate).parameters) != 5:
            raise ValueError(f'{generator.name} não aceita cenário, clientes, balanceamento e partição.')
        directory = Path(module['dir_path'])
        metadata = read_config(directory)
        count = clients if clients is not None else metadata.get('num_clients', module['num_clients'])
        if not isinstance(count, int) or count < 1:
            raise ValueError('NUM_CLIENTS deve ser positivo.')
        native_check = generate.__globals__.get('check')
        check_globals = getattr(native_check, '__globals__', {})
        expected = dict(num_clients=count, non_iid=scenario == 'non-iid', balance=balance,
                        partition=partition if scenario == 'non-iid' else None)
        # These are native partitioning constants, distinct from training BATCH_SIZE.
        for key in ('alpha', 'batch_size'):
            if key in check_globals:
                expected[key] = check_globals[key]
        classes, reason = inspect_partitions(directory, expected)
        if reason:
            print(f'Gerando {dataset}: {reason}.', file=sys.stderr)
            for split in ('train', 'test'):
                (directory / split).mkdir(parents=True, exist_ok=True)
            # Native check() only checks metadata and can skip missing/corrupt files.
            # Bypass that early return only when our verification requires rebuilding.
            original_check = generate.__globals__.get('check')
            generate.__globals__['check'] = lambda *args, **kwargs: False
            try:
                generate(str(directory) + '/', count, expected['non_iid'], balance, expected['partition'])
            finally:
                generate.__globals__['check'] = original_check
            classes, reason = inspect_partitions(directory, expected)
            if reason:
                raise ValueError(f'O gerador não produziu partições válidas: {reason}.')
        else:
            print(f'Dataset {dataset} já gerado com a configuração solicitada.', file=sys.stderr)
        return count, classes, directory.as_posix().rstrip('/')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--scenario', choices=['iid', 'non-iid'], required=True)
    parser.add_argument('--num-clients', type=int)
    parser.add_argument('--balance', choices=['balance', 'unbalanced', '-'], default='balance')
    parser.add_argument('--partition', default='dir')
    args = parser.parse_args()
    try:
        print(*prepare_dataset(args.dataset_root, args.dataset, args.scenario, args.num_clients,
                               args.balance == 'balance', args.partition))
    except ValueError as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
