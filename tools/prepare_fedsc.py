#!/usr/bin/env python3
"""Prepare/reuse FedSC long-tail + Dirichlet data for the simulation launcher."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys

from prepare_dataset import inspect_partitions
from prepare_hera_shift import prepare_from_partitions


def prepare_fedsc(source, num_clients=None, alpha=0.05, imb_factor=0.1, seed=66,
                  reuse_only=False):
    source = Path(source).resolve()
    config = json.loads((source / 'config.json').read_text())
    clients = config['num_clients'] if num_clients is None else num_clients
    if clients != config['num_clients']:
        raise ValueError('NUM_CLIENTS must match the source dataset; prepare a separate baseline first.')
    options = dict(num_clients=clients, num_classes=config['num_classes'], alpha=alpha,
                   imb_factor=imb_factor, imb_type='exp', seed=seed, allocation='dirichlet')
    # Include the input contents, not just their path: changed bases must never
    # silently reuse an old scenario. Outputs live alongside the original base.
    hashes = {f'{split}/{i}.npz': hashlib.sha256((source / split / f'{i}.npz').read_bytes()).hexdigest()
              for split in ('train', 'test') for i in range(clients)}
    signature = hashlib.sha256(json.dumps(dict(options=options, config=config, hashes=hashes),
                                         sort_keys=True).encode()).hexdigest()[:16]
    output = source.parent / f'{source.name}FedSC_{signature}'
    if output.exists():
        expected = dict(options, protocol='fedsc_long_tail_v1', source_sha256=hashes,
                        source_dataset=str(source), non_iid=True)
        _, reason = inspect_partitions(output, expected)
        if reason:
            raise ValueError(f'Existing FedSC dataset is invalid ({reason}): {output}')
        print(f'Reusing FedSC dataset: {output.name}', file=sys.stderr)
    elif reuse_only:
        raise ValueError(f'FedSC dataset has not been prepared: {output.name}')
    else:
        with redirect_stdout(sys.stderr):
            prepare_from_partitions(source, output, **options)
        print(f'Prepared FedSC dataset: {output.name}', file=sys.stderr)
    return clients, config['num_classes'], output.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dataset', type=Path, required=True)
    parser.add_argument('--num-clients', type=int)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--imb-factor', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--reuse-only', action='store_true')
    args = parser.parse_args()
    try:
        print(*prepare_fedsc(args.source_dataset, args.num_clients, args.alpha,
                             args.imb_factor, args.seed, args.reuse_only))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
