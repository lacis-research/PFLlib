#!/usr/bin/env python3
"""Create independent image partitions with FedSC Dirichlet and optional long tail."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from fedsc_partition import clients_indices


def partition_indices(train_labels, test_labels, num_clients=30, num_classes=10,
                      alpha=0.05, seed=66,
                      allocation='dirichlet', imb_factor=None, imb_type='exp'):
    """Use FedSC's Dirichlet/balancing helpers after optional long-tail selection.

    No local class mask or per-class minimum is imposed. A class is absent
    precisely when that client's sampled partition has no examples of it.
    The optional uniform mode is a separate baseline, not FedSC's sampler.
    Long-tail selection follows utils/long_tailed_cifar10.py in FedSC commit
    a04dc8e630706a0c3f12f716c45857ead95a4940, using an isolated MT19937.
    """
    if num_clients < 1 or num_classes < 2:
        raise ValueError('Require positive clients and at least two classes.')
    if allocation not in ('dirichlet', 'uniform'):
        raise ValueError('allocation must be dirichlet or uniform.')
    if allocation == 'dirichlet' and num_clients < 10:
        raise ValueError('The original FedSC sampler requires at least 10 clients.')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be positive and finite.')
    if imb_type != 'exp':
        raise ValueError('FedSC implements only exponential imbalance (exp).')
    if imb_factor is not None and (not math.isfinite(imb_factor) or not 0 < imb_factor <= 1):
        raise ValueError('imb_factor must be finite and in (0, 1].')
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    for labels in (train_labels, test_labels):
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= num_classes):
            raise ValueError('Labels must be integer class IDs in [0, num_classes).')
    # Shuffle class indices with isolated MT19937 for deterministic assignment.
    rng = np.random.RandomState(seed)
    labels_to_indices = []
    for label in range(num_classes):
        candidates = np.flatnonzero(train_labels == label)
        rng.shuffle(candidates)
        if not len(candidates):
            raise ValueError(f'Class {label}: no training samples in the source.')
        if imb_factor is not None:
            # FedSC utils/long_tailed_cifar10.py: mean source class size,
            # integer truncation, then slice the shuffled class indices.
            target = int(len(train_labels) / num_classes * imb_factor ** (label / (num_classes - 1.0)))
            candidates = candidates[:target]
            if not len(candidates):
                raise ValueError(f'Class {label}: imbalance leaves no training samples.')
        labels_to_indices.append(candidates.tolist())
    selected_count = sum(map(len, labels_to_indices))
    if selected_count < 2 * num_clients:
        raise ValueError('Too few selected training samples for the FedSC sampler.')
    if allocation == 'dirichlet':
        train = clients_indices(labels_to_indices, num_classes, num_clients, alpha, seed)
    else:
        train = [[] for _ in range(num_clients)]
        for indices in labels_to_indices:
            for client, chunk in enumerate(np.array_split(indices, num_clients)):
                train[client].extend(chunk.tolist())
    if len(train) != num_clients or any(not indices for indices in train):
        raise ValueError('Partitioner did not produce one nonempty partition per client.')
    # This test protocol is intentionally HERA-specific: independent official
    # test examples, all classes per UE. It does not affect training draws.
    test_rng = np.random.RandomState(seed)
    test = [[] for _ in range(num_clients)]
    for label in range(num_classes):
        heldout = np.flatnonzero(test_labels == label)
        if len(heldout) < num_clients:
            raise ValueError(f'Test class {label} needs at least one example per client.')
        test_rng.shuffle(heldout)
        for client, indices in enumerate(np.array_split(heldout, num_clients)):
            test[client].extend(indices.tolist())
    return train, test


def prepare_from_partitions(source, output, **settings):
    """Select/redistribute baseline training images, copying each UE test verbatim."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a new directory.')
    config = json.loads((source / 'config.json').read_text())
    options = dict(num_clients=config['num_clients'], num_classes=config['num_classes'],
                   alpha=0.05, seed=66, allocation='dirichlet', imb_factor=None, imb_type='exp')
    options.update(settings)
    if options['num_clients'] != config['num_clients'] or options['num_classes'] != config['num_classes']:
        raise ValueError('Keep the baseline client and class counts to preserve each UE test.')
    clients, classes = options['num_clients'], options['num_classes']
    images, labels, tests, offsets, hashes = [], [], [], [], {}
    cursor = 0
    for client in range(clients):
        for split in ('train', 'test'):
            path = source / split / f'{client}.npz'
            with np.load(path, allow_pickle=True) as archive:
                data = archive['data'].item()
            if len(data['x']) != len(data['y']) or not len(data['y']):
                raise ValueError(f'Invalid source partition: {path}')
            hashes[f'{split}/{client}.npz'] = hashlib.sha256(path.read_bytes()).hexdigest()
            if split == 'train':
                images.append(data['x']); labels.append(data['y'])
                offsets.append(dict(client=client, start=cursor, stop=cursor + len(data['y'])))
                cursor += len(data['y'])
            else:
                if len(set(data['y'])) != classes:
                    raise ValueError(f'Baseline UE {client} test does not contain every class.')
                tests.append(data['y'])
    x, y, test_y = np.concatenate(images), np.concatenate(labels), np.concatenate(tests)
    del images, labels
    train, _ = partition_indices(y, test_y, **options)
    used = np.asarray([idx for indices in train for idx in indices], dtype=np.int64)
    if len(np.unique(used)) != len(used) or np.any(used < 0) or np.any(used >= len(y)):
        raise ValueError('Training partitions must be disjoint subsets of the source.')
    if options['imb_factor'] is None and sorted(used.tolist()) != list(range(len(y))):
        raise ValueError('Training redistribution must use every source image exactly once.')
    excluded = np.setdiff1d(np.arange(len(y)), used).tolist()
    counts = [np.bincount(y[idx], minlength=classes).tolist() for idx in train]
    metadata = dict(config, **options)
    metadata.update(dataset=config.get('dataset', source.name),
                    protocol='fedsc_long_tail_v1' if options['imb_factor'] is not None else 'hera_dirichlet_full_data_v3',
                    non_iid=options['allocation'] == 'dirichlet',
                    source_dataset=str(source), source_dataset_config=config,
                    source_sha256=hashes, source_train_offsets=offsets,
                    train_class_counts=counts,
                    test_class_counts=[np.bincount(t, minlength=classes).tolist() for t in tests],
                    local_class_counts=[sum(n > 0 for n in c) for c in counts],
                    train_samples=len(used), test_samples=len(test_y),
                    source_train_samples=len(y), excluded_train_samples=len(excluded),
                    source_train_class_counts=np.bincount(y, minlength=classes).tolist(),
                    selected_train_class_counts=np.bincount(y[used], minlength=classes).tolist(),
                    temporal_drift=False,
                    all_source_training_samples_used_once=not excluded,
                    train_partition='fedsc_clients_indices' if options['allocation'] == 'dirichlet' else 'uniform_by_class',
                    test_protocol='original_baseline_test_per_ue_unchanged',
                    normalization='preserved_from_source_arrays',
                    absent_classes_policy='observed_from_sampled_training_labels',
                    partition='dir' if options['allocation'] == 'dirichlet' else 'uniform',
                    source_split='original_herafl_baseline',
                    fedsc_reference_commit='a04dc8e630706a0c3f12f716c45857ead95a4940')
    metadata.pop('Size of samples for labels in clients', None)
    output.mkdir(parents=True)
    (output / 'train').mkdir(); (output / 'test').mkdir()
    for client, idx in enumerate(train):
        np.savez_compressed(output / 'train' / f'{client}.npz', data={'x': x[idx], 'y': y[idx]})
        shutil.copy2(source / 'test' / f'{client}.npz', output / 'test' / f'{client}.npz')
    (output / 'indices.json').write_text(json.dumps(dict(train=train, excluded_train=excluded, source_train_offsets=offsets)))
    (output / 'config.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dataset', type=Path, required=True,
                        help='Existing baseline: preserve its pixels, labels and each UE test.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--num-clients', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--allocation', choices=['dirichlet', 'uniform'], default='dirichlet')
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--imb-factor', type=float, default=None,
                        help='FedSC exponential long tail, e.g. 0.1; omitted preserves every training sample.')
    parser.add_argument('--imb-type', choices=['exp'], default='exp')
    args = parser.parse_args()
    options = vars(args).copy()
    output, source = options.pop('output_dir'), options.pop('source_dataset')
    try:
        result = prepare_from_partitions(source, output, **options)
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
