#!/usr/bin/env python3
"""Print reproducible ablations; execute only with --execute. Run from repo root."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def commands(args):
    variants = {
        'fedavg': ['-algo', 'FedAvg'],
        'fedload': ['-algo', 'FedLoad'],
        'hera_fedload_control': ['-algo', 'HERAFL'],
        'preserved': ['-algo', 'HERAFL', '--hera-semantic', '--hera-semantic-policy', 'legacy'],
        'heuristic': ['-algo', 'HERAFL', '--hera-semantic', '--hera-semantic-policy', 'heuristic'],
        'linucb': ['-algo', 'HERAFL', '--hera-semantic'],
        'only_H': ['-algo', 'HERAFL', '--hera-semantic', '--hera-semantic-features', 'H'],
        'without_D': ['-algo', 'HERAFL', '--hera-semantic', '--hera-semantic-features', 'H,U,Q'],
        'without_Q': ['-algo', 'HERAFL', '--hera-semantic', '--hera-semantic-features', 'H,U,D'],
        'shuffled': ['-algo', 'HERAFL', '--hera-semantic', '--hera-semantic-shuffle'],
        'reward': ['-algo', 'HERAFL', '--hera-semantic', '--hera-semantic-reward-weight', '0.2'],
    }
    for arm in (.25, .5, .75, 1.):
        variants[f'fixed_{arm}'] = ['-algo', 'HERAFL', '--hera-semantic', '--hera-semantic-policy', 'legacy', '--pruning-arms', str(arm)]
    for model in args.models:
        for seed in args.seeds:
            for name, options in variants.items():
                base = [sys.executable, 'main.py', '-data', args.dataset, '-m', model, '-dev', args.device,
                        '-gr', str(args.rounds), '-ls', '1', '-lbs', '64', '-jr', '0.2',
                        '--policy-seed', str(seed), '--aligned-comparison', '--reward-validation-fraction', '0.2',
                        '--detailed-evaluation', '--pruning-arms', '.25', '.5', '.75', '1',
                        '--experiment-id', f'{args.dataset}_{model}_{name}_seed{seed}']
                if args.wait_for_all:
                    base += ['--wait-for-all']
                yield base + options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='Cifar10_dir02_seed42')
    parser.add_argument('--models', nargs='+', choices=['CNN', 'DNN', 'VGG11', 'VGG16'], default=['VGG11', 'VGG16'])
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    parser.add_argument('--wait-for-all', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    for command in commands(args):
        print('(cd system && ' + shlex.join(command) + ')', flush=True)
        if args.execute:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[1]/'system', check=True)


if __name__ == '__main__':
    main()
