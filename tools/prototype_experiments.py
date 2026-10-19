#!/usr/bin/env python3
"""Print the prototype ablation matrix; train only with --execute.

Each seed/variant gets a unique goal. Execution writes per-model manifests for
contribution_plots.py, without automatically merging unrelated result sessions.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def commands(args):
    prototype = ['-algo', 'HERAFL', '--hera-communication', 'prototypes']
    variants = {
        'fedavg': ['-algo', 'FedAvg', '--record-resources'],
        'fedload': ['-algo', 'FedLoad'],
        'parameter_semantic_control': ['-algo', 'HERAFL', '--hera-semantic'],
        'preserved_parameter_control': ['-algo', 'HERAFL', '--hera-semantic', '--hera-semantic-policy', 'legacy'],
        'prototype_simple': prototype,
        'prototype_full': prototype + ['--pruning-arms', '1'],
        'prototype_fixed': prototype + ['--pruning-arms', '.5'],
        'local_only': prototype + ['--hera-local-only'],
        'zero_alignment': prototype + ['--hera-align-weight', '0'],
        'shuffled_reference': prototype + ['--hera-semantic-shuffle'],
        'semantic_policy': prototype + ['--hera-prototype-policy', 'semantic'],
        'relational': prototype + ['--hera-prototype-kind', 'relational', '--hera-rpcl-weight', '1', '--hera-cpdr-weight', '1'],
        'rpcl_only': prototype + ['--hera-prototype-kind', 'relational', '--hera-rpcl-weight', '1'],
        'cpdr_only': prototype + ['--hera-prototype-kind', 'relational', '--hera-cpdr-weight', '1'],
    }
    for model in args.models:
        for seed in args.seeds:
            for name in args.variants or variants:
                if name not in variants:
                    raise ValueError(f'Unknown variant {name}; choices: {list(variants)}')
                goal = f'{args.experiment}_{model}_{name}_seed{seed}'
                base = [sys.executable, 'main.py', '-data', args.dataset, '-m', model, '-dev', args.device,
                        '-gr', str(args.rounds), '-ls', str(args.local_epochs), '-lbs', '64', '-jr', str(args.join_ratio),
                        '--policy-seed', str(seed), '--aligned-comparison', '--reward-validation-fraction', '.2',
                        '--pruning-arms', '.25', '.5', '.75', '1', '--experiment-id', goal, '-go', goal]
                if args.wait_for_all:
                    base += ['--wait-for-all']
                yield model, name, goal, base + variants[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--models', nargs='+', choices=['CNN', 'DNN', 'VGG11', 'VGG16'], default=['VGG11'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--join-ratio', type=float, default=.2)
    parser.add_argument('--experiment', required=True, help='Unique identifier for this experiment batch.')
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--wait-for-all', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--manifest-dir', type=Path, default=Path('results/ablation_manifests'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifests = {}
    for model, name, goal, command in commands(args):
        print('(cd system && ' + shlex.join(command) + ')', flush=True)
        if not args.execute:
            continue
        subprocess.run(command, cwd=root/'system', check=True)
        algorithm = command[command.index('-algo')+1]
        candidates = list((root/'results').glob(f'**/{args.dataset}_{algorithm}_{goal}_0.h5'))
        if not candidates:
            raise FileNotFoundError(f'No result for {goal}')
        path = max(candidates, key=lambda p: p.stat().st_mtime)
        manifests.setdefault(model, {}).setdefault(name, []).append(str(path.resolve()))
        args.manifest_dir.mkdir(parents=True, exist_ok=True)
        (args.manifest_dir/f'{args.experiment}_{model}.json').write_text(json.dumps(manifests[model], indent=2))


if __name__ == '__main__':
    main()
