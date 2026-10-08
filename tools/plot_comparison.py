#!/usr/bin/env python3
"""Compare native PFLlib HDF5 metrics and optional heterogeneous-policy logs."""
import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from comparison_metrics import render_reference_comparison


METRICS = {
    'rs_test_acc': ('accuracy', 'Acurácia de teste', 100.0, '%'),
    'rs_test_auc': ('auc', 'AUC de teste', 1.0, 'AUC'),
    'rs_train_loss': ('train_loss', 'Perda de treino', 1.0, 'Perda'),
}
RESOURCES = {
    'energy': ('Energia estimada total por rodada', 'J'),
    'communication': ('Comunicação dos submodelos por rodada', 'MiB'),
    'client_time': ('Maior tempo de cliente por rodada (estimado)', 's'),
    'retention': ('Retenção média selecionada por rodada', '%'),
}


def summarize(series):
    """Align actual round coordinates; missing rounds are not interpolated."""
    rounds = sorted({r for values in series for r in values})
    mean, std = [], []
    for round_index in rounds:
        values = [s[round_index] for s in series if round_index in s]
        mean.append(np.mean(values))
        std.append(np.std(values))
    return np.asarray(rounds), np.asarray(mean), np.asarray(std)


def draw(data, title, ylabel, output, formats):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for algorithm, repetitions in data.items():
        rounds, mean, std = summarize(repetitions)
        line, = ax.plot(rounds, mean, label=f'{algorithm} (n={len(repetitions)})', linewidth=2)
        if len(repetitions) > 1:
            ax.fill_between(rounds, mean - std, mean + std, color=line.get_color(), alpha=0.16)
    ax.set(title=title, xlabel='Rodada', ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for extension in formats:
        target = output.with_suffix('.' + extension)
        fig.savefig(target, dpi=180)
        print(target)
    plt.close(fig)


def read_policy(path):
    with path.open(encoding='utf-8') as stream:
        reports = json.load(stream)
    values = {name: {} for name in RESOURCES}
    for report in reports:
        round_index = int(report['round'])
        clients = report['clients']
        if not clients:
            continue
        # All trained clients incur these costs, including rejected responses.
        stats = [c['round_stats'] for c in clients]
        values['energy'][round_index] = sum(s['energy_consumption'] for s in stats)
        values['communication'][round_index] = sum(s['total_communication_bytes'] for s in stats) / 2**20
        values['client_time'][round_index] = max(s['round_time'] for s in stats)
        values['retention'][round_index] = float(np.mean([c['retention'] for c in clients])) * 100
    return values


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=root / 'results')
    parser.add_argument('--dataset', default='MNIST')
    parser.add_argument('--goal', default='test')
    parser.add_argument('--algorithms', nargs='+', default=['FedAvg', 'FedLoad', 'HERAFL'])
    parser.add_argument('--runs', type=int, nargs='+', help='Repetition indices; default: all matching files.')
    parser.add_argument('--eval-gap', type=int, default=1, help='The -eg value used for ALL compared experiments.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png', 'pdf'])
    parser.add_argument('--extra-metrics', action='store_true', help='Also generate the original loss/AUC and per-round resource plots.')
    args = parser.parse_args()
    if args.eval_gap < 1:
        parser.error('--eval-gap must be positive.')
    if args.runs is not None and any(r < 0 for r in args.runs):
        parser.error('--runs must be nonnegative.')
    aliases = {'herafl': 'HERAFL', 'hera-fl': 'HERAFL', 'hera_fl': 'HERAFL', 'fedload': 'FedLoad'}
    algorithms = list(dict.fromkeys(aliases.get(a.lower(), a) for a in args.algorithms))
    metrics = {name: {} for name in METRICS}
    resources = {name: {} for name in RESOURCES}
    policy_runs = {}
    for algorithm in algorithms:
        prefix = f'{args.dataset}_{algorithm}_{args.goal}_'
        pattern = re.compile(re.escape(prefix) + r'(\d+)\.h5$')
        matching = []
        for path in args.results_dir.glob('*.h5'):
            match = pattern.fullmatch(path.name)
            if match and (args.runs is None or int(match[1]) in args.runs):
                matching.append((int(match[1]), path))
        matching.sort()
        if not matching:
            parser.error(f'No results for {algorithm}: expected {args.results_dir / (prefix + "<run>.h5")}')
        if args.runs is not None:
            missing = set(args.runs) - {run for run, _ in matching}
            if missing:
                parser.error(f'{algorithm}: missing repetitions {sorted(missing)}.')
        lengths = []
        policy_count = 0
        for run, path in matching:
            print(f'Reading {path}')
            with h5py.File(path, 'r') as stream:
                for name, (_, _, scale, _) in METRICS.items():
                    if name not in stream:
                        print(f'Notice: {path.name} has no {name}; skipping this metric.')
                        continue
                    values = np.asarray(stream[name], dtype=float)
                    if values.ndim != 1:
                        parser.error(f'{path.name}: invalid dimensions for {name}.')
                    if not len(values) or not np.isfinite(values).any():
                        print(f'Notice: {path.name}: no usable values for {name}; skipping.')
                        continue
                    if not np.isfinite(values).all():
                        print(f'Notice: {path.name}: nonfinite {name} values are omitted.')
                    series = {i * args.eval_gap: float(v * scale) for i, v in enumerate(values) if np.isfinite(v)}
                    metrics[name].setdefault(algorithm, []).append(series)
                    if name == 'rs_test_acc':
                        lengths.append(len(values))
            policy_path = path.with_name(path.stem + '_policy.json')
            if policy_path.exists():
                policy_count += 1
                with policy_path.open(encoding='utf-8') as stream:
                    policy_runs.setdefault(algorithm, []).append((run, json.load(stream)))
                for name, series in read_policy(policy_path).items():
                    if series:
                        resources[name].setdefault(algorithm, []).append(series)
        if lengths and len(set(lengths)) > 1:
            print(f'Notice: {algorithm} has unequal run lengths; each round uses available runs only.')
        if policy_count < len(matching):
            print(f'Notice: {algorithm}: resource logs available for {policy_count}/{len(matching)} runs.')
    output = args.output_dir or args.results_dir / 'comparison' / f'{args.dataset}_{args.goal}'
    output.mkdir(parents=True, exist_ok=True)
    global_accuracy = {algorithm: [{r: v / 100 for r, v in series.items()} for series in runs]
                       for algorithm, runs in metrics['rs_test_acc'].items()}
    render_reference_comparison(global_accuracy, policy_runs, output / 'plots', args.formats)
    if not args.extra_metrics:
        print('Resources and mean client accuracy require policy JSON; missing methods are omitted.')
        print('Total round time sums client times; it is not process wall-clock time.')
        return
    for name, (filename, title, _, unit) in METRICS.items():
        if metrics[name]:
            draw(metrics[name], f'{args.dataset} — {title}', unit, output / filename, args.formats)
    for name, (title, unit) in RESOURCES.items():
        if resources[name]:
            draw(resources[name], f'{args.dataset} — {title}', unit, output / name, args.formats)
    print('Bands: mean ± population standard deviation across repetitions.')
    print('Resource plots use policy logs only. Client time is not process wall-clock time.')


if __name__ == '__main__':
    main()
