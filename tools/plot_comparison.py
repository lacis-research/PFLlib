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
from comparison_metrics import render_reference_comparison, style


METRICS = {
    'rs_test_acc': ('accuracy', 'Acurácia de teste', 100.0, '%'),
    'rs_test_auc': ('auc', 'AUC de teste', 1.0, 'AUC'),
    'rs_train_loss': ('train_loss', 'Perda de treino', 1.0, 'Perda'),
    'rs_test_acc_std': ('test_accuracy_std', 'Desvio padrão da acurácia entre clientes', 100.0, 'p.p.'),
    'rs_test_auc_std': ('test_auc_std', 'Desvio padrão da AUC entre clientes', 1.0, 'AUC'),
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
        line, = ax.plot(rounds, mean, label=f'{algorithm} (n={len(repetitions)})', linewidth=2, color=style(algorithm)[1])
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


def render_global_metrics(metrics, evaluation_types, output, formats=('png',)):
    """Never substitute personalized/local evaluations for a global model."""
    output.mkdir(parents=True, exist_ok=True)
    for key, filename, title, ylabel in [
        ('rs_test_acc', 'global_accuracy_per_round_comparison', 'Acurácia de teste dos modelos globais', '%'),
        ('rs_train_loss', 'global_train_loss_per_round_comparison', 'Loss de treino dos modelos globais', 'Entropia cruzada'),
        ('rs_test_auc', 'global_test_auc_per_round_comparison', 'AUC de teste dos modelos globais', 'AUC')]:
        data = {a: runs for a, runs in metrics.get(key, {}).items()
                if evaluation_types.get(a) == 'global_model' and any(runs)}
        if data:
            draw(data, title, ylabel, output / filename, formats)
        else:
            print(f'Notice: no global model data for {filename}.')


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
        stats = [c['round_stats'] for c in clients if c['round_stats']]
        if not stats:
            continue
        values['energy'][round_index] = sum(s['energy_consumption'] for s in stats)
        values['communication'][round_index] = sum(s['total_communication_bytes'] for s in stats) / 2**20
        values['client_time'][round_index] = max(s['round_time'] for s in stats)
        values['retention'][round_index] = float(np.mean([c['retention'] for c in clients if c.get('retention') is not None])) * 100
    return values


def latest_common_goal(results_dir, dataset, algorithms):
    """Choose a single latest completed experiment shared by every method."""
    timestamps = {}
    for algorithm in algorithms:
        pattern = re.compile(re.escape(f'{dataset}_{algorithm}_') + r'(.+)_(\d+)\.h5$')
        goals = {}
        for path in results_dir.glob('**/*.h5'):
            match = pattern.fullmatch(path.name)
            if match:
                goal = match[1]
                goals[goal] = max(goals.get(goal, 0), path.stat().st_mtime)
        timestamps[algorithm] = goals
    common = set.intersection(*(set(goals) for goals in timestamps.values()))
    if not common:
        raise ValueError(f'No complete comparison for {dataset} shared by {", ".join(algorithms)}.')
    return max(common, key=lambda goal: max(goals[goal] for goals in timestamps.values()))


def comparison_directory_name(algorithms, dataset, goal, sources):
    """Use saved experiment metadata, including for --compare-only recovery."""
    scenarios, models = set(), set()
    for source in sources:
        config = source.parent / 'config.json'
        if not config.exists():
            continue
        metadata = json.loads(config.read_text())
        training = metadata.get('training_config') or {}
        partition = metadata.get('partition_metadata') or {}
        scenario = training.get('scenario') or metadata.get('scenario')
        if 'non_iid' in partition:
            scenario = 'non-iid' if partition['non_iid'] else 'iid'
        if scenario:
            scenarios.add(str(scenario).replace('non_iid', 'non-iid').replace('noniid', 'non-iid'))
        model = metadata.get('model')
        if model:
            models.add(str(model))
    safe = lambda value: re.sub(r'[^a-zA-Z0-9_.-]', '_', value)
    scenario = '+'.join(sorted(scenarios)) or 'scenario-unknown'
    model = '+'.join(sorted(models)) or 'model-unknown'
    return '__vs__'.join(algorithms) + f'_{safe(dataset)}_{safe(scenario)}_{safe(model)}_{safe(goal)}'


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=root / 'results')
    parser.add_argument('--dataset', default='MNIST')
    parser.add_argument('--goal', help='Experiment goal; default: test.')
    parser.add_argument('--latest', action='store_true', help='Compare the latest goal shared by all requested methods.')
    parser.add_argument('--algorithms', nargs='+', default=['FedAvg', 'FedLoad', 'HERAFL'])
    parser.add_argument('--runs', type=int, nargs='+', help='Repetition indices; default: all matching files.')
    parser.add_argument('--eval-gap', type=int, default=1, help='The -eg value used for ALL compared experiments.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--accuracy-target', type=float, help='Optional common accuracy target in [0,1].')
    parser.add_argument('--formats', nargs='+', choices=['png'], default=['png'])
    parser.add_argument('--extra-metrics', action='store_true', help='Also generate the original loss/AUC and per-round resource plots.')
    args = parser.parse_args()
    if args.eval_gap < 1:
        parser.error('--eval-gap must be positive.')
    if args.runs is not None and any(r < 0 for r in args.runs):
        parser.error('--runs must be nonnegative.')
    aliases = {'herafl': 'HERAFL', 'hera-fl': 'HERAFL', 'hera_fl': 'HERAFL', 'fedload': 'FedLoad', 'fedavg': 'FedAvg'}
    algorithms = list(dict.fromkeys(aliases.get(a.lower(), a) for a in args.algorithms))
    if args.latest and args.goal is not None:
        parser.error('--latest and --goal are mutually exclusive.')
    if args.latest:
        try:
            args.goal = latest_common_goal(args.results_dir, args.dataset, algorithms)
        except ValueError as error:
            parser.error(str(error))
    else:
        args.goal = args.goal or 'test'
    print(f'Comparing dataset={args.dataset}, goal={args.goal}')
    metrics = {name: {} for name in METRICS}
    resources = {name: {} for name in RESOURCES}
    policy_runs = {}
    contribution_runs = {}
    evaluation_types = {}
    variant_types = {}
    sources = []
    for algorithm in algorithms:
        prefix = f'{args.dataset}_{algorithm}_{args.goal}_'
        pattern = re.compile(re.escape(prefix) + r'(\d+)\.h5$')
        matching = []
        for path in args.results_dir.glob('**/*.h5'):
            match = pattern.fullmatch(path.name)
            if match and (args.runs is None or int(match[1]) in args.runs):
                matching.append((int(match[1]), path))
        latest = {}
        for run, path in matching:
            sources.append(path)
            accuracy = {}
            if run not in latest or path.stat().st_mtime > latest[run].stat().st_mtime:
                latest[run] = path
        matching = sorted(latest.items())
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
                evaluation_type = stream.attrs.get('evaluation_type', 'global_model')
                variant = stream.attrs.get('variant', 'legacy')
                if variant_types.setdefault(algorithm, variant) != variant:
                    parser.error(f'{algorithm}: cannot mix distinct variants as repetitions.')
                previous_type = evaluation_types.setdefault(algorithm, evaluation_type)
                if previous_type != evaluation_type:
                    parser.error(f'{algorithm}: cannot mix global and local evaluation runs.')
                for name, (_, _, scale, _) in METRICS.items():
                    source_name = 'rs_local_test_acc' if name == 'rs_test_acc' and evaluation_type == 'local_models_sample_weighted' else name
                    if source_name not in stream:
                        print(f'Notice: {path.name} has no {name}; skipping this metric.')
                        continue
                    values = np.asarray(stream[source_name], dtype=float)
                    if values.ndim != 1:
                        parser.error(f'{path.name}: invalid dimensions for {name}.')
                    if not len(values) or not np.isfinite(values).any():
                        print(f'Notice: {path.name}: no usable values for {name}; skipping.')
                        continue
                    if not np.isfinite(values).all():
                        print(f'Notice: {path.name}: nonfinite {name} values are omitted.')
                    rounds = np.asarray(stream['eval_rounds']) if 'eval_rounds' in stream else np.arange(len(values)) * args.eval_gap
                    series = {int(r): float(v * scale) for r, v in zip(rounds, values) if np.isfinite(v)}
                    metrics[name].setdefault(algorithm, []).append(series)
                    if name == 'rs_test_acc':
                        accuracy = {r: v / 100 for r, v in series.items()}
                        lengths.append(len(values))
            policy_path = path.with_name(path.stem + '_policy.json')
            reports = []
            if policy_path.exists():
                policy_count += 1
                with policy_path.open(encoding='utf-8') as stream:
                    reports = json.load(stream)
                    policy_runs.setdefault(algorithm, []).append((run, reports))
                for name, series in read_policy(policy_path).items():
                    if series:
                        resources[name].setdefault(algorithm, []).append(series)
            label = algorithm + (' (modelos locais)' if evaluation_type == 'local_models_sample_weighted' else ' (modelo global)')
            contribution_runs.setdefault(label, []).append(
                dict(id=f'run {run}', accuracy=accuracy, reports=reports, evaluation_type=evaluation_type))
        if lengths and len(set(lengths)) > 1:
            print(f'Notice: {algorithm} has unequal run lengths; each round uses available runs only.')
        if policy_count < len(matching):
            print(f'Notice: {algorithm}: resource logs available for {policy_count}/{len(matching)} runs.')
    output = args.output_dir or args.results_dir / 'comparison' / comparison_directory_name(algorithms, args.dataset, args.goal, sources)
    output.mkdir(parents=True, exist_ok=True)
    global_accuracy = {algorithm: [{r: v / 100 for r, v in series.items()} for series in runs]
                       for algorithm, runs in metrics['rs_test_acc'].items()}
    render_reference_comparison(global_accuracy, policy_runs, output / 'plots', args.formats,
                                evaluation_types=evaluation_types)
    render_global_metrics(metrics, evaluation_types, output / 'plots', args.formats)
    print('Costs include all recorded client work. Coverage differs by protocol; accumulated client time is not wall-clock duration.')


if __name__ == '__main__':
    main()
