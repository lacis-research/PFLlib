"""The six metric views used by MininetFed's results/comparison."""
import matplotlib.pyplot as plt
import numpy as np

STYLES = {
    'HERAFL': ('Hera-FL', '#ff7f0e', 's', '--'),
    'FedLoad': ('FedLoad', '#1f77b4', 'o', '-'),
    'FedAvg': ('FedAvg', '#2ca02c', '^', '-.'),
    'FedMP': ('FedMP', '#d62728', 'D', ':'),
}
TOTAL_METRICS = {
    'energy_consumption': ('total_energy_per_experiment_comparison', 'Total Energy Per Experiment Comparison', 'Recorded Estimated Client Energy (J)'),
    'round_time': ('total_round_time_per_experiment_comparison', 'Accumulated Client Time Per Experiment', 'Sum of Client Times (s; compute + estimated transfer)'),
    'total_communication_bytes': ('total_communication_per_experiment_comparison', 'Total Communication Per Experiment Comparison', 'Recorded Modeled Communication (bytes)'),
}


def style(algorithm):
    for name, values in STYLES.items():
        normalized = algorithm.lower().replace('-', '').replace('_', '')
        if normalized.startswith(name.lower()):
            return values
    return (algorithm, None, 'o', '-')


def save(fig, output, filename, formats):
    for extension in formats:
        path = output / (filename + '.' + extension)
        fig.savefig(path, dpi=150)
        print(path)
    plt.close(fig)


def experiment_total(reports, metric, include_all=False):
    if include_all:
        initialization = {'round_time': 'initialization_time', 'energy_consumption': 'initialization_energy', 'total_communication_bytes': 'initialization_bytes'}[metric]
        evaluation = {'round_time': 'evaluation_total_time', 'energy_consumption': 'evaluation_energy', 'total_communication_bytes': 'evaluation_telemetry_bytes'}[metric]
        return sum(sum(c['round_stats'][metric] for c in r['clients'] if c.get('round_stats'))
                   + r.get(initialization, 0) + (r.get('evaluation_communication_bytes', r.get(evaluation, 0))
                     if metric == 'total_communication_bytes' else r.get(evaluation, 0)) for r in reports)
    return sum(c['round_stats'][metric] for r in reports for c in r['clients'] if c.get('status', 'accepted') == 'accepted')


def draw_lines(data, output, filename, title, ylabel, formats, labels=None):
    if not data:
        print(f'Notice: no data for {filename}.')
        return
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for algorithm, repetitions in data.items():
        label, color, marker, linestyle = style(algorithm)
        label = (labels or {}).get(algorithm, label)
        rounds = sorted({r for series in repetitions for r in series})
        values = [[series[r] for series in repetitions if r in series] for r in rounds]
        mean, std = [np.mean(v) for v in values], [np.std(v) for v in values]
        line, = ax.plot(rounds, mean, label=label, color=color, marker=marker, linestyle=linestyle, linewidth=2)
        if len(repetitions) > 1:
            ax.fill_between(rounds, np.asarray(mean)-std, np.asarray(mean)+std, color=line.get_color(), alpha=0.15)
    ax.set(title=title, xlabel='Round', ylabel=ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, output, filename, formats)


def draw_totals(policy_runs, output, formats, include_all=False):
    for metric, (filename, title, unit) in TOTAL_METRICS.items():
        entries = [(algorithm, [experiment_total(reports, metric, include_all) for _, reports in runs])
                   for algorithm, runs in policy_runs.items() if runs]
        if not entries:
            continue
        means = [float(np.mean(totals)) for _, totals in entries]
        errors = [float(np.std(totals)) for _, totals in entries]
        labels = [style(a)[0] for a, _ in entries]
        colors = [style(a)[1] or '#7f7f7f' for a, _ in entries]
        reference = next((m for (a, _), m in zip(entries, means) if a == 'HERAFL'), None)
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.bar(labels, means, color=colors, width=0.58, yerr=errors, capsize=4)
        for (algorithm, _), bar, value, error in zip(entries, bars, means, errors):
            text = f'{value:,.2f}'
            if algorithm != 'HERAFL' and reference is not None and reference > 0:
                text += f'\n({(value-reference)/reference*100:+.1f}%)'
            ax.annotate(text, xy=(bar.get_x()+bar.get_width()/2, value+error),
                        xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
        ax.set(title=title, ylabel=unit)
        ax.margins(y=0.2)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        save(fig, output, filename, formats)


def draw_capacity(policy_runs, output, formats):
    panels = []
    for algorithm, runs in policy_runs.items():
        for run, reports in runs:
            points = []
            for report in reports:
                for client in report['clients']:
                    context = client.get('context', client.get('selection_context', {}))
                    capacity = context.get('cpu_capacity', context.get('f_i'))
                    if capacity is not None and client.get('retention') is not None:
                        points.append((report['round'], client['id'], client['retention'], float(capacity)))
            if points:
                title = style(algorithm)[0] + (f' — run {run}' if len(runs) > 1 else '')
                panels.append((title, points))
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 4.6*len(panels)), squeeze=False, constrained_layout=True)
    scatter = None
    for ax, (title, points) in zip(axes[:, 0], panels):
        clients = sorted({p[1] for p in points})
        positions = {client: i for i, client in enumerate(clients)}
        for low, high, marker, label in [(0, 0.33, 's', 'Low capacity'), (0.33, 0.66, 'o', 'Medium capacity'), (0.66, 1.01, '^', 'High capacity')]:
            group = [p for p in points if low <= p[3] < high]
            if group:
                scatter = ax.scatter([p[0] for p in group], [positions[p[1]] for p in group],
                                     c=[p[2] for p in group], s=170, cmap='viridis', vmin=0.0625, vmax=1,
                                     marker=marker, edgecolors='black', linewidths=0.35, alpha=0.85, label=label)
        ax.set(title=title, xlabel='Round', ylabel='Client', yticks=range(len(clients)), yticklabels=clients)
        rounds = sorted({p[0] for p in points})
        ticks = rounds[::max(1, len(rounds)//10)]
        if rounds[-1] not in ticks:
            ticks.append(rounds[-1])
        ax.set_xticks(ticks)
        ax.grid(alpha=0.2)
        ax.legend(title='Compute Capacity (normalized, 0–1)', loc='upper right')
    if scatter is not None:
        fig.colorbar(scatter, ax=list(axes[:, 0]), label='Pruning Rate / Model Retention (fraction, 0–1)')
    fig.suptitle('Client, Round, Compute Capacity and Pruning Rate Comparison')
    save(fig, output, 'client_round_compute_capacity_pruning_bubble_comparison', formats)


def render_reference_comparison(global_accuracy, policy_runs, output, formats, evaluation_types=None):
    output.mkdir(parents=True, exist_ok=True)
    draw_totals(policy_runs, output, formats, include_all=True)
    if set(global_accuracy).issubset(policy_runs):
        render_local_test_metrics(policy_runs, output, formats)


def render_local_test_metrics(policy_runs, output, formats=('png',)):
    output.mkdir(parents=True, exist_ok=True)
    if len(policy_runs) > 1 and any(c.get('local_test', {}).get('protocol') == 'local-test-v1'
            for runs in policy_runs.values() for _, reports in runs for r in reports for c in r['clients']):
        note = ('Legacy local-test-v1 hashes depend on shuffled sample order and cannot verify partition identity. '
                'Comparison matches repetition, round, client ID and sample count; identical test partitions are assumed. '
                'New local-test-v2 runs verify order-independent hashes.\n')
        print('Notice: ' + note.strip())
        (output / 'local_test_comparison_notes.txt').write_text(note, encoding='utf-8')
    for metric, filename, title, ylabel in [
        ('correct', 'local_test_accuracy_per_round_comparison', 'Acurácia de teste dos modelos locais', 'Acurácia'),
        ('loss_sum', 'local_test_loss_per_round_comparison', 'Loss de teste dos modelos locais', 'Entropia cruzada'),
        ('auc', 'local_test_auc_per_round_comparison', 'AUC de teste dos modelos locais', 'Micro-AUC por cliente, ponderada por amostras')]:
        data = matched_local_accuracy(policy_runs, metric=metric, strict=True)
        draw_lines(data, output, filename, title + ' — ' + ' × '.join(policy_runs), ylabel, formats)


def matched_local_accuracy(policy_runs, metric="correct", strict=False):
    indexed = {a: {run: {r['round']: {c['id']: c['local_test'] for c in r['clients']
                if c.get('local_test', {}).get('samples', 0) > 0} for r in reports}
                for run, reports in runs} for a, runs in policy_runs.items()}
    if not indexed:
        return {}
    result = {a: [] for a in indexed}
    for run in sorted(set.intersection(*(set(v) for v in indexed.values()))):
        series = {a: {} for a in indexed}
        for rnd in sorted(set.intersection(*(set(v[run]) for v in indexed.values()))):
            records = [v[run][rnd] for v in indexed.values()]
            common = set.intersection(*(set(c) for c in records))
            common = {i for i in common if len({c[i]['samples'] for c in records}) == 1}
            if strict:
                common = {i for i in common if all(c[i].get('protocol') in ('local-test-v1', 'local-test-v2') and
                          c[i].get('partition_sha256') and all(c[i].get(k) is not None and np.isfinite(c[i][k])
                          for k in ('correct', 'loss_sum', 'auc')) for c in records) and
                          (all(c[i]['protocol'] == 'local-test-v1' for c in records) or
                           (all(c[i]['protocol'] == 'local-test-v2' for c in records) and
                            len({c[i]['partition_sha256'] for c in records}) == 1))}
            if not common:
                continue
            for a, v in indexed.items():
                series[a][rnd] = sum(v[run][rnd][i][metric] * (v[run][rnd][i]['samples'] if metric == 'auc' else 1) for i in common)/sum(v[run][rnd][i]['samples'] for i in common)
        for a in series:
            if series[a]:
                result[a].append(series[a])
    return {a: v for a, v in result.items() if v}
