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
    'energy_consumption': ('total_energy_per_experiment_comparison', 'Total Energy Per Experiment Comparison', 'Total Energy Consumption (J)'),
    'round_time': ('total_round_time_per_experiment_comparison', 'Total Round Time Per Experiment Comparison', 'Total Round Time (s)'),
    'total_communication_bytes': ('total_communication_per_experiment_comparison', 'Total Communication Per Experiment Comparison', 'Total Communication (bytes)'),
}


def style(algorithm):
    return STYLES.get(algorithm, (algorithm, None, 'o', '-'))


def save(fig, output, filename, formats):
    for extension in formats:
        path = output / (filename + '.' + extension)
        fig.savefig(path, dpi=150)
        print(path)
    plt.close(fig)


def mean_accuracy(reports):
    result = {}
    for report in reports:
        # The source averages responding clients, without weighting by dataset size.
        accuracies = [c['accuracy'] for c in report['clients']
                      if c.get('status') == 'accepted' and c.get('accuracy') is not None]
        if accuracies:
            result[int(report['round'])] = float(np.mean(accuracies))
    return result


def experiment_total(reports, metric):
    return sum(c['round_stats'][metric] for r in reports for c in r['clients'])


def draw_lines(data, output, filename, title, ylabel, formats):
    if not data:
        print(f'Notice: no data for {filename}.')
        return
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for algorithm, repetitions in data.items():
        label, color, marker, linestyle = style(algorithm)
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


def draw_totals(policy_runs, output, formats):
    for metric, (filename, title, unit) in TOTAL_METRICS.items():
        entries = [(algorithm, [experiment_total(reports, metric) for _, reports in runs])
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
                    context = client.get('selection_context', {})
                    capacity = context.get('cpu_capacity', context.get('f_i'))
                    if capacity is not None:
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


def render_reference_comparison(global_accuracy, policy_runs, output, formats):
    output.mkdir(parents=True, exist_ok=True)
    local_accuracy = {}
    for algorithm, runs in policy_runs.items():
        series = [mean_accuracy(reports) for _, reports in runs]
        series = [s for s in series if s]
        if series:
            local_accuracy[algorithm] = series
    draw_lines(local_accuracy, output, 'mean_accuracy_per_round_comparison',
               'Mean Accuracy Per Round Comparison', 'Mean Accuracy (fraction, 0–1)', formats)
    draw_lines(global_accuracy, output, 'global_accuracy_per_round_comparison',
               'Global Accuracy Per Round Comparison', 'Global Accuracy (fraction, 0–1)', formats)
    draw_capacity(policy_runs, output, formats)
    draw_totals(policy_runs, output, formats)
