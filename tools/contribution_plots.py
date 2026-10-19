"""Contribution views; costs include all logged clients, including rejected ones."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from comparison_metrics import save

COSTS = {
    'round_time': ('Tempo acumulado dos clientes (s; medido + transferência estimada)', 1),
    'energy_consumption': ('Energia acumulada estimada (J)', 1),
    'total_communication_bytes': ('Comunicação acumulada modelada (MiB)', 2**20),
}


def cumulative_cost(reports, metric):
    """Missing/invalid cost or a missing round invalidates subsequent totals."""
    result, total, previous = {0: 0.0}, 0.0, 0
    for report in sorted(reports, key=lambda r: int(r['round'])):
        index = int(report['round'])
        if index != previous + 1:
            break
        values = [c.get('round_stats', {}).get(metric) for c in report['clients'] if c.get('status') != 'infeasible']
        if any(v is None or not np.isfinite(v) or v < 0 for v in values):
            break
        extra = {'round_time': 'initialization_time', 'energy_consumption': 'initialization_energy', 'total_communication_bytes': 'initialization_bytes'}[metric]
        total += sum(values) + report.get(extra, 0)
        evaluation_key = {'round_time': 'evaluation_total_time', 'energy_consumption': 'evaluation_energy', 'total_communication_bytes': 'evaluation_telemetry_bytes'}[metric]
        total += report.get(evaluation_key, 0)
        result[index] = total
        previous = index
    return result


def draw_cost_accuracy(runs, output, formats):
    for metric, (label, scale) in COSTS.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        count = 0
        for name, repetitions in runs.items():
            color = None
            for run in repetitions:
                costs = cumulative_cost(run['reports'], metric)
                rounds = sorted(set(costs) & set(run['accuracy']))
                if not rounds:
                    continue
                line, = ax.plot([costs[r]/scale for r in rounds],
                                [100*run['accuracy'][r] for r in rounds],
                                label=f"{name} — {run['id']}", color=color, alpha=.8)
                color = line.get_color()
                count += 1
        if count:
            ax.set(xlabel=label, ylabel='Acurácia de teste (%; protocolo na legenda)',
                   title='Qualidade por orçamento — trajetórias individuais')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            save(fig, output, 'accuracy_vs_cumulative_' + metric, formats)
        else:
            plt.close(fig)


def draw_retention(runs, output, formats):
    entries = [(name, run) for name, repetitions in runs.items() for run in repetitions if run['reports']]
    if not entries:
        return
    clients = sorted({c['id'] for _, run in entries for r in run['reports'] for c in r['clients']})
    rounds = sorted({int(r['round']) for _, run in entries for r in run['reports']})
    if not clients or not rounds:
        return
    for field in ('retention', 'parameter_fraction'):
        panels = []
        for name, run in entries:
            matrix = np.full((len(clients), len(rounds)), np.nan)
            for report in run['reports']:
                for client in report['clients']:
                    value = client.get('retention') if field == 'retention' else client.get('round_stats', {}).get(field)
                    if value is not None:
                        matrix[clients.index(client['id']), rounds.index(int(report['round']))] = value
            if np.isfinite(matrix).any():
                panels.append((f"{name} — {run['id']}", matrix))
        if not panels:
            print(f'Notice: no recorded {field}; heatmap omitted.')
            continue
        fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3.5*len(panels)), squeeze=False, constrained_layout=True)
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('#eeeeee')
        for ax, (title, matrix) in zip(axes[:, 0], panels):
            im = ax.imshow(matrix, aspect='auto', vmin=0, vmax=1, cmap=cmap, interpolation='nearest')
            ticks = np.arange(0, len(rounds), max(1, len(rounds)//12))
            ax.set(title=title, xlabel='Rodada', ylabel='Cliente', xticks=ticks,
                   xticklabels=[rounds[i] for i in ticks], yticks=range(len(clients)), yticklabels=clients)
        label = 'Retenção nominal' if field == 'retention' else 'Fração real de parâmetros'
        fig.colorbar(im, ax=list(axes[:, 0]), label=label + ' (0–1); cinza: sem registro')
        save(fig, output, field + '_heatmap', formats)


def cost_components(reports):
    """Do not invent a semantic/training split for older logs."""
    totals = dict(training=0., extraction=0., unsplit_compute=0., other_compute=0., transfer=0., initialization=0., initialization_time=0., evaluation_bytes=0., evaluation_time=0.,
                  parameters=0., semantic=0., other_bytes=0., server=0.)
    for report in reports:
        totals['server'] += report.get('semantic_aggregation_time', 0.)
        totals['initialization'] += report.get('initialization_bytes', 0)
        totals['initialization_time'] += report.get('initialization_time', 0)
        totals['evaluation_bytes'] += report.get('evaluation_telemetry_bytes', 0)
        totals['evaluation_time'] += report.get('evaluation_total_time', 0)
        for client in report['clients']:
            if client.get('status') == 'infeasible':
                continue
            s = client.get('round_stats', {})
            required = ('compute_time', 'download_time', 'upload_time', 'total_communication_bytes')
            if any(k not in s for k in required):
                return None
            if 'training_time' in s and 'semantic_extraction_time' in s:
                totals['training'] += s['training_time']
                totals['extraction'] += s['semantic_extraction_time']
                totals['other_compute'] += s['compute_time']-s['training_time']-s['semantic_extraction_time']
            else:
                totals['unsplit_compute'] += s['compute_time']
            totals['transfer'] += s['download_time'] + s['upload_time']
            parameters = s.get('parameter_download_bytes', 0) + s.get('parameter_upload_bytes', 0)
            semantic = s.get('semantic_download_bytes', 0) + s.get('semantic_upload_bytes', 0)
            totals['parameters'] += parameters
            totals['semantic'] += semantic
            totals['other_bytes'] += s['total_communication_bytes'] - parameters - semantic
    return totals


def draw_components(runs, output, formats):
    entries = [(name, run['id'], cost_components(run['reports'])) for name, reps in runs.items()
               for run in reps if run['reports']]
    entries = [e for e in entries if e[2] is not None]
    if not entries:
        return
    groups = [
        ('client_time_breakdown', [('training', 'Treino medido'), ('extraction', 'Extração semântica medida'),
         ('unsplit_compute', 'Computação medida sem decomposição'), ('other_compute', 'Poda/validação medidas'), ('transfer', 'Transferência estimada'), ('initialization_time', 'Distribuição inicial estimada'), ('evaluation_time', 'Avaliação + telemetria')], 'Soma dos tempos dos clientes (s)', 1),
        ('communication_breakdown', [('parameters', 'Parâmetros'), ('semantic', 'Semântica'),
         ('other_bytes', 'Controle / outros / sem decomposição'), ('initialization', 'Pesos iniciais'), ('evaluation_bytes', 'Telemetria de avaliação')], 'Comunicação modelada (MiB)', 2**20),
        ('semantic_server_time', [('server', 'Agregação semântica registrada')], 'Tempo medido no servidor (s)', 1),
    ]
    for filename, components, ylabel, scale in groups:
        if filename == 'semantic_server_time' and not any(e[2]['server'] for e in entries):
            continue
        fig, ax = plt.subplots(figsize=(max(9, len(entries)*1.1), 5))
        bottom = np.zeros(len(entries))
        for key, label in components:
            values = np.array([e[2][key]/scale for e in entries])
            ax.bar(range(len(entries)), values, bottom=bottom, label=label)
            bottom += values
        ax.set(xticks=range(len(entries)), xticklabels=[f'{a}\n{r}' for a, r, _ in entries], ylabel=ylabel)
        ax.tick_params(axis='x', labelrotation=25)
        ax.margins(y=.12)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save(fig, output, filename, formats)


def draw_ablations(runs, output, formats):
    # Compare a common evaluation round rather than unequal final checkpoints.
    series = [run['accuracy'] for reps in runs.values() for run in reps]
    common = set.intersection(*(set(s) for s in series)) if series else set()
    if not common:
        raise ValueError('Ablations have no common evaluation round.')
    final = max(common)
    fields = [('accuracy', 'Acurácia de teste (%)', 1)] + [(k, v[0], v[1]) for k, v in COSTS.items()]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    rows = []
    for ax, (metric, label, scale) in zip(axes.flat, fields):
        for i, (name, reps) in enumerate(runs.items()):
            values = []
            for run in reps:
                value = 100*run['accuracy'][final] if metric == 'accuracy' else cumulative_cost(run['reports'], metric).get(final)
                if value is not None:
                    value /= scale
                    values.append(value)
                    rows.append(dict(variant=name, run=run['id'], round=final, metric=metric, value=value))
            if values:
                ax.scatter([i]*len(values), values, alpha=.5)
                ax.errorbar(i, np.mean(values), yerr=np.std(values), fmt='D', color='black', capsize=4)
                ax.annotate(f'n={len(values)}', (i, np.mean(values)), xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax.set(xticks=range(len(runs)), xticklabels=list(runs), ylabel=label)
        ax.tick_params(axis='x', labelrotation=30)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(f'Ablações na rodada comum {final} — média ± desvio padrão; pontos: execuções')
    fig.tight_layout()
    save(fig, output, 'ablation_summary', formats)
    with (output/'ablation_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['variant', 'run', 'round', 'metric', 'value'])
        writer.writeheader()
        writer.writerows(rows)


def draw_client_distribution(runs, output, formats):
    entries = []
    for name, repetitions in runs.items():
        for run in repetitions:
            evaluated = [r for r in run['reports'] if r.get('test_clients')]
            if evaluated:
                report = evaluated[-1]
                values = [100*c['accuracy'] for c in report['test_clients'] if c.get('accuracy') is not None]
                if values:
                    entries.append((f"{name}\n{run['id']} / rodada {report['round']}", values))
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(max(9, len(entries)*1.4), 5))
    ax.boxplot([v for _, v in entries])
    for i, (_, values) in enumerate(entries, 1):
        ax.scatter(np.full(len(values), i), values, alpha=.4, s=12)
    ax.set(xticks=range(1, len(entries)+1), xticklabels=[k for k, _ in entries],
           ylabel='Acurácia por cliente (%)', title='Distribuição na última avaliação disponível de cada execução')
    ax.tick_params(axis='x', labelrotation=20)
    fig.tight_layout()
    save(fig, output, 'client_accuracy_distribution', formats)


def draw_target_cost(runs, output, formats, target):
    if target is None:
        return
    if not 0 <= target <= 1:
        raise ValueError('Accuracy target must be in [0,1].')
    entries = [(name, run) for name, reps in runs.items() for run in reps]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    rows = []
    for ax, (metric, (label, scale)) in zip(axes, COSTS.items()):
        for i, (name, run) in enumerate(entries):
            reached = sorted(r for r, a in run['accuracy'].items() if a >= target)
            r = reached[0] if reached else None
            value = cumulative_cost(run['reports'], metric).get(r) if r is not None else None
            if value is not None:
                ax.bar(i, value/scale)
            else:
                ax.text(i, .02, 'Não atingiu' if r is None else 'Custo ausente',
                        transform=ax.get_xaxis_transform(), rotation=90, va='bottom', ha='center', fontsize=8)
            rows.append(dict(variant=name, run=run['id'], metric=metric, target=target,
                             round=r, cost=value/scale if value is not None else None,
                             status='not_reached' if r is None else ('missing_cost' if value is None else 'reached')))
        short_label = {'round_time': 'Tempo acumulado (s)', 'energy_consumption': 'Energia estimada (J)', 'total_communication_bytes': 'Comunicação (MiB)'}[metric]
        ax.set(xticks=range(len(entries)), xticklabels=[f"{n} / {r['id']}" for n, r in entries], ylabel=short_label, xlim=(-.6, max(len(entries)-.4, .6)))
        ax.tick_params(axis='x', labelrotation=60)
    fig.suptitle(f'Custo até a primeira avaliação com acurácia ≥ {100*target:g}%')
    fig.tight_layout()
    save(fig, output, 'cost_to_accuracy_target', formats)
    if rows:
        with (output/'cost_to_accuracy_target.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def render_contributions(runs, output, formats=('png',), ablations=False, accuracy_target=None):
    output.mkdir(parents=True, exist_ok=True)
    draw_cost_accuracy(runs, output, formats)
    draw_retention(runs, output, formats)
    draw_components(runs, output, formats)
    draw_client_distribution(runs, output, formats)
    draw_target_cost(runs, output, formats, accuracy_target)
    if ablations:
        draw_ablations(runs, output, formats)
    (output/'contribution_notes.txt').write_text(
        'Costs include all logged clients, including rejected responses, initialization and recorded evaluation telemetry.\n'
        'Client time sums measured compute and estimated transfer; it is not wall-clock time.\n'
        'Server semantic aggregation time is separate. Energy and communication use the profile model.\n'
        'Gray heatmap cells are unavailable, not zero. Missing breakdowns are not inferred.\n'
        'Cost curves show individual runs without interpolation. Ablations use the latest common evaluation round.\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Plot explicit ablation groups from a JSON manifest: {label: [HDF5 paths]}.')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--accuracy-target', type=float, help='Optional common test accuracy target in [0,1].')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    runs = {}
    seen = set()
    types = {}
    for label, paths in manifest.items():
        if not isinstance(paths, list) or not paths:
            parser.error(f'{label}: expected a nonempty list of HDF5 paths.')
        runs[label] = []
        for index, entry in enumerate(paths):
            path = (args.manifest.parent / entry).resolve()
            if path in seen:
                parser.error(f'Duplicate run: {path}')
            seen.add(path)
            with h5py.File(path) as stream:
                if 'eval_rounds' not in stream:
                    parser.error(f'{path}: eval_rounds required for ablations.')
                evaluation_type = str(stream.attrs.get('evaluation_type', 'global_model'))
                if label in types and types[label] != evaluation_type:
                    parser.error(f'{label}: cannot mix local and global models as repetitions.')
                types[label] = evaluation_type
                accuracy = dict(zip(map(int, stream['eval_rounds']), map(float, stream['rs_local_test_acc'] if 'rs_local_test_acc' in stream else stream['rs_test_acc'])))
                if not accuracy or not all(np.isfinite(v) for v in accuracy.values()):
                    parser.error(f'{path}: invalid accuracy.')
            policy = path.with_name(path.stem + '_policy.json')
            reports = json.loads(policy.read_text()) if policy.exists() else []
            runs[label].append(dict(id=f'run {index}', accuracy=accuracy, reports=reports))
    if len(runs) < 2:
        parser.error('At least two explicitly labeled variants are required.')
    labeled_runs = {label + (' (locais)' if types[label] == 'local_models_sample_weighted' else ' (global)'): reps for label, reps in runs.items()}
    render_contributions(labeled_runs, args.output_dir, ablations=True, accuracy_target=args.accuracy_target)


if __name__ == '__main__':
    main()
