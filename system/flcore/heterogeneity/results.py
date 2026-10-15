"""Mininet-style method/execution/analysis directories."""
import csv
import math
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path


def result_directory(server):
    if not hasattr(server, 'result_directory'):
        method = server.algorithm.lower()
        config = Path('../dataset') / server.dataset / 'config.json'
        metadata = json.loads(config.read_text()) if config.exists() else {}
        scenario = ('non_iid' if metadata['non_iid'] else 'iid') if 'non_iid' in metadata else getattr(server, 'scenario', 'iid').replace('-', '_')
        stamp = datetime.now(ZoneInfo('America/Belem')).strftime('%Y%m%d_%Hh%Mm%Ss_%f')
        name = 'hera_fl' if method == 'herafl' else method
        server.result_directory = Path('../results') / method / f'{stamp}_{name}_experiment_{server.dataset.lower()}_{scenario}_run{server.times}'
    server.result_directory.mkdir(parents=True, exist_ok=True)
    return server.result_directory


def save_policy_results(server):
    directory = result_directory(server)
    stem = f'{server.dataset}_{server.algorithm}_{server.goal}_{server.times}'
    aligned = getattr(server, 'aligned_comparison', False)
    metadata = dict(aligned_comparison=aligned, synchronous=True,
                    effective_seed=getattr(server, 'experiment_seed', getattr(server.args, 'policy_seed', 42) + server.times),
                    wait_for_all=getattr(server, 'wait_for_all', False),
                    round_timeout=None if math.isinf(getattr(server, 'base_round_timeout', 0)) else getattr(server, 'base_round_timeout', None),
                    time_select=server.time_select,
                    selection='uniform_shared_schedule' if aligned else ('stratified' if server.algorithm == 'HERAFL' else 'native'),
                    evaluation='after_aggregation' if aligned or server.algorithm in ('FedLoad', 'HERAFL') else 'before_aggregation',
                    join_ratio=server.join_ratio, client_drop_rate=server.client_drop_rate,
                    learning_rate_decay=server.args.learning_rate_decay,
                    learning_rate_decay_gamma=server.args.learning_rate_decay_gamma,
                    pruning_arms=server.bandit.pruning_rates if hasattr(server, 'bandit') else [1.0],
                    bandit_alpha=getattr(server.args, 'bandit_alpha', None),
                    bandit_beta=getattr(server.args, 'bandit_beta', None),
                    bandit_alpha_strategy=getattr(server.args, 'bandit_alpha_strategy', 'binary'),
                    reward_weights=getattr(server.args, 'reward_weights', None),
                    straggler_predictor=getattr(server, 'straggler_enabled', False),
                    algorithm=server.algorithm, dataset=server.dataset, run=server.times,
                    model=type(server.global_model).__name__, num_classes=server.num_classes,
                    resource_preset=getattr(server.args, "resource_preset", "mininet"),
                    seed=getattr(server.args, 'policy_seed', 42), num_clients=server.num_clients,
                    global_rounds=server.global_rounds, local_epochs=server.local_epochs,
                    batch_size=server.batch_size, learning_rate=server.learning_rate,
                    client_profiles={str(c.id): c.profile for c in server.clients})
    if server.algorithm == 'HERAFL':
        metadata.update(hera_retention_floors=server.bandit.retention_floors,
                        hera_capacity_thresholds=server.bandit.capacity_thresholds,
                        policy_learning_context='selection_context')
    (directory / 'config.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    (directory / f'{stem}_policy.json').write_text(json.dumps(server.round_reports, indent=2, allow_nan=False))
    analysis = directory / 'analysis' / stem
    csv_dir = analysis / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    clients, summary, selections = [], [], []
    for report in server.round_reports:
        accepted = [c for c in report['clients'] if c['status'] == 'accepted']
        stats = [c['round_stats'] for c in accepted]
        summary.append(dict(round=report['round'], mean_accuracy=sum(c['accuracy'] for c in accepted) / len(accepted) if accepted else 0,
                            global_accuracy=report.get('global_accuracy'),
                            round_time_reference=sum(s['round_time'] for s in stats) / len(stats) if stats else 0,
                            communication_reference=sum(s['total_communication_bytes'] for s in stats) / len(stats) if stats else 0))
        for c in report['clients']:
            row = dict(round=report['round'], id=c['id'], pruning_rate=c['retention'], accuracy=c['accuracy'],
                       reward=c.get('reward', 0), num_samples=c['num_samples'],
                       status='completed' if c['status']=='accepted' else c['status'], responded=c['status']=='accepted')
            row.update({'round_stats.' + k: v for k, v in c['round_stats'].items()})
            row.update({'context.' + key: value for key, value in c.get('context', c['selection_context']).items()})
            clients.append(row)
            selection = {key: row[key] for key in ('round', 'id', 'pruning_rate')}
            selection.update({'context.' + key: value for key, value in c['selection_context'].items()})
            selections.append(selection)
    for name, rows in [('round_clients', clients), ('round_summary', summary), ('selections', selections)]:
        if rows:
            with (csv_dir / f'{name}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
                writer.writeheader()
                writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
    from comparison_metrics import render_reference_comparison
    global_series = {r['round']: r['global_accuracy'] for r in server.round_reports if r.get('global_accuracy') is not None}
    render_reference_comparison({server.algorithm: [global_series]}, {server.algorithm: [(server.times, server.round_reports)]}, analysis / 'plots', ['png'])
