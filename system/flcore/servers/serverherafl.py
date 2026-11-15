"""HERAFL adds capacity guidance and an online straggler predictor to FedLoad."""
import numpy as np
import json
from pathlib import Path
from flcore.servers.serverfedload import FedLoad
from flcore.clients.clientherafl import clientHERAFL
from flcore.heterogeneity.linucb_hera import LinUCBBandit
from flcore.heterogeneity.straggler import OnlineStragglerPredictor


class HERAFL(FedLoad):
    client_class = clientHERAFL
    bandit_class = LinUCBBandit

    def __init__(self, args, times):
        super().__init__(args, times)
        self.straggler_enabled = getattr(args, 'straggler_predictor', True)
        self.predictor = OnlineStragglerPredictor(temperature=0.12, time_threshold=0.55)
        from flcore.heterogeneity.selection import StratifiedRandom
        self.selector = StratifiedRandom()
        self.selector.selection_fraction = self.join_ratio

    def prepare_contexts(self):
        contexts = super().prepare_contexts()
        predictions = [self.predictor.predict_round_time(c) for c in contexts.values()] if self.straggler_enabled else []
        threshold = float(np.mean(predictions)) if predictions else self.predictor.time_threshold
        for client_id, context in contexts.items():
            initial = self.initial_contexts[client_id]
            context.update(static_cpu_capacity=initial['cpu_capacity'],
                           static_compute_cost=initial['compute_cost'],
                           static_bandwidth=initial['bandwidth'],
                           static_energy_efficiency=initial['energy_efficiency'])
            if not self.straggler_enabled:
                continue
            prediction = self.predictor.predict_straggler_probability(context, threshold)
            context.update(predicted_round_time=prediction['predicted_round_time'],
                           straggler_risk=prediction['straggler_probability'],
                           straggler_threshold=prediction['threshold'])
        return contexts

    def round_timeout(self, contexts):
        if self.wait_for_all or self.aligned_comparison or not self.straggler_enabled:
            return self.base_round_timeout
        seconds = [c['predicted_round_time'] * max(c.get('round_time_scale', 60.0), 1.0) for c in contexts.values()]
        return max(self.base_round_timeout, max(seconds) * 1.35, float(np.mean(seconds)) * 1.5)

    def update_policy(self, client, context, reward):
        super().update_policy(client, context, reward)
        observed = np.clip(client.round_stats['round_time'] / max(context.get('round_time_scale', 1.0), 1.0), 0, 1)
        if self.straggler_enabled:
            self.predictor.update(context, observed)

    def select_clients(self):
        if self.aligned_comparison:
            return super().select_clients()
        self.selector.set_contexts(self.contexts, self.initial_contexts)
        ids = self.selector.select_trainers_for_round([c.id for c in self.clients], {})
        return [self.clients[i] for i in ids]

    def save_policy_results(self):
        super().save_policy_results()
        from flcore.heterogeneity.results import result_directory
        config = result_directory(self) / 'config.json'
        metadata = json.loads(config.read_text())
        source = Path('../dataset') / self.dataset / 'config.json'
        if source.exists():
            metadata['dataset_config'] = json.loads(source.read_text())
        metadata['local_class_evaluation'] = {
            'model': 'trained_local_submodel_before_aggregation',
            'seen_definition': 'class_present_in_actual_local_training_data',
            'absent_definition': 'class_absent_from_local_training_data_not_necessarily_global_training',
            'empty_group_accuracy': None,
            'reward': 'unchanged_all_local_test_samples',
            'train_classes_by_client': {str(c.id): c.training_class_ids() for c in self.clients},
        }
        config.write_text(json.dumps(metadata, indent=2, allow_nan=False))
