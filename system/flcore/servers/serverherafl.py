"""HERAFL adds capacity guidance and an online straggler predictor to FedLoad."""
import numpy as np
from flcore.servers.serverfedload import FedLoad
from flcore.clients.clientherafl import clientHERAFL
from flcore.heterogeneity.linucb_hera import LinUCBBandit
from flcore.heterogeneity.straggler import OnlineStragglerPredictor


class HERAFL(FedLoad):
    client_class = clientHERAFL
    bandit_class = LinUCBBandit

    def __init__(self, args, times):
        super().__init__(args, times)
        self.predictor = OnlineStragglerPredictor()

    def prepare_contexts(self):
        contexts = super().prepare_contexts()
        predictions = [self.predictor.predict_round_time(c) for c in contexts.values()]
        threshold = float(np.mean(predictions))
        for client_id, context in contexts.items():
            initial = self.initial_contexts[client_id]
            context.update(static_cpu_capacity=initial['cpu_capacity'],
                           static_compute_cost=initial['compute_cost'],
                           static_bandwidth=initial['bandwidth'],
                           static_energy_efficiency=initial['energy_efficiency'])
            prediction = self.predictor.predict_straggler_probability(context, threshold)
            context.update(predicted_round_time=prediction['predicted_round_time'],
                           straggler_risk=prediction['straggler_probability'],
                           straggler_threshold=prediction['threshold'])
        return contexts

    def round_timeout(self, contexts):
        seconds = [c['predicted_round_time'] * max(c['round_time_scale'], 1.0) for c in contexts.values()]
        return max(self.base_round_timeout, max(seconds) * 1.35, float(np.mean(seconds)) * 1.5)

    def update_policy(self, client, context, reward):
        super().update_policy(client, context, reward)
        observed = np.clip(client.round_stats['round_time'] / max(context['round_time_scale'], 1.0), 0, 1)
        self.predictor.update(context, observed)
